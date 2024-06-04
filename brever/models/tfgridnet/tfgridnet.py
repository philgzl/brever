# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# https://github.com/espnet/espnet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

from ...modules import STFT
from ..base import BreverBaseModel, ModelRegistry


@ModelRegistry.register('tfgridnet')
class TFGridNet(BreverBaseModel):
    """Proposed in [1]_ and [2]_.

    References
    ----------
    .. [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
           "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech
           Separation", in IEEE/ACM TASLP, 2023.
    .. [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim and S. Watanabe,
           "TF-GridNet: Making Time-Frequency Domain Models Great Again for
           Monaural Speaker Separation", in Proc. IEEE ICASSP, 2023.
    """

    def __init__(
        self,
        n_srcs: int = 1,
        n_fft: int = 256,
        stride: int = 128,
        window: str = "hann",
        n_layers: int = 6,
        lstm_hidden_units: int = 128,
        attn_n_head: int = 4,
        attn_approx_qk_dim: int = 512,
        emb_dim: int = 32,
        emb_ks: int = 4,
        emb_hs: int = 4,
        activation: str = "PReLU",
        eps: float = 1e-5,
        criterion: str = 'multiresyu',
        optimizer: str = 'Adam',
        learning_rate: float = 0.001,
        grad_clip: float = 1.0,
    ):
        super().__init__(criterion=criterion)

        self.n_srcs = n_srcs
        self.n_layers = n_layers
        n_freqs = n_fft // 2 + 1

        self.stft = STFT(
            frame_length=n_fft,
            hop_length=stride,
            window=window,
            normalized=False,
        )

        n_imics = 2
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                GridNetV2Block(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks,
                                         padding=padding)

        self.optimizer = self.init_optimizer(optimizer, lr=learning_rate)
        self.grad_clip = grad_clip

        self.scheduler = self.init_lr_scheduler()

    def forward(self, x):
        n_samples = x.shape[-1]

        mix_std_ = torch.std(x, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        x = x / mix_std_  # RMS normalization

        batch = self.stft(x)  # [B, M, F, T]
        batch = batch.transpose(2, 3)  # [B, M, T, F]
        batch = torch.cat((batch.real, batch.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        batch = self.conv(batch)  # [B, -1, T, F]

        for i in range(self.n_layers):
            batch = self.blocks[i](batch)  # [B, -1, T, F]

        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]

        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])
        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])
        batch = batch.transpose(2, 3)
        batch = self.stft.backward(batch)[..., :n_samples]

        batch = batch * mix_std_  # reverse the RMS normalization

        return batch

    def loss(self, batch, lengths, use_amp):
        inputs, labels = batch[:, 0], batch[:, 1:]

        # The reference signal for the objective metrics is defined here as the
        # average direct-path signal across left and right channels including
        # early reflections. In the TFGridNet paper, it was defined as the
        # direct-path signal without any reflections at the first microphone.
        labels = labels.mean(axis=-2)

        device = batch.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            outputs = self(inputs)
            loss = self.criterion(outputs, labels, lengths)
        return loss.mean()

    def _enhance(self, x, use_amp):
        device = x.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x = self.forward(x)
        return x

    def update(self, loss, scaler):
        # overwrite to clip gradients
        super().update(loss, scaler, grad_clip=self.grad_clip)

    def on_validate(self, val_loss):
        self.scheduler.step(val_loss)

    def state_dict(self):
        return {
            'net': super().state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict['net'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def init_lr_scheduler(self):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
        )


class GridNetV2Block(nn.Module):
    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="PReLU",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True,
            bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True,
            bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        self.attn_conv_Q = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_Q = AllHeadPReLULayerNormalization4DCF(
            (n_head, E, n_freqs),
            eps=eps
        )

        self.attn_conv_K = nn.Conv2d(emb_dim, n_head * E, 1)
        self.attn_norm_K = AllHeadPReLULayerNormalization4DCF(
            (n_head, E, n_freqs),
            eps=eps
        )

        self.attn_conv_V = nn.Conv2d(emb_dim, n_head * emb_dim // n_head, 1)
        self.attn_norm_V = AllHeadPReLULayerNormalization4DCF(
            (n_head, emb_dim // n_head, n_freqs),
            eps=eps
        )

        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            getattr(nn, activation)(),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape

        olp = self.emb_ks - self.emb_hs
        T = (
            math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs)
            * self.emb_hs + self.emb_ks
        )
        Q = (
            math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs)
            * self.emb_hs + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)  # [B, old_T, old_Q, C]
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, T, Q, C]
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])
            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, Q//I, H]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, Q//I, I*C]
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])  # [BT, Q, C]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, C, Q]
            intra_rnn = F.unfold(
                intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BT, C*I, -1]
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]

            intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]

            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input_  # [B, T, Q, C]

        intra_rnn = intra_rnn.transpose(1, 2)  # [B, Q, T, C]

        # inter RNN
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, Q, T, C]
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, T//I, H]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, T//I, I*C]
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)  # [BQ, T, C]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BQ, C*I, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]

            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]

            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + input_  # [B, Q, T, C]

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)  # [B, C, T, Q]

        inter_rnn = inter_rnn[..., olp:olp + old_T, olp:olp + old_Q]
        batch = inter_rnn

        Q = self.attn_norm_Q(self.attn_conv_Q(batch))
        K = self.attn_norm_K(self.attn_conv_K(batch))
        V = self.attn_norm_V(self.attn_conv_V(batch))
        Q = Q.view(-1, *Q.shape[2:])  # [B*n_head, C, T, Q]
        K = K.view(-1, *K.shape[2:])  # [B*n_head, C, T, Q]
        V = V.view(-1, *V.shape[2:])  # [B*n_head, C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]

        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])  # [B', C*Q, T]

        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.contiguous().view(
            [B, self.n_head * emb_dim, old_T, old_Q]
        )  # [B, C, T, Q])
        batch = self.attn_concat_proj(batch)  # [B, C, T, Q])

        out = batch + inter_rnn
        return out


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError(
                "Expect x to have 4 dimensions, but got {}".format(x.ndim)
            )
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 3
        H, E, n_freqs = input_dimension
        param_size = [1, H, E, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E
        self.n_freqs = n_freqs

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, _ = x.shape
        x = x.view([B, self.H, self.E, T, self.n_freqs])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2, 4)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x

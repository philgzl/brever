# Copyright 2020 Yanxin Hu
# https://github.com/huyanxin/DeepComplexCRN
#
# Copyright 2023 Philippe Gonzalez
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


import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import STFT
from ..base import BreverBaseModel, ModelRegistry
from .complex_batchnorm import ComplexBatchNorm2d


@ModelRegistry.register('dccrn')
class DCCRN(BreverBaseModel):
    def __init__(
        self,
        stft_frame_length: int = 512,
        stft_hop_length: int = 128,
        stft_window: str = 'hann',
        channels: list[int] = [16, 32, 64, 128, 128, 128],
        kernel_size: tuple[int, int] = (5, 2),
        stride: tuple[int, int] = (2, 1),
        padding: tuple[int, int] = (2, 0),
        output_padding: tuple[int, int] = (1, 0),
        lstm_channels: int = 128,
        lstm_layers: int = 2,
        use_complex_batchnorm: bool = False,
        criterion: str = 'snr',
        optimizer: str = 'Adam',
        learning_rate: float = 0.0001,
    ):
        super().__init__(criterion=criterion)

        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels

        self.stft = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            window=stft_window,
        )

        self.mask_net = DCCRNMaskNet(
            input_dim=self.stft.frame_length//2,
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            lstm_channels=lstm_channels,
            lstm_layers=lstm_layers,
            use_complex_batchnorm=use_complex_batchnorm,
        )

        optimizer_cls = getattr(torch.optim, optimizer)
        self.optimizer = optimizer_cls(self.parameters(), lr=learning_rate)

    def optimizers(self):
        return self.optimizer

    def forward(self, x):
        length = x.shape[-1]
        x = self.stft(x)
        # (batch_size, freqs, frames)
        x = x[..., 1:, :]  # remove dc component
        x = torch.stack([x.real, x.imag], dim=1)  # stack along channel dim
        mask = self.mask_net(x)
        x = self.apply_mask(x, mask)
        x = x.squeeze(1)  # remove channel dimension
        x = F.pad(x, (0, 0, 1, 0))  # pad dc component
        x = self.stft.backward(x)
        x = x[..., :length]
        return x

    def apply_mask(self, x, mask):
        in_real, in_imag = torch.chunk(x, 2, dim=1)
        in_mag = (in_real**2 + in_imag**2).sqrt()
        in_phase = torch.atan2(in_imag, in_real)
        mask_real, mask_imag = torch.chunk(mask, 2, dim=1)
        mask_mag = (mask_real**2 + mask_imag**2 + 1e-7).sqrt().tanh()
        mask_real = mask_real + (mask_real == 0)*1e-7
        mask_phase = torch.atan2(mask_imag, mask_real)
        out_mag = in_mag*mask_mag
        out_phase = in_phase + mask_phase
        out_real = out_mag*out_phase.cos()
        out_imag = out_mag*out_phase.sin()
        return torch.complex(out_real, out_imag)

    def transform(self, sources):
        assert sources.shape[0] == 2  # mixture, foreground
        sources = sources.mean(axis=-2)  # make monaural
        return sources

    def _step(self, batch, lengths, use_amp):
        inputs, labels = batch[:, 0], batch[:, 1]
        device = batch.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            outputs = self(inputs)
            loss = self.criterion(outputs, labels, lengths)
        return loss.mean()

    def train_step(self, batch, lengths, use_amp, scaler):
        self.optimizer.zero_grad()
        loss = self._step(batch, lengths, use_amp)
        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        scaler.step(self.optimizer)
        scaler.update()
        return loss

    def val_step(self, batch, lengths, use_amp):
        return self._step(batch, lengths, use_amp)

    def _enhance(self, x, use_amp):
        x = x.mean(axis=-2)  # (batch_size, length)
        device = x.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x = self.forward(x)  # (batch_size, length)
        return x

    @property
    def latency(self):
        _, kernel_size = self.kernel_size
        _, stride = self.stride
        layers = len(self.channels)
        enc_dec_lat = (kernel_size - 1)*sum(stride**i for i in range(layers))
        return self.stft.frame_length + enc_dec_lat*self.stft.hop_length


class DCCRNMaskNet(nn.Module):
    def __init__(
        self,
        input_dim,
        channels: list[int] = [16, 32, 64, 128, 128, 128],
        kernel_size: tuple[int, int] = (5, 2),
        stride: tuple[int, int] = (2, 1),
        padding: tuple[int, int] = (2, 0),
        output_padding: tuple[int, int] = (1, 0),
        lstm_channels: int = 128,
        lstm_layers: int = 2,
        use_complex_batchnorm: bool = False,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        for i in range(len(channels)):
            self.encoder.append(EncoderBlock(
                in_channels=1 if i == 0 else channels[i-1],
                out_channels=channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_complex_batchnorm=use_complex_batchnorm,
            ))

        self.decoder = nn.ModuleList()
        for i in range(len(channels)-1, -1, -1):
            self.decoder.append(DecoderBlock(
                in_channels=channels[i]*2,
                out_channels=1 if i == 0 else channels[i-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                use_complex_batchnorm=use_complex_batchnorm,
                output_padding=output_padding,
                norm=i != 0,
                activation=i != 0,
            ))

        lstm_input_size = self.calc_lstm_input_size(
            input_dim, channels, kernel_size[0], stride[0], padding[0]
        )
        self.lstm = LSTMBlock(
            input_size=lstm_input_size,
            hidden_size=lstm_channels,
            num_layers=lstm_layers,
        )

    def calc_lstm_input_size(
        self, input_dim, channels, kernel_size, stride, padding,
    ):
        enc_out_dim = input_dim
        for channel in channels:
            enc_out_dim = (enc_out_dim + 2*padding - kernel_size)//stride + 1
        return channels[-1]*enc_out_dim

    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        # permute to (batch_size, frames, channels, freqs)
        x = x.permute(0, x.ndim-1, *range(1, x.ndim-1))
        # reshape to (batch_size, frames, channels*freqs) for lstm, pass to
        # lstm and reshape back to original shape
        x = self.lstm(x.reshape(*x.shape[:2], -1)).reshape(*x.shape)
        # permute back to (batch_size, channels, freqs, time)
        x = x.permute(0, *range(2, x.ndim), 1)
        # pass to decoder
        for decoder_block, encoder_output in zip(
            self.decoder, reversed(encoder_outputs),
        ):
            real, imag = x.chunk(2, dim=1)
            skip_real, skip_imag = encoder_output.chunk(2, dim=1)
            x = torch.cat([real, skip_real, imag, skip_imag], dim=1)
            x = decoder_block(x)
        return x


class ComplexWrapper(nn.Module):
    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.module_real = module_cls(*args, **kwargs)
        self.module_imag = module_cls(*args, **kwargs)

    def forward(self, x):
        in_real, in_imag = torch.chunk(x, 2, dim=1)
        out_real = self.module_real(in_real) - self.module_imag(in_imag)
        out_imag = self.module_real(in_imag) + self.module_imag(in_real)
        return torch.cat([out_real, out_imag], dim=1)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_complex_batchnorm):
        super().__init__()
        self.conv = ComplexWrapper(
            module_cls=nn.Conv2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_complex_batchnorm:
            self.norm = ComplexBatchNorm2d(out_channels)
        else:
            self.norm = nn.BatchNorm2d(2*out_channels)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, use_complex_batchnorm, output_padding, norm=True,
                 activation=True):
        super().__init__()
        self.conv = ComplexWrapper(
            module_cls=nn.ConvTranspose2d,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )
        self.norm, self.activation = None, None
        if norm:
            if use_complex_batchnorm:
                self.norm = ComplexBatchNorm2d(out_channels)
            else:
                self.norm = nn.BatchNorm2d(2*out_channels)
        if activation:
            self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = ComplexLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.linear_r = nn.Linear(hidden_size, input_size)
        self.linear_i = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        real, imag = torch.chunk(x, 2, dim=-1)
        real, imag = self.lstm(real, imag)
        real, imag = self.linear_r(real), self.linear_i(imag)
        return torch.cat([real, imag], dim=-1)


class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SingleLayerComplexLSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=False,
            ))

    def forward(self, real, imag):
        for layer in self.layers:
            real, imag = layer(real, imag)
        return real, imag


class SingleLayerComplexLSTM(ComplexWrapper):
    """
    Directly calling `ComplexWrapper` with `module_cls=nn.LSTM` leads to a
    wrong implementation of a multi-layer complex LSTM, since each layer needs
    to be wrapped. A single-layer complex LSTM is first defined here by
    sub-classing `ComplexWrapper`.

    Also `nn.LSTM` has multiple outputs so we need to rewrite `forward`.
    """
    def __init__(self, *args, **kwargs):
        if len(args) > 2 or 'num_layers' in kwargs:
            raise ValueError(f'{self.__class__.__name__} does not support '
                             'num_layers argument')
        super().__init__(nn.LSTM, *args, **kwargs)

    def forward(self, real, imag):
        real_real, _ = self.module_real(real)
        imag_imag, _ = self.module_imag(imag)
        real_imag, _ = self.module_real(imag)
        imag_real, _ = self.module_imag(real)
        real = real_real - imag_imag
        imag = real_imag + imag_real
        return real, imag

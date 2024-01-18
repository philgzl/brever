# https://github.com/naplab/Conv-TasNet (c) 2019 Yi Luo
# Modifications (c) 2023 Philippe Gonzalez
#
# https://github.com/naplab/Conv-TasNet is licensed under a
# Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work. If not, see <https://creativecommons.org/licenses/by-nc-sa/3.0/>.


import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import CausalLayerNorm
from ..base import BreverBaseModel, ModelRegistry


@ModelRegistry.register('convtasnet')
class ConvTasNet(BreverBaseModel):
    def __init__(
        self,
        filters: int = 512,
        filter_length: int = 32,
        bottleneck_channels: int = 128,
        hidden_channels: int = 512,
        skip_channels: int = 128,
        kernel_size: int = 3,
        layers: int = 8,
        repeats: int = 3,
        output_sources: int = 1,
        causal: bool = False,
        criterion: str = 'snr',
        optimizer: str = 'Adam',
        learning_rate: float = 0.001,
        grad_clip: float = 5.0,
    ):
        super().__init__(criterion=criterion)

        self.encoder = Encoder(filters, filter_length)
        self.decoder = Decoder(filters, filter_length)
        self.tcn = TCN(
            input_channels=filters,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            layers=layers,
            repeats=repeats,
            sources=output_sources,
            causal=causal,
        )

        self.optimizer = self.init_optimizer(optimizer, lr=learning_rate)
        self.grad_clip = grad_clip

    def forward(self, x):
        length = x.shape[-1]
        x = self.encoder(x)
        masks = self.tcn(x)
        x = self.decoder(x, masks)
        x = x[:, :, :length]
        return x

    def transform(self, sources):
        sources = sources.mean(axis=-2)  # make monaural
        return sources

    def loss(self, batch, lengths, use_amp):
        inputs, labels = batch[:, 0], batch[:, 1:]
        device = batch.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            outputs = self(inputs)
            loss = self.criterion(outputs, labels, lengths)
        return loss.mean()

    def update(self, loss, scaler):
        # overwrite to clip gradients
        super().update(loss, scaler, grad_clip=self.grad_clip)

    def _enhance(self, x, use_amp):
        x = x.mean(axis=-2)  # (batch_size, length)
        device = x.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x = self.forward(x)  # (batch_size, sources, length)
        return x


class Encoder(nn.Module):
    def __init__(self, filters, filter_length, stride=None):
        super().__init__()
        if stride is None:
            stride = filter_length//2
        self.filter_length = filter_length
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=filter_length,
            stride=stride,
            bias=False,
        )

    def pad(self, x):
        batch_size, length = x.shape
        # pad right to obtain integer number of frames
        padding = (self.filter_length - length) % self.stride
        x = F.pad(x, (0, padding))
        return x

    def forward(self, x):
        x = self.pad(x)
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, filters, filter_length, stride=None):
        super().__init__()
        if stride is None:
            stride = filter_length//2
        self.filter_length = filter_length
        self.stride = stride
        self.trans_conv = nn.ConvTranspose1d(
            in_channels=filters,
            out_channels=1,
            kernel_size=filter_length,
            stride=stride,
            bias=False,
        )

    def forward(self, x, masks):
        batch_size, sources, channels, length = masks.shape
        x = x.unsqueeze(1)
        x = x*masks
        x = x.view(batch_size*sources, channels, length)
        x = self.trans_conv(x)
        x = x.view(batch_size, sources, -1)
        return x


class TCN(nn.Module):
    def __init__(self, input_channels, bottleneck_channels, hidden_channels,
                 skip_channels, kernel_size, layers, repeats, sources, causal):
        super().__init__()
        self.sources = sources
        self.layer_norm = init_norm(causal, input_channels)
        self.bottleneck_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
        )
        self.conv_blocks = nn.ModuleList()
        for b in range(repeats):
            for i in range(layers):
                dilation = 2**i
                last = b == repeats-1 and i == layers-1
                self.conv_blocks.append(
                    Conv1DBlock(
                        input_channels=bottleneck_channels,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        dilation=dilation,
                        kernel_size=kernel_size,
                        causal=causal,
                        last=last,
                    )
                )
        self.prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=input_channels*sources,
            kernel_size=1,
        )

    def forward(self, x):
        batch_size, channels, length = x.shape
        x = self.layer_norm(x)
        x = self.bottleneck_conv(x)
        skip_sum = 0
        for conv_block in self.conv_blocks:
            x, skip = conv_block(x)
            skip_sum += skip
        x = self.prelu(skip_sum)
        x = self.output_conv(x)
        x = torch.sigmoid(x)
        return x.view(batch_size, self.sources, channels, length)


class Conv1DBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, skip_channels,
                 kernel_size, dilation, causal, last=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=1,
        )
        self.d_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
        )
        # the output of the residual convolution in the last block is not used
        if last:
            self.res_conv = None
        else:
            self.res_conv = nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=input_channels,
                kernel_size=1,
            )
        self.skip_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=skip_channels,
            kernel_size=1,
        )
        self.norm_1 = init_norm(causal, hidden_channels)
        self.norm_2 = init_norm(causal, hidden_channels)
        self.prelu_1 = nn.PReLU()
        self.prelu_2 = nn.PReLU()

    def forward(self, input_):
        x = self.conv(input_)
        x = self.prelu_1(x)
        x = self.norm_1(x)
        padding = (self.kernel_size - 1) * self.dilation
        if self.causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left
        x = F.pad(x, (padding_left, padding_right))
        x = self.d_conv(x)
        x = self.prelu_2(x)
        x = self.norm_2(x)
        if self.res_conv is None:
            output = None
        else:
            output = input_ + self.res_conv(x)
        skip = self.skip_conv(x)
        return output, skip


def init_norm(causal, dim):
    if causal:
        module = CausalLayerNorm(num_channels=dim, time_dim=-1, eps=1e-8)
    else:
        module = nn.GroupNorm(num_channels=dim, num_groups=1,  eps=1e-8)
    return module

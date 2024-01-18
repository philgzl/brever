import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import Resample
from ..base import BreverBaseModel


class DiffusionUNet(BreverBaseModel):
    # inherit BreverBaseModel to get compile method
    def __init__(
        self,
        num_freqs,  # same as img_resolution in NVlabs/edm
        base_channels,
        channel_mult,
        num_res_blocks,
        noise_channels,
        emb_channels,
        fir_kernel,
        attn_resolutions,
        attn_bottleneck,
        in_channels=4,
        out_channels=2,
    ):
        # Encoder and decoder type according to different sources:
        #
        # source              | encoder_type | decoder_type
        # ------------------- | ------------ | ------------
        # yang-song/score_sde | 'residual'   | 'standard'
        # NVlabs/edm          | 'residual'   | 'standard'
        # sp-uhh/sgmse        | 'skip'       | 'skip'
        #
        # Type description:
        # - 'standard': no progressive path, though there is still a final norm
        # and conv layer that adapts the number of output channels
        # - 'skip': TODO
        # - 'residual': TODO
        #
        # Notes:
        # - yang-song/score_sde uses a Combiner module in the skip connections
        # from the progressive path to the encoder. The Combiner can either
        # concatenate ('cat') or sum ('sum') the channels after channel
        # adaptation. However the 'sum' option is used in all experiments.
        # Similarly, combine_method is set to 'sum' by default in sp-uhh/sgmse.
        # Consequently, NVlabs/edm removes the option and hard codes the
        # summation.
        super().__init__()

        self.resampler = Resample(fir_kernel, buffer_padding=True)

        self.emb = NoiseEmbedding(noise_channels, emb_channels)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.output_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.encoder = nn.ModuleList()
        self.prog_downs = nn.ModuleList()

        c_out = base_channels
        skip_channels = [base_channels]
        for i, mult in enumerate(channel_mult):
            res = num_freqs >> i
            encoder_block = nn.ModuleList()
            for j in range(num_res_blocks):
                c_in = c_out
                c_out = base_channels*mult
                encoder_block.append(ResidualBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    emb_channels=emb_channels,
                    attention=res in attn_resolutions,
                ))
                skip_channels.append(c_out)
            if i == len(channel_mult) - 1:
                self.prog_downs.append(None)
            else:
                encoder_block.append(ResidualBlock(
                    in_channels=c_out,
                    out_channels=c_out,
                    emb_channels=emb_channels,
                    resampler=self.resampler,
                    up_or_down='down',
                ))
                skip_channels.append(c_out)
                self.prog_downs.append(ProgressiveDown(
                    in_channels=in_channels,
                    out_channels=c_out,
                    resampler=self.resampler,
                ))
            self.encoder.append(encoder_block)

        self.bottleneck_block_1 = ResidualBlock(
            in_channels=c_out,
            out_channels=c_out,
            emb_channels=emb_channels,
            attention=attn_bottleneck,
        )
        self.bottleneck_block_2 = ResidualBlock(
            in_channels=c_out,
            out_channels=c_out,
            emb_channels=emb_channels,
        )

        self.decoder = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()
        self.prog_ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult))):
            res = num_freqs >> i
            if i == len(channel_mult) - 1:
                self.upsampling_blocks.append(None)
            else:
                self.upsampling_blocks.append(ResidualBlock(
                    in_channels=c_out,
                    out_channels=c_out,
                    emb_channels=emb_channels,
                    resampler=self.resampler,
                    up_or_down='up',
                ))
            decoder_block = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                c_in = c_out + skip_channels.pop()
                c_out = base_channels*mult
                decoder_block.append(ResidualBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    emb_channels=emb_channels,
                    attention=res in attn_resolutions,
                ))
            self.decoder.append(decoder_block)
            self.prog_ups.append(ProgressiveUp(
                in_channels=c_out,
                out_channels=in_channels,
                resampler=self.resampler if (i != len(channel_mult) - 1) else None,  # noqa: E501
            ))

    def forward(self, x, sigma):
        emb = self.emb(sigma)
        prog = x
        x = self.input_conv(x)
        skips = [x]
        for blocks in zip(self.encoder, self.prog_downs):
            encoder_block, prog_block = blocks
            for res_block in encoder_block:
                x = res_block(x, emb)
                skips.append(x)
            if prog_block is not None:
                x, prog = prog_block(x, prog)
                skips[-1] = x

        x = self.bottleneck_block_1(x, emb)
        x = self.bottleneck_block_2(x, emb)

        prog = None
        for blocks in zip(self.decoder, self.prog_ups, self.upsampling_blocks):
            decoder_block, prog_block, upsampling_block = blocks
            if upsampling_block is not None:
                x = upsampling_block(x, emb)
            for res_block in decoder_block:
                x = torch.cat([x, skips.pop()], dim=1)
                x = res_block(x, emb)
            prog = prog_block(x, prog)

        prog = self.output_conv(prog)
        return prog


class NoiseEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fourier_proj = GaussianFourierProjection(in_channels)
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.view(-1)
        x = self.fourier_proj(x)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        x = F.silu(x)
        return x


class ProgressiveDown(nn.Module):
    def __init__(self, in_channels, out_channels, resampler):
        super().__init__()
        self.resampler = resampler
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, prog):
        prog = self.resampler(prog, 'down')
        h = self.conv(prog)
        return x + h, prog


class ProgressiveUp(nn.Module):
    def __init__(self, in_channels, out_channels, resampler=None):
        super().__init__()
        self.resampler = resampler
        self.norm = GroupNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x, prog):
        if self.resampler is not None:
            prog = self.resampler(prog, 'up')
        h = self.norm(x)
        h = F.silu(h)
        h = self.conv(h)
        prog = h if prog is None else prog + h
        return prog


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        resampler=None,
        up_or_down='none',
        dropout=0.1,
        skip_scale=0.5**0.5,
        attention=False
    ):
        super().__init__()
        self.skip_scale = skip_scale
        self.norm_1 = GroupNorm(in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.linear = nn.Linear(emb_channels, out_channels)
        self.norm_2 = GroupNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels or up_or_down != 'none':
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = None
        self.resampler = resampler
        self.up_or_down = up_or_down
        if attention:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = None

    def forward(self, x, emb):
        h = self.norm_1(x)
        h = F.silu(h)
        if self.resampler is not None:
            h = self.resampler(h, self.up_or_down)
            x = self.resampler(x, self.up_or_down)
        h = self.conv_1(h)
        h += self.linear(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm_2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv_2(h)
        if self.skip_conv is not None:
            x = self.skip_conv(x)
        x = (x + h)*self.skip_scale
        if self.attn is not None:
            x = self.attn(x)
            x = x*self.skip_scale
        return x


class AttentionBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = GroupNorm(num_channels)
        self.conv_query = nn.Conv2d(num_channels, num_channels, 1)
        self.conv_key = nn.Conv2d(num_channels, num_channels, 1)
        self.conv_value = nn.Conv2d(num_channels, num_channels, 1)
        self.conv_out = nn.Conv2d(num_channels, num_channels, 1)

    def forward(self, x):
        N, _, H, W = x.shape

        x_norm = self.norm(x)
        queries = self.conv_query(x_norm)
        keys = self.conv_key(x_norm)
        values = self.conv_value(x_norm)

        queries = queries.reshape(N, -1, H*W).permute(0, 2, 1)
        keys = keys.reshape(N, -1, H*W)
        weights = torch.bmm(queries, keys/keys.shape[1]**0.5).softmax(dim=-1)

        values = values.reshape(N, -1, H*W).permute(0, 2, 1)
        attentions = torch.bmm(weights, values)

        attentions = attentions.permute(0, 2, 1).reshape(N, -1, H, W)
        return x + self.conv_out(attentions)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4,
                 eps=1e-6):
        super().__init__(
            num_groups=min(num_groups, num_channels//min_channels_per_group),
            num_channels=num_channels,
            eps=eps,
        )


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size, scale=16.0):
        super().__init__()
        b = torch.randn(embedding_size//2)*scale
        self.register_buffer('b', b)

    def forward(self, x):
        x = 2*math.pi*x.outer(self.b)
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x

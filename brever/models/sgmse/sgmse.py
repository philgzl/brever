import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expi

from ...modules import STFT, Resample
from ...registry import Registry
from ..base import BreverBaseModel, ModelRegistry

SDERegistry = Registry('sde')
SolverRegistry = Registry('solver')


@ModelRegistry.register('sgmse')
class SGMSE(BreverBaseModel):
    def __init__(
        self,
        stft_frame_length: int = 512,
        stft_hop_length: int = 128,
        stft_window: str = 'hann',
        stft_compression_factor: float = 0.5,
        stft_scale_factor: float = 0.15,
        stft_discard_nyquist: bool = True,
        sde_name: str = 'richter-ouve',
        sde_stiffness: float = 1.5,
        sde_ve_sigma_min: float = 0.05,
        sde_ve_sigma_max: float = 0.5,
        sde_vp_beta_min: float = 0.01,
        sde_vp_beta_max: float = 1.0,
        sde_cosine_lambda_min: float = -12.0,
        sde_cosine_lambda_max: float = float('inf'),
        sde_cosine_shift: float = 3.0,
        sde_cosine_beta_clamp: float = 10.0,
        solver_name: str = 'pc',
        solver_num_steps: int = 16,
        solver_edm_schurn: float = 0.0,
        solver_edm_smin: float = 0.0,
        solver_edm_smax: float = float('inf'),
        solver_edm_snoise: float = 1.0,
        solver_pc_corrector_steps: int = 1,
        solver_pc_corrector_snr: float = 0.5,
        net_base_channels: int = 128,
        net_channel_mult: list[int] = [1, 2, 2, 2],
        net_num_res_blocks: int = 1,
        net_noise_channels: int = 256,
        net_emb_channels: int = 512,
        net_fir_kernel: list[int] = [1, 3, 3, 1],
        net_attn_resolutions: list[int] = [0],
        net_attn_bottleneck: bool = True,
        preconditioning_cskip: str = 'richter',
        preconditioning_cout: str = 'richter',
        preconditioning_cin: str = 'richter',
        preconditioning_cnoise: str = 'richter',
        preconditioning_cshift: str = 'richter',
        preconditioning_weight: str = 'richter',
        preconditioning_sigma_data: float = 0.1,
        t_eps: float = 0.01,
        criterion: str = 'mse',
        optimizer: str = 'Adam',
        learning_rate: float = 0.0001,
    ):
        super().__init__(criterion=criterion)

        self.stft = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            window=stft_window,
            compression_factor=stft_compression_factor,
            scale_factor=stft_scale_factor,
            normalized=False,
        )
        self.stft_discard_nyquist = stft_discard_nyquist

        sde_cls = SDERegistry.get(sde_name)
        self.sde = sde_cls(
            stiffness=sde_stiffness,
            sigma_min=sde_ve_sigma_min,
            sigma_max=sde_ve_sigma_max,
            beta_min=sde_vp_beta_min,
            beta_max=sde_vp_beta_max,
            lambda_min=sde_cosine_lambda_min,
            lambda_max=sde_cosine_lambda_max,
            shift=sde_cosine_shift,
            beta_clamp=sde_cosine_beta_clamp,
        )

        solver_cls = SolverRegistry.get(solver_name)
        self.solver = solver_cls(
            num_steps=solver_num_steps,
            schurn=solver_edm_schurn,
            smin=solver_edm_smin,
            smax=solver_edm_smax,
            snoise=solver_edm_snoise,
            corrector_steps=solver_pc_corrector_steps,
            corrector_snr=solver_pc_corrector_snr,
        )

        raw_net = DiffusionUNet(
            num_freqs=stft_frame_length//2,
            base_channels=net_base_channels,
            channel_mult=net_channel_mult,
            num_res_blocks=net_num_res_blocks,
            noise_channels=net_noise_channels,
            emb_channels=net_emb_channels,
            fir_kernel=net_fir_kernel,
            attn_resolutions=net_attn_resolutions,
            attn_bottleneck=net_attn_bottleneck,
        )
        self.model = Preconditioning(
            raw_net=raw_net,
            sde=self.sde,
            cskip=preconditioning_cskip,
            cout=preconditioning_cout,
            cin=preconditioning_cin,
            cnoise=preconditioning_cnoise,
            cshift=preconditioning_cshift,
            weight=preconditioning_weight,
            sigma_data=preconditioning_sigma_data,
        )

        self.t_eps = t_eps

        optimizer_cls = getattr(torch.optim, optimizer)
        self.optimizer = optimizer_cls(self.parameters(), lr=learning_rate)

    def optimizers(self):
        return self.optimizer

    def transform(self, sources):
        # TODO: adapt for multiple output sources, see "Diffusion-based
        # Generative Speech Source Separation" by R. Scheibler et al.
        # in the meantime assert there is only one output source
        assert sources.shape[0] == 2  # mixture, foreground
        sources = sources.mean(axis=-2)  # make monaural
        sources /= sources[0].abs().max()  # normalize
        sources = self.stft(sources)
        if self.stft_discard_nyquist:
            sources = sources[..., :-1, :]
        return sources

    def forward(self, x, y, sigma, t):
        return self.model(x, y, sigma, t)

    def _step(self, batch, lengths, use_amp):
        y, x_0 = batch[:, 0], batch[:, 1]  # y is noisy, x_0 is clean
        y, x_0 = y.unsqueeze(1), x_0.unsqueeze(1)
        t = torch.rand(x_0.shape[0], 1, 1, 1, device=y.device) \
            * (1 - self.t_eps) + self.t_eps
        sigma = self.sde.sigma(t)
        n = sigma*torch.randn_like(x_0)
        weight = self.model.weight(sigma)
        device = batch.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            d = self(x_0 - y + n, y, sigma, t)
            loss = self.criterion(d, x_0 - y, lengths, weight=weight)
        return loss.mean()

    def train_step(self, batch, lengths, use_amp, scaler):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._step(batch, lengths, use_amp)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        return loss

    def val_step(self, batch, lengths, use_amp):
        return self._step(batch, lengths, use_amp)

    def _enhance(self, x, use_amp):
        length = x.shape[-1]
        x = x.mean(axis=-2, keepdims=True)  # (batch_size, channels=1, length)
        norm = x.abs().amax(axis=-1, keepdims=True)
        x /= norm
        x = self.stft(x)
        if self.stft_discard_nyquist:
            x = x[..., :-1, :]
        device = x.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x, nfe = self.solver(self.sde, x, self.model)
        x = F.pad(x, (0, 0, 0, 1))  # pad nyquist frequency
        x = self.stft.backward(x)
        x *= norm
        return x[..., :length].squeeze(1)

    def compile(self, *args, **kwargs):
        # overwrite default compile method to compile underlying model only
        self.model.net.compile(*args, **kwargs)


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
        """
        Encoder and decoder type according to different sources:

        source              | encoder_type | decoder_type
        ------------------- | ------------ | ------------
        yang-song/score_sde | 'residual'   | 'standard'
        NVlabs/edm          | 'residual'   | 'standard'
        sp-uhh/sgmse        | 'skip'       | 'skip'

        Type description:
        - 'standard': no progressive path, though there is still a final norm
        and conv layer that adapts the number of output channels
        - 'skip': TODO
        - 'residual': TODO

        Notes:
        - yang-song/score_sde uses a Combiner module in the skip connections
        from the progressive path to the encoder. The Combiner can either
        concatenate ('cat') or sum ('sum') the channels after channel
        adaptation. However the 'sum' option is used in all experiments.
        Similarly, combine_method is set to 'sum' by default in sp-uhh/sgmse.
        Consequently, NVlabs/edm removes the option and hard codes the
        summation.
        """
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


class Preconditioning(nn.Module):
    def __init__(self, raw_net, sde, cskip, cout, cin, cshift, cnoise, weight,
                 sigma_data):
        super().__init__()
        self.net = raw_net
        self.sde = sde

        _preconditionings = {
            'richter': dict(
                cskip=lambda sigma: 1,
                cout=lambda sigma, scaling, t: - scaling * sigma**2 / t,
                cin=lambda sigma, scaling: scaling,
                cshift=lambda y, cin, scaling: y,
                cnoise=lambda sigma, t: t.log(),
                weight=lambda sigma: 1 / sigma**2,
            ),
            'edm': dict(
                cskip=lambda sigma: sigma_data**2 / (sigma**2 + sigma_data**2),
                cout=lambda sigma, scaling, t: sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5,  # noqa: E501
                cin=lambda sigma, scaling: 1 / (sigma**2 + sigma_data**2)**0.5,
                cshift=lambda y, cin, scaling: 0,
                cnoise=lambda sigma, t: sigma.log()/4,
                weight=lambda sigma: (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2,  # noqa: E501
            ),
            'edm-scaled-shift': dict(
                cshift=lambda y, cin, scaling: cin * y / scaling,
            )
        }

        for arg in ['cskip', 'cout', 'cin', 'cshift', 'cnoise', 'weight']:
            val = eval(arg)  # richter or edm
            if val not in _preconditionings:
                raise ValueError(f'Invalid preconditioning {arg}: {val}')
            setattr(self, arg, _preconditionings[val][arg])

    def forward(self, x, y, sigma, t):
        scaling = self.sde.s(t)

        cskip = self.cskip(sigma)
        cout = self.cout(sigma, scaling, t)
        cin = self.cin(sigma, scaling)
        cshift = self.cshift(y, cin, scaling)
        cnoise = self.cnoise(sigma, t)

        x_in = cin*x + cshift

        net_in = torch.cat([x_in.real, x_in.imag, y.real, y.imag], dim=1)
        net_out = self.net(net_in, cnoise)
        net_out = torch.complex(net_out[:, 0], net_out[:, 1]).unsqueeze(1)

        return cskip*x + cout*net_out

    def score(self, x, y, sigma, t):
        return (self(x, y, sigma, t) - x) / (self.sde.s(t) * sigma**2)


class _BaseSDE:
    def probability_flow(self, x, y, score, t):
        return self.f(x, y, t) - 0.5*self.g(t)**2*score

    def reverse_step(self, x, y, score, t, dt):
        noise = self.g(t)*(-dt)**0.5*torch.randn(x.shape, device=x.device)
        return (self.f(x, y, t) - self.g(t)**2*score)*dt + noise

    def prior(self, y):
        t = torch.tensor(1, device=y.device)
        sigma = self.s(t)*self.sigma(t)
        return y + sigma*torch.randn_like(y)

    def s(self, t):
        raise NotImplementedError

    def sigma(self, t):
        raise NotImplementedError

    def f(self, x, y, t):
        raise NotImplementedError

    def g(self, t):
        raise NotImplementedError

    def sigma_inv(self, t):
        raise NotImplementedError


class _BaseOUVESDE(_BaseSDE):
    def __init__(self, stiffness, sigma_min, sigma_max, **kwargs):
        self.stiffness = stiffness
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._sigma_p = sigma_max/sigma_min
        self._log_sigma_p = math.log(sigma_max/sigma_min)

    def s(self, t):
        return (-self.stiffness * t).exp()

    def f(self, x, y, t):
        return self.stiffness * (y - x)


@SDERegistry.register('richter-ouve')
class RichterOUVESDE(_BaseOUVESDE):
    def sigma(self, t):
        return self.sigma_min * (
            ((self._sigma_p**t / self.s(t))**2 - 1)
            / (1 + self.stiffness / self._log_sigma_p)
        )**0.5

    def g(self, t):
        return self.sigma_min * self._sigma_p**t * (2 * self._log_sigma_p)**0.5

    def sigma_inv(self, sigma):
        return 0.5 * (
            1 + (1 + self.stiffness/self._log_sigma_p)
            * (sigma / self.sigma_min)**2
        ).log() / (self.stiffness + self._log_sigma_p)


@SDERegistry.register('brever-ouve')
class BreverOUVESDE(_BaseOUVESDE):
    def sigma(self, t):
        return self.sigma_min * (self._sigma_p**(2*t) - 1)**0.5

    def g(self, t):
        return self.s(t) * self.sigma_min * self._sigma_p**t \
            * (2 * self._log_sigma_p)**0.5

    def sigma_inv(self, sigma):
        return 0.5 * ((sigma/self.sigma_min)**2 + 1).log() / self._log_sigma_p


class _BaseVPSDE(_BaseSDE):
    def s(self, t):
        return (-self.stiffness * t).exp() / (1 + self.sigma(t)**2)**0.5

    def f(self, x, y, t):
        return (self.stiffness + 0.5 * self.beta(t)) * (y - x)

    def g(self, t):
        return (-self.stiffness * t).exp() * self.beta(t)**0.5


@SDERegistry.register('brever-ouvp')
class BreverOUVPSDE(_BaseVPSDE):
    def __init__(self, stiffness, beta_min, beta_max, **kwargs):
        self.stiffness = stiffness
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._beta_d = beta_max - beta_min

    def beta(self, t):
        return self.beta_min + self._beta_d * t

    def sigma(self, t):
        return ((0.5*self._beta_d*t**2 + self.beta_min*t).exp() - 1)**0.5

    def sigma_inv(self, sigma):
        return (
            (self.beta_min**2 + 2*self._beta_d*(sigma ** 2 + 1).log())**0.5
            - self.beta_min
        ) / self._beta_d


@SDERegistry.register('brever-oucosine')
class BreverOUCosineSDE(_BaseVPSDE):
    def __init__(self, stiffness, lambda_min, lambda_max, shift, beta_clamp,
                 **kwargs):
        self.stiffness = stiffness
        self.shift = shift
        self.lambda_min = lambda_min + shift
        self.lambda_max = lambda_max + shift
        self.t_min = self.lambda_inv(self.lambda_min)
        self.t_max = self.lambda_inv(self.lambda_max)
        self.t_d = self.t_min - self.t_max
        self.beta_clamp = beta_clamp

    def lambda_(self, t):
        return -2 * (math.pi * t / 2).tan().log() + self.shift

    def lambda_inv(self, lambda_):
        if isinstance(lambda_, torch.Tensor):
            return 2 / math.pi * ((-lambda_ + self.shift)/2).exp().atan()
        else:
            return 2 / math.pi * math.atan(math.exp((-lambda_ + self.shift)/2))

    def lambda_tilde(self, t):
        return self.lambda_(self.t_max + self.t_d*t)

    def lambda_tilde_inv(self, lambda_):
        return (self.lambda_inv(lambda_) - self.t_max) / self.t_d

    def beta(self, t):
        pi_t_half = math.pi * (self.t_max + self.t_d*t) / 2
        return (
            math.pi * self.t_d
            / pi_t_half.cos()**2
            * pi_t_half.tan()
            / (math.exp(self.shift) + pi_t_half.tan()**2)
        ).clamp(max=self.beta_clamp)

    def sigma(self, t):
        return (-self.lambda_tilde(t) / 2).exp()

    def sigma_inv(self, sigma):
        return self.lambda_tilde_inv(-2 * sigma.log())


@SDERegistry.register('bbed')
class BBEDSDE(_BaseSDE):
    def __init__(self, c=0.01, k=10, **kwargs):
        self.c = c
        self.k = k
        self._k2 = k**2
        self._logk2 = 2*math.log(k)
        self.t_max = 0.999

    def clamp(self, t):
        return t * self.t_max

    def s(self, t):
        return 1 - self.clamp(t)

    def f(self, x, y, t):
        return (y - x) / (1 - self.clamp(t))

    def g(self, t):
        return self.c**0.5 * self.k**self.clamp(t)

    def sigma(self, t):
        t_clamp = self.clamp(t)
        return (
            self.c * (
                self._k2 * self._logk2 * (
                    expi((t_clamp.cpu() - 1)*self._logk2).to(t.device)
                    - expi(-self._logk2)
                )
                - self._k2**t_clamp / (t_clamp - 1) - 1
            )
        )**0.5


@SolverRegistry.register('edm')
class EDMSolver:
    def __init__(self, num_steps, schurn, smin, smax, snoise, **kwargs):
        self.num_steps = num_steps
        self.schurn = schurn
        self.smin = smin
        self.smax = smax
        self.snoise = snoise
        self._gamma = min(schurn/num_steps, 2**0.5 - 1)

    def __call__(self, sde, y, model):
        t = torch.linspace(1, 0, self.num_steps+1, device=y.device)
        sigma = sde.sigma(t)
        x = sde.prior(y)

        for i in range(self.num_steps):
            # stochastic step
            eps = self.snoise*torch.randn_like(x)
            gamma = self._gamma if self.smin <= sigma[i] <= self.smax else 0
            sigma_hat = sigma[i]*(1 + gamma)
            t_hat = sde.sigma_inv(sigma_hat)
            x_hat = sde.s(t_hat)/sde.s(t[i])*(x - y) + y \
                + sde.s(t_hat)*(sigma_hat**2 - sigma[i]**2)**0.5*eps

            # deterministic step
            x_tilde = (x_hat - y)/sde.s(t_hat)  # undo scaling and shifting
            score = model.score(x_tilde, y, sigma_hat, t_hat)
            d_hat = sde.probability_flow(x_hat, y, score, t_hat)
            x = x_hat + (t[i+1] - t_hat)*d_hat
            if i < self.num_steps - 1:
                x_tilde = (x - y)/sde.s(t[i+1])  # undo scaling and shifting
                score = model.score(x_tilde, y, sigma[i+1], t[i+1])
                d_next = sde.probability_flow(x, y, score, t[i+1])
                x = x_hat + 0.5*(t[i+1] - t_hat)*(d_hat + d_next)

        nfe = 2*self.num_steps
        return x, nfe


@SolverRegistry.register('pc')
class PCSolver:
    def __init__(self, num_steps, corrector_steps, corrector_snr, **kwargs):
        self.num_steps = num_steps
        self.corrector_steps = corrector_steps
        self.corrector_snr = corrector_snr

    def __call__(self, sde, y, model):
        dt = -1/self.num_steps
        t = torch.arange(1, 0, dt, device=y.device)
        sigma = sde.sigma(t)
        x = sde.prior(y)
        eps = 2*(self.corrector_snr*sde.s(t)*sigma)**2

        for i in range(self.num_steps):
            # corrector step
            for _ in range(self.corrector_steps):
                x_tilde = (x - y)/sde.s(t[i])  # undo scaling and shifting
                score = model.score(x_tilde, y, sigma[i], t[i])
                x += eps[i]*score + (2*eps[i])**0.5*torch.randn_like(x)

            # predictor step
            x_tilde = (x - y)/sde.s(t[i])  # undo scaling and shifting
            score = model.score(x_tilde, y, sigma[i], t[i])
            if i < self.num_steps - 1:
                x += sde.reverse_step(x, y, score, t[i], dt)
            else:  # don't add noise on the last step
                x += dt*sde.probability_flow(x, y, score, t[i])

        nfe = self.num_steps * (self.corrector_steps + 1)
        return x, nfe

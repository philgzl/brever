import torch
import torch.nn.functional as F

from ...modules import STFT
from ..base import BreverBaseModel, ModelRegistry
from .net import DiffusionUNet
from .preconditioning import Preconditioning
from .sdes import SDERegistry
from .solvers import SolverRegistry


@ModelRegistry.register('sgmsep')
class SGMSEp(BreverBaseModel):
    """Proposed in [1]_.

    References
    ----------
    .. [1] J. Richter, S. Welker, J.-M. Lemercier, B. Lay and T. Gerkmann,
           "Speech Enhancement and Dereverberation with Diffusion-Based
           Generative Models", in IEEE/ACM TASLP, 2023.
    """

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
        sde_bb_scaling: float = 0.1,
        sde_bb_k: float = 10.0,
        solver_name: str = 'pc',
        solver_num_steps: int = 16,
        solver_edm_schurn: float = float('inf'),
        solver_edm_smin: float = 0.0,
        solver_edm_smax: float = float('inf'),
        solver_edm_snoise: float = 1.0,
        solver_pc_corrector_steps: int = 1,
        solver_pc_corrector_snr: float = 0.5,
        net_base_channels: int = 128,
        net_channel_mult: list[int] = [1, 1, 2, 2, 2, 2, 2],
        net_num_blocks_per_res: int = 2,
        net_noise_channel_mult: int = 2,
        net_emb_channel_mult: int = 4,
        net_fir_kernel: list[int] = [1, 3, 3, 1],
        net_attn_resolutions: list[int] = [16],
        net_attn_bottleneck: bool = True,
        net_encoder_type: str = 'skip',
        net_decoder_type: str = 'skip',
        net_block_type: str = 'ncsn',
        net_skip_scale: float = 0.5 ** 0.5,
        net_dropout: float = 0.0,
        net_aux_out_channels: int = 4,
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
            scaling=sde_bb_scaling,
            k=sde_bb_k,
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
            num_blocks_per_res=net_num_blocks_per_res,
            noise_channel_mult=net_noise_channel_mult,
            emb_channel_mult=net_emb_channel_mult,
            fir_kernel=net_fir_kernel,
            attn_resolutions=net_attn_resolutions,
            attn_bottleneck=net_attn_bottleneck,
            encoder_type=net_encoder_type,
            decoder_type=net_decoder_type,
            block_type=net_block_type,
            skip_scale=net_skip_scale,
            dropout=net_dropout,
            aux_out_channels=net_aux_out_channels,
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

        self.optimizer = self.init_optimizer(optimizer, lr=learning_rate)

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

    def loss(self, batch, lengths, use_amp):
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


@ModelRegistry.register('sgmsepm')
class SGMSEpM(SGMSEp):
    """Proposed in [1]_.

    References
    ----------
    .. [1] J.-M. Lemercier, J. Richter, S. Welker and T. Gerkmann, "Analysing
           Diffusion-Based Generative Approaches Versus Discriminative
           Approaches for Speech Restoration", in Proc. IEEE ICASSP, 2023.
    """

    _is_submodel = True

    def __init__(
        self,
        net_channel_mult: list[int] = [1, 2, 2, 2],
        net_num_blocks_per_res: int = 1,
        net_attn_resolutions: list[int] = [],
        **kwargs,
    ):
        super().__init__(
            net_channel_mult=net_channel_mult,
            net_num_blocks_per_res=net_num_blocks_per_res,
            net_attn_resolutions=net_attn_resolutions,
            **kwargs,
        )


@ModelRegistry.register('sgmsepheun')
class sgmsepheun(SGMSEp):

    _is_submodel = True

    def __init__(
        self,
        sde_name: str = 'brever-oucosine',
        sde_stiffness: float = 0.0,
        solver_name: str = 'edm',
        preconditioning_cskip: str = 'edm',
        preconditioning_cout: str = 'edm',
        preconditioning_cin: str = 'edm',
        preconditioning_cnoise: str = 'edm',
        preconditioning_cshift: str = 'edm',
        preconditioning_weight: str = 'edm',
        **kwargs,
    ):
        super().__init__(
            sde_name=sde_name,
            solver_name=solver_name,
            preconditioning_cskip=preconditioning_cskip,
            preconditioning_cout=preconditioning_cout,
            preconditioning_cin=preconditioning_cin,
            preconditioning_cnoise=preconditioning_cnoise,
            preconditioning_cshift=preconditioning_cshift,
            preconditioning_weight=preconditioning_weight,
            **kwargs,
        )


@ModelRegistry.register('sgmsepmheun')
class sgmsepmheun(SGMSEpM):

    _is_submodel = True

    def __init__(
        self,
        sde_name: str = 'brever-oucosine',
        sde_stiffness: float = 0.0,
        solver_name: str = 'edm',
        preconditioning_cskip: str = 'edm',
        preconditioning_cout: str = 'edm',
        preconditioning_cin: str = 'edm',
        preconditioning_cnoise: str = 'edm',
        preconditioning_cshift: str = 'edm',
        preconditioning_weight: str = 'edm',
        **kwargs,
    ):
        super().__init__(
            sde_name=sde_name,
            solver_name=solver_name,
            preconditioning_cskip=preconditioning_cskip,
            preconditioning_cout=preconditioning_cout,
            preconditioning_cin=preconditioning_cin,
            preconditioning_cnoise=preconditioning_cnoise,
            preconditioning_cshift=preconditioning_cshift,
            preconditioning_weight=preconditioning_weight,
            **kwargs,
        )


@ModelRegistry.register('idmse')
class IDMSE(SGMSEp):

    _is_submodel = True

    def __init__(
        self,
        sde_name: str = 'brever-oucosine',
        sde_stiffness: float = 0.0,
        solver_name: str = 'edm',
        preconditioning_cskip: str = 'edm',
        preconditioning_cout: str = 'edm',
        preconditioning_cin: str = 'edm',
        preconditioning_cnoise: str = 'edm',
        preconditioning_cshift: str = 'edm',
        preconditioning_weight: str = 'edm',
        net_base_channels: int = 64,
        net_channel_mult: list[int] = [1, 2, 3, 4],
        net_num_blocks_per_res: int = 1,
        net_noise_channel_mult: int = 1,
        net_emb_channel_mult: int = 4,
        net_fir_kernel: list[int] = [1, 1],
        net_attn_resolutions: list[int] = [],
        net_encoder_type: str = 'standard',
        net_decoder_type: str = 'standard',
        net_block_type: str = 'adm',
        **kwargs,
    ):
        super().__init__(
            sde_name=sde_name,
            solver_name=solver_name,
            preconditioning_cskip=preconditioning_cskip,
            preconditioning_cout=preconditioning_cout,
            preconditioning_cin=preconditioning_cin,
            preconditioning_cnoise=preconditioning_cnoise,
            preconditioning_cshift=preconditioning_cshift,
            preconditioning_weight=preconditioning_weight,
            net_base_channels=net_base_channels,
            net_channel_mult=net_channel_mult,
            net_num_blocks_per_res=net_num_blocks_per_res,
            net_noise_channel_mult=net_noise_channel_mult,
            net_emb_channel_mult=net_emb_channel_mult,
            net_fir_kernel=net_fir_kernel,
            net_attn_resolutions=net_attn_resolutions,
            net_encoder_type=net_encoder_type,
            net_decoder_type=net_decoder_type,
            net_block_type=net_block_type,
            **kwargs,
        )

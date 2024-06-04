import inspect
from itertools import permutations

import torch

from .modules import STFT
from .registry import Registry

eps = torch.finfo(torch.float32).eps

CriterionRegistry = Registry('criterion')


def init_criterion(name, **kwargs):
    criterion = CriterionRegistry.get(name)
    if inspect.isclass(criterion):
        criterion = criterion(**kwargs)
    return criterion


@CriterionRegistry.register('sisnr')
def sisnr(x, y, lengths):
    """Scale-invariant signal-to-noise ratio (SI-SNR).

    With permutation invariant training (PIT).

    Parameters
    ----------
    x : torch.Tensor
        Estimated sources. Shape `(batch_size, sources, length)`.
    y : torch.Tensor
        True sources. Shape `(batch_size, sources, length)`.
    lengths : torch.Tensor
        Original length of unbatched sources. Shape `(batch_size,)`.

    Returns
    -------
    loss : torch.Tensor
        SI-SNR loss. Shape `(batch_size,)`.
    """
    assert x.shape == y.shape
    assert x.ndim == 3

    # apply mask a first time to get correct normalization statistics
    x, y = apply_mask(x, y, lengths)

    # normalize
    x = x - x.sum(2, keepdim=True)/lengths.view(-1, 1, 1)
    y = y - y.sum(2, keepdim=True)/lengths.view(-1, 1, 1)

    # apply mask a second time since trailing samples are now non-zero
    x, y = apply_mask(x, y, lengths)

    # calculate pair-wise snr
    s_hat = x.unsqueeze(1)  # (batch, 1, sources, length)
    s = y.unsqueeze(2)  # (batch, sources, 1, length)
    s_target = (s_hat * s).sum(3, keepdim=True) * s \
        / s.pow(2).sum(3, keepdim=True)  # (batch, sources, sources, length)
    e_noise = s_hat - s_target  # (batch, sources, sources, length)
    si_snr = s_target.pow(2).sum(3) / (e_noise.pow(2).sum(3) + eps)
    si_snr = 10*torch.log10(si_snr + eps)  # (batch, sources, sources)

    # permute
    S = x.shape[1]
    perms = x.new_tensor(list(permutations(range(S))), dtype=torch.long)
    index = perms.unsqueeze(2)
    one_hot = x.new_zeros((*perms.shape, S)).scatter_(2, index, 1)
    snr_set = torch.einsum('bij,pij->bp', [si_snr, one_hot])
    max_snr = snr_set.amax(1)
    max_snr /= S

    return - max_snr


@CriterionRegistry.register('snr')
def snr(x, y, lengths):
    """Signal-to-noise ratio (SNR).

    Without permutation invariant training (PIT), i.e. element-wise SNRs
    are calculated before averaging.

    Parameters
    ----------
    x : torch.Tensor
        Estimated sources. Shape `(batch_size, ..., length)`.
    y : torch.Tensor
        True sources. Shape `(batch_size, ..., length)`.
    lengths : torch.Tensor
        Original length of unbatched sources. Shape `(batch_size,)`.

    Returns
    -------
    loss : torch.Tensor
        SNR loss. Shape `(batch_size,)`.
    """
    assert x.shape == y.shape
    assert x.ndim >= 2
    x, y = apply_mask(x, y, lengths)
    snr = y.pow(2).sum(-1) / ((y - x).pow(2).sum(-1) + eps)
    snr = 10*torch.log10(snr + eps)
    return - snr.mean(tuple(range(1, x.ndim - 1)))


@CriterionRegistry.register('mse')
def mse(x, y, lengths, weight=None):
    """Mean squared error (MSE) between two tensors.

    Parameters
    ----------
    x : torch.Tensor
        Batched tensor. Arbitrary shape `(batch_size, ..., length)`.
    y : torch.Tensor
        Batched tensor. Same shape as `x`.
    lengths : torch.Tensor
        Original length of unbatched sources. Shape `(batch_size,)`.
    weight : torch.Tensor or None
        Weighting. Shape `(batch_size,)`. Default is `None`, i.e. no
        weighting is applied.

    Returns
    -------
    loss : torch.Tensor
        MSE loss. Shape `(batch_size,)`.
    """
    assert x.shape == y.shape
    assert x.ndim >= 2
    x, y = apply_mask(x, y, lengths)
    loss = (x - y).abs().pow(2).sum(-1)
    loss /= lengths.view(-1, *[1]*(x.ndim - 2))
    if weight is not None:
        loss *= weight.view(-1, *[1]*(x.ndim - 2))
    return loss.mean(tuple(range(1, x.ndim - 1)))


@CriterionRegistry.register('multiresyu')
class MultiResYuLoss:
    """Multi-resolution STFT magnitude + L1 time-domain loss.

    Proposed in [1]_.

    Parameters
    ----------
    frame_lengths : list of int
        Frame lengths for STFTs. Default is `[512]`.
    hop_lengths : list of int or None
        Hop lengths for STFTs. If `None`, defaults to half the frame lengths.
    time_domain_weight : float
        Weight for the time-domain loss. Default is `0.5`.
    spectral_weight : float
        Weight for the spectral loss. Default is `0.5`.
    scale_invariant : bool
        Whether to make the loss scale-invariant. Default is `False`.

    References
    ----------
    .. [1] Y.-J. Lu, S. Cornell, X. Chang, W. Zhang, C. Li, Z. Ni, Z.-Q. Wang
           and S. Watanabe, "Towards Low-Distortion Multi-Channel Speech
           Enhancement: The ESPNet-SE Submission to the L3DAS22 Challenge", in
           Proc. IEEE ICASSP, 2022.
    """

    def __init__(
        self,
        frame_lengths=[512],
        hop_lengths=None,
        time_domain_weight=0.5,
        spectral_weight=0.5,
        scale_invariant=False,
    ):
        if hop_lengths is None:
            hop_lengths = [x // 2 for x in frame_lengths]

        self.stfts = [
            STFT(
                frame_length=frame_length,
                hop_length=hop_length,
                window=None,
                normalized=False,
            )
            for frame_length, hop_length in zip(frame_lengths, hop_lengths)
        ]

        self.time_domain_weight = time_domain_weight
        self.spectral_weight = spectral_weight
        self.scale_invariant = scale_invariant

    def __call__(self, x, y, lengths):
        """Calculate loss.

        Parameters
        ----------
        x : torch.Tensor
            Estimated sources. Shape `(batch_size, ..., length)`.
        y : torch.Tensor
            True sources. Shape `(batch_size, ..., length)`.
        lengths : torch.Tensor
            Original length of unbatched sources. Shape `(batch_size,)`.

        Returns
        -------
        loss : torch.Tensor
            Loss. Shape `(batch_size,)`.
        """
        assert x.shape == y.shape

        x, y = apply_mask(x, y, lengths)

        if self.scale_invariant:
            scaling_factor = (x * y).sum(-1, keepdim=True) / \
                (x.pow(2).sum(-1, keepdim=True) + eps)
        else:
            scaling_factor = 1

        time_domain_loss = (scaling_factor * x - y).abs().sum(-1)

        output = self.time_domain_weight * time_domain_loss

        for stft in self.stfts:
            y_mag = stft(y).abs()
            x_mag = stft(scaling_factor * x).abs()
            spectral_loss = (x_mag - y_mag).abs().sum((-2, -1))

            output += self.spectral_weight * spectral_loss / len(self.stfts)

        output /= lengths.view(-1, *[1]*(x.ndim - 2))
        return output.mean(tuple(range(1, x.ndim - 1)))


def apply_mask(x, y, lengths):
    assert len(lengths) == x.shape[0]
    mask = torch.zeros(x.shape, device=x.device)
    for i, length in enumerate(lengths):
        mask[i, ..., :length] = 1
    return x*mask, y*mask

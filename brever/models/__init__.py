import torch

from .base import ModelRegistry  # noqa: F401
from .convtasnet import ConvTasNet  # noqa: F401
from .dccrn import DCCRN  # noqa: F401
from .ffnn import FFNN  # noqa: F401
from .manner import MANNER  # noqa: F401
from .sgmse import SGMSE  # noqa: F401


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def set_all_weights(model, val):
    for p in model.parameters():
        p.fill_(val)

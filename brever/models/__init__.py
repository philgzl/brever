import torch

from .base import ModelRegistry
from .convtasnet import ConvTasNet
from .dccrn import DCCRN
from .ffnn import FFNN
from .manner import MANNER
from .metricganokd import MetricGANOKD, MetricGANp
from .sgmse import IDMSE, SGMSEp, SGMSEpM
from .tfgridnet import TFGridNet

__all__ = [
    'ModelRegistry',
    'ConvTasNet',
    'DCCRN',
    'FFNN',
    'MANNER',
    'MetricGANOKD',
    'MetricGANp',
    'IDMSE',
    'SGMSEp',
    'SGMSEpM',
    'TFGridNet',
]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def set_all_weights(model, val=1e-3, buffers=False):
    for p in model.parameters():
        p.fill_(val)
    if buffers:
        for b in model.buffers():
            b.fill_(val)

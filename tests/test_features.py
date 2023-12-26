import pytest
import torch

from brever.modules import FeatureExtractor, MelFilterbank

FEATURES = [
    'ild',
    'ipd',
    'ic',
    'fbe',
    'logfbe',
    'cubicfbe',
    'pdf',
    'logpdf',
    'cubicpdf',
    'mfcc',
    'cubicmfcc',
    'pdfcc',
]


@pytest.mark.parametrize(
    'features',
    [
        *[[f] for f in FEATURES],  # individually
        FEATURES,  # all together
    ]
)
def test_feature_extractor(features):
    x = torch.randn(2, 257, 30, dtype=torch.complex64)
    mel_fb = MelFilterbank()
    extractor = FeatureExtractor(features=features, mel_fb=mel_fb)
    extractor(x)

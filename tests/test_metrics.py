import numpy as np
import pytest
import torch
import torch.nn.functional as F

from brever.metrics import MetricRegistry

BATCH_SIZE = 2
MIN_LENGTH = 16000
MAX_LENGTH = 48000


@pytest.mark.parametrize(
    'metric', MetricRegistry.keys()
)
def test_metric(metric):
    # seed
    torch.manual_seed(42)

    # randomize lengths
    lengths = torch.randint(MIN_LENGTH, MAX_LENGTH, (BATCH_SIZE,))

    # create targets with length lengths
    targets = [torch.randn(length) for length in lengths]

    # pad targets and make batch
    batched_targets = torch.stack([
        F.pad(t, (0, MAX_LENGTH - t.shape[-1])) for t in targets
    ])

    # create noisy inputs
    batched_inputs = batched_targets + 0.5*torch.randn(*batched_targets.shape)
    inputs = [x[..., :length] for x, length in zip(batched_inputs, lengths)]

    # init criterion
    metric = MetricRegistry.get(metric)

    # 2 ways of calculating: either batch processing...
    try:
        batched_metrics = metric(
            batched_inputs, batched_targets, lengths=lengths, batched=True
        )
    except TypeError:
        batched_metrics = metric(
            batched_inputs, batched_targets, lengths=lengths,
        )
    if isinstance(batched_metrics, np.ndarray):
        batched_metrics = torch.from_numpy(batched_metrics).float()

    # ...or one-by-one
    metrics = torch.tensor([metric(x, y) for x, y in zip(inputs, targets)])

    # both should give the same result
    assert torch.allclose(batched_metrics, metrics)

import pytest
import torch
import torch.nn.functional as F

from brever.criterion import CriterionRegistry

BATCH_SIZE = 8
SOURCES = 4
MIN_LENGTH = 100
MAX_LENGTH = 200


@pytest.mark.parametrize(
    'criterion', CriterionRegistry.keys()
)
def test_criterion(criterion):
    # seed
    torch.manual_seed(0)

    # randomize lengths
    lengths = torch.randint(MIN_LENGTH, MAX_LENGTH, (BATCH_SIZE,))

    # create inputs with length lengths
    inputs = [torch.randn(SOURCES, length) for length in lengths]

    # pad inputs and make batch
    batched_inputs = torch.stack([
        F.pad(x, (0, MAX_LENGTH - x.shape[-1])) for x in inputs
    ])

    # mimic neural net processing
    dummy_net_func = lambda x: x + torch.randn(*x.shape)  # noqa: E731
    batched_outputs = dummy_net_func(batched_inputs)
    outputs = [x[..., :length] for x, length in zip(batched_outputs, lengths)]

    # create targets
    targets = [torch.randn(SOURCES, length) for length in lengths]

    # pad targets and make batch
    batched_targets = torch.stack([
        F.pad(x, (0, MAX_LENGTH - x.shape[-1])) for x in targets
    ])

    # init criterion
    criterion = CriterionRegistry.get(criterion)

    # 2 ways of calculating: either batch processing...
    batched_losses = criterion(batched_outputs, batched_targets, lengths)

    # ...or one-by-one
    losses = torch.tensor([
        criterion(x.unsqueeze(0), y.unsqueeze(0), torch.tensor([length]))
        for x, y, length in zip(outputs, targets, lengths)
    ])

    # both should give the same result
    assert torch.allclose(batched_losses, losses)

import itertools

import numpy as np
import pytest
import torch
from utils import DummyDataset, DummyModel

from brever.batching import BatchSamplerRegistry, BreverBatchSampler
from brever.data import BreverDataLoader

FS = 16000
CHANNELS = 2
SOURCES = 3
BATCH_SIZE = 4
N_EXAMPLES = 100
MIN_LENGTH = int(FS*0.1)
MAX_LENGTH = FS*10


class BatchTester:
    def __init__(self, batch_sampler_name, batch_sampler, dynamic,
                 transform_length):
        self.batch_sampler_name = batch_sampler_name
        self.batch_sampler = batch_sampler
        self.dynamic = dynamic
        self.batch_sizes, self.pad_amounts = batch_sampler.calc_batch_stats(
            transform_length=transform_length
        )
        self.batch_size_error_count = 0
        if batch_sampler_name == 'bucket':
            self.max_batch_size_error = batch_sampler.num_buckets
        else:
            self.max_batch_size_error = 1

    def reset_error_count(self):
        # need to reset error count between model inputs
        self.batch_size_error_count = 0

    def test(self, inputs, all_lengths):
        # convert to single item list if inputs is a tensor
        assert isinstance(all_lengths, torch.Tensor)
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
            assert all_lengths.ndim == 1
            all_lengths = all_lengths.reshape(-1, 1)
        else:
            assert isinstance(inputs, list)
            assert all_lengths.ndim == 2
        for x, lengths in zip(inputs, all_lengths.T):
            self.reset_error_count()
            if self.dynamic:
                assert x.shape[0]*x.shape[-1] <= self.batch_sampler.batch_size
            else:
                # batch size can be different only when it's the last one
                # except for bucketing where it can happen for each bucket
                try:
                    assert x.shape[0] == BATCH_SIZE
                except AssertionError:
                    if self.batch_size_error_count < self.max_batch_size_error:
                        self.batch_size_error_count += 1
                    else:
                        raise
            assert x.shape[-1] == max(lengths)
            assert all((y[..., k:] == 0).all() for y, k in zip(x, lengths))
            assert all((y[..., k-1] != 0).all() for y, k in zip(x, lengths))
            if self.batch_sampler_name == 'sorted':
                assert _is_sorted(lengths)
            elif self.batch_sampler_name == 'bucket':
                i = np.searchsorted(
                    self.batch_sampler.right_bucket_limits, x.shape[-1]
                )
                left = 0 if i == 0 \
                    else self.batch_sampler.right_bucket_limits[i-1]
                right = self.batch_sampler.right_bucket_limits[i]
                assert all(left <= k <= right for k in lengths)
            elif self.batch_sampler_name != 'random':
                raise ValueError(
                    f'unexpected batch_sampler_name: {self.batch_sampler_name}'
                )
        x = inputs[0]
        batch_size = x.shape[0]*x.shape[-1]
        pad_amount = sum(x.shape[-1] - k for k in all_lengths[:, 0]).item()
        self.batch_sizes.remove(batch_size)
        self.pad_amounts.remove(pad_amount)


def _get_dataloader(dataset, batch_sampler_name, dynamic, shuffle):
    batch_sampler_cls = BatchSamplerRegistry.get(batch_sampler_name)
    batch_sampler = batch_sampler_cls(
        dataset=dataset,
        batch_size=(BATCH_SIZE*MAX_LENGTH)/FS if dynamic else BATCH_SIZE,
        dynamic=dynamic,
        shuffle=shuffle,
    )
    dataloader = BreverDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
    )
    return batch_sampler, dataloader


def _is_sorted(list_):
    return all(list_[i] <= list_[i+1] for i in range(len(list_) - 1))


def _test_batch_sampler_on_dset(batch_sampler_name, dataset, dynamic,
                                transform_length):
    batch_sampler, dataloader = _get_dataloader(
        dataset, batch_sampler_name, dynamic, shuffle=True
    )
    n = len(batch_sampler)
    # test every batch
    batch_tester = BatchTester(
        batch_sampler_name, batch_sampler, dynamic, transform_length,
    )
    i = 0
    for inputs, lengths in dataloader:
        batch_tester.test(inputs, lengths)
        i += 1
    assert n == i
    # test error when attempting to iterate again
    with pytest.raises(ValueError):
        for inputs, lengths in dataloader:
            break
    # test error fix by setting epoch
    dataloader.set_epoch(1)
    for inputs, lengths in dataloader:
        break
    # check shuffle
    dataloader.set_epoch(2)
    for inputs, lengths in dataloader:
        break
    dataloader.set_epoch(3)
    for (inputs_), lengths_ in dataloader:
        for x, x_ in zip(inputs, inputs_):
            if x.shape == x_.shape:
                assert (x != x_).any()
        if lengths.shape == lengths_.shape:
            assert (lengths != lengths_).any()
        break
    # check no shuffle
    batch_sampler, dataloader = _get_dataloader(
        dataset, batch_sampler_name, dynamic, shuffle=False
    )
    for inputs, lengths in dataloader:
        break
    dataloader.set_epoch(1)
    for (inputs_), lengths_ in dataloader:
        for x, x_ in zip(inputs, inputs_):
            assert (x == x_).all()
        assert (lengths == lengths_).all()
        break


@pytest.mark.parametrize(
    'batch_sampler_name, dynamic',
    itertools.product(BatchSamplerRegistry.keys(), [False, True])
)
def test_batch_sampler(batch_sampler_name, dynamic):
    for use_model in [False, True]:
        for use_transform in [False, True]:
            if not use_model and use_transform:
                continue
            dataset, transform_length = _get_model_and_dataset(
                use_model, use_transform
            )
            _test_batch_sampler_on_dset(
                batch_sampler_name, dataset, dynamic, transform_length,
            )
            # test on a torch.utils.data.Subset
            train_split, _ = torch.utils.data.random_split(
                dataset, [N_EXAMPLES//2, N_EXAMPLES//2],
                torch.Generator().manual_seed(0),
            )
            _test_batch_sampler_on_dset(
                batch_sampler_name, train_split, dynamic, transform_length,
            )


def _get_model_and_dataset(use_model, use_transform):
    if use_model:
        model = DummyModel(use_transform=use_transform)
        transform = model.transform
        transform_length = model.transform_length
    else:
        transform = None
        transform_length = None
    dataset = DummyDataset(
        n_examples=N_EXAMPLES,
        n_sources=SOURCES,
        n_channels=CHANNELS,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
        transform=transform,
    )
    return dataset, transform_length


def test_errors():
    dataset = DummyDataset(
        n_examples=N_EXAMPLES,
        n_sources=SOURCES,
        n_channels=CHANNELS,
        min_length=MIN_LENGTH,
        max_length=MAX_LENGTH,
    )
    batch_sampler = BreverBatchSampler(dataset, BATCH_SIZE)
    with pytest.raises(NotImplementedError):
        next(iter(batch_sampler))
    with pytest.raises(NotImplementedError):
        len(batch_sampler)

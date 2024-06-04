import logging
import random

import numpy as np
import torch

from .data import BreverDataset
from .registry import Registry

BatchSamplerRegistry = Registry('batch_sampler')


class BreverBatchSampler(torch.utils.data.Sampler):
    """Base class for all samplers.

    Integrates with BreverDataset to make batches of segments.

    Subclasses should implement the _generate_batches method which takes as
    input the dataset indices and returns a list of batches where each batch
    is a list of `(segment_idx, segment_length)` tuples. The `__init__` method
    should also be overwritten in case the sampler requires extra arguments.

    Also implements a `set_epoch` method to shuffle batches with the correct
    seed when resuming training. The `set_epoch` method must be called before
    iterating over the batch sampler, unless `shuffle == False`.

    Parameters
    ----------
    dataset : BreverDataset
        BreverDataset instance.
    batch_size : int or float
        Batch size. If `dynamic == False`, it is defined as a number of
        segments in each batch (fixed batch size). If `dynamic == True`, it is
        a total length of segments in seconds (dynamic batch size).
    drop_last : bool, optional
        Whether to drop the last segments in the dataset if they don't form
        a full batch. Default is `False`.
    shuffle : bool, optional
        Whether to shuffle the batches before each epoch. Default is `True`.
    seed : int, optional
        Random seed for shuffling. Default is `0`.
    dynamic : bool, optional
        Whether `batch_size` is defined as a number of segments in each batch
        (fixed batch size) or as a total length of segments (dynamic batch
        size) in seconds. Default is `False`, i.e. a fixed batch size is used.
    sort : bool, optional
        Whether to sort the segments by length before generating the batches.
        If `shuffle == True`, segments are sorted but segments of equal length
        are shuffled. Default is `False`.
    fs : int, optional
        Sampling rate. Used when `dynamic == True` to convert the batch size
        in seconds to a number of samples. Default is `16000`. Ignored if
        `dynamic == False`.
    reverse : bool, optional
        Whether to reverse the order of the batches. Default is `False`.
        Ignored if `sort == False`.
    """

    def __init__(
        self,
        dataset: BreverDataset,
        batch_size: int | float,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        dynamic: bool = False,
        sort: bool = False,
        fs: int = 16000,
        reverse: bool = False,
    ):
        self.dataset = dataset
        if dynamic:
            self.batch_size = round(fs*batch_size)
        else:
            if isinstance(batch_size, float):
                logging.warning('Got float batch_size even though dynamic is '
                                'False. Casting batch_size to int.')
            self.batch_size = int(batch_size)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dynamic = dynamic
        self.sort = sort
        self.reverse = reverse
        self._seed = random.Random(seed).randrange(2**32)
        self._epoch = 0
        self._previous_epoch = -1
        self._segment_lengths = None
        self._batches = None

    def __iter__(self):
        if self.shuffle:
            if self._epoch == self._previous_epoch:
                raise ValueError(
                    'the set_epoch method must be called before iterating '
                    'over the dataloader in order to regenerate the batches '
                    'with the correct seed'
                )
            self.generate_batches()
            self.shuffle_batches()
            self._previous_epoch = self._epoch
        elif self._batches is None:
            self.generate_batches()
        for batch in self._batches:
            yield [idx for idx, length in batch]

    def generate_batches(self):
        indices = self._generate_indices()
        self._batches = self._generate_batches(indices)

    def _generate_indices(self):
        self.get_segment_lengths()
        if self.sort:
            if self.shuffle:
                # sort by length but randomize segments of same length
                randomizer = random.Random(self._seed + self._epoch)
                lengths = sorted(self._segment_lengths,
                                 key=lambda x: (x[1], randomizer.random()),
                                 reverse=self.reverse)
            else:
                lengths = sorted(self._segment_lengths, key=lambda x: x[1],
                                 reverse=self.reverse)
            indices = [idx for idx, length in lengths]
        else:
            indices = list(range(len(self._segment_lengths)))
            if self.shuffle:
                randomizer = random.Random(self._seed + self._epoch)
                randomizer.shuffle(indices)
        return indices

    def get_segment_lengths(self):
        if isinstance(self.dataset, torch.utils.data.Subset):
            dataset = self.dataset.dataset
            indices = self.dataset.indices
        else:
            dataset = self.dataset
            indices = range(len(dataset))
        if self._segment_lengths is None or dataset.rmm_dset is not None:
            self._segment_lengths = [
                (i, dataset.get_segment_length(j))
                for i, j in enumerate(indices)
            ]

    def _generate_batches(self, indices):
        raise NotImplementedError

    def set_epoch(self, epoch):
        self._epoch = epoch

    def shuffle_batches(self):
        randomizer = random.Random(self._seed + self._epoch)
        randomizer.shuffle(self._batches)

    def __len__(self):
        if self._batches is None:
            self.generate_batches()
        return len(self._batches)

    def calc_batch_stats(self, transform_length=None):
        if transform_length is None:
            transform_length = lambda x: x  # noqa: E731
        batch_sizes = []
        pad_amounts = []
        for batch in self._batches:
            batch_lengths = [transform_length(length) for idx, length in batch]
            max_length = max(batch_lengths)
            batch_sizes.append(len(batch)*max_length)
            pad_amounts.append(
                sum(max_length - length for length in batch_lengths)
            )
        return batch_sizes, pad_amounts


class _BaseRandSortBatchSampler(BreverBatchSampler):
    """Base class for the random and sorted batch samplers."""

    def _generate_batches(self, indices):
        batches = []
        batch = []
        for i in indices:
            segment_idx, segment_length = self._segment_lengths[i]
            if self._new_batch(batch, segment_length):
                batches.append(batch)
                batch = []
                batch.append((segment_idx, segment_length))
            else:
                batch.append((segment_idx, segment_length))
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        return batches

    def _new_batch(self, batch, segment_length):
        output = False
        if self.dynamic:
            if segment_length > self.batch_size:
                raise ValueError(
                    'got a segment that is longer than the dynamic batch size'
                )
            batch_length = max(x[1] for x in batch) if batch else 0
            if (len(batch) + 1)*max(segment_length, batch_length) \
                    > self.batch_size:
                output = True
        elif len(batch) + 1 > self.batch_size:
            output = True
        return output


@BatchSamplerRegistry.register('random')
class RandomBatchSampler(_BaseRandSortBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sort=False, **kwargs)


@BatchSamplerRegistry.register('sorted')
class SortedBatchSampler(_BaseRandSortBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, sort=True, **kwargs)


@BatchSamplerRegistry.register('bucket')
class BucketBatchSampler(BreverBatchSampler):
    """Bucket batching.

    Segments are grouped into different buckets according to their length.
    Batches are formed with segments from the same bucket. This reduces the
    amount of zero-padding while keeping some randomness.

    Parameters
    ----------
    num_buckets : int, optional
        The number of buckets. This defines a compromise between padding and
        randomization; the more buckets, the less the padding, but also the
        less randomization. Bucket limits are uniformly spaced between the
        minimum and maximum segment length in the dataset. Default is `10`.
    """

    def __init__(self, *args, num_buckets=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_buckets = num_buckets

    def _generate_batches(self, indices):
        max_length = max(x[1] for x in self._segment_lengths)

        right_bucket_limits = np.linspace(
            max_length/self.num_buckets, max_length, self.num_buckets,
        )
        self.right_bucket_limits = right_bucket_limits  # for unit testing

        if self.dynamic:
            bucket_sizes = self.batch_size//right_bucket_limits
        else:
            bucket_sizes = [self.batch_size]*self.num_buckets

        batches = []
        buckets = [[] for _ in range(self.num_buckets)]
        for i in indices:
            segment_idx, segment_length = self._segment_lengths[i]
            bucket_idx = np.searchsorted(right_bucket_limits, segment_length)
            if not 0 <= bucket_idx < self.num_buckets:
                raise ValueError(
                    'attempted to assign a segment to a non-existent bucket'
                )
            buckets[bucket_idx].append((segment_idx, segment_length))
            if len(buckets[bucket_idx]) == bucket_sizes[bucket_idx]:
                batches.append(buckets[bucket_idx])
                buckets[bucket_idx] = []
            elif len(buckets[bucket_idx]) > bucket_sizes[bucket_idx]:
                raise ValueError(
                    'maximum number of segments allowed in bucket exceeded'
                )

        if not self.drop_last:
            for bucket_idx, batch in enumerate(buckets):
                if len(batch) > 0:
                    batches.append(batch)

        return batches


class DistributedBatchSamplerWrapper(torch.utils.data.DistributedSampler):
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(dataset=sampler, *args, **kwargs)
        self.sampler = sampler

    def __iter__(self):
        for dist_index in super().__iter__():
            yield [i for i, length in self.sampler._batches[dist_index]]

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.sampler.set_epoch(epoch)

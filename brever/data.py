import logging
import os
import random
import re
import sys
import tarfile
from typing import Callable

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from .inspect import NoParse, Path

Transform = Callable[[torch.Tensor], torch.Tensor | tuple[torch.Tensor, ...]]


class BreverDataset(torch.utils.data.Dataset):
    """Main dataset class.

    Reads audio files from a dataset created using `scripts/create_dataset.py`.

    Parameters
    ----------
    path : str
        Path to the dataset directory.
    segment_length : float, optional
        If different from `0.0`, audio files are split into segments of fixed
        length equal to `segment_length` in seconds. Default is `0.0`.
    overlap_length : float, optional
        Amount of overlap between segments in seconds. Default is `0.0`.
        Ignored if `segment_length == 0.0`.
    fs : int, optional
        Sampling rate. Default is `16000`.
    sources : list of str, optional
        Sources to read from the dataset directory. Default is `['mixture',
        'foreground']`.
    segment_strategy : {'drop', 'pass', 'pad', 'overlap', 'random'}, optional
        How to deal with trailing segments:
        * `'drop'`: Trailing segments are discarded.
        * `'pass'`: Trailing segments are included as is.
        * `'pad'`: Trailing segments are included after zero-padding to match
        `segment_length`.
        * `'overlap'`: Trailing segments overlap with the second-to-last
        segment.
        * `'random'`: Segmentation is skipped altogether. A random segment with
        length `segment_length` is sampled on-the-fly from each audio file for
        each epoch. If the file is shorter than `segment_length` then the file
        is padded. Incompatible with preloading.
        Default is `'pass'`. Ignored if `segment_length == 0.0`.
    max_segment_length : float, optional
        Maximum segment length in seconds. When specified, audio files longer
        than `max_segment_length` are segmented into segments with length equal
        to `max_segment_length`. This is done by enforcing `segment_length =
        max_segment_length`. Trailing segments are dealt with according to
        `segment_strategy`. Default is `0.0`, i.e. segments can have any
        length. Ignored if `segment_length != 0.0`.
    tar : bool, optional
        Whether the audio files are stored in a tar archive. Default is `True`.
    transform : callable or None, optional
        Function used to pre-process the audio segments before making
        mini-batches. Takes as input a tensor with shape `(n_sources,
        n_channels, n_samples)` where `n_sources = len(sources)`, `n_channels
        = 2` (left and right) and `n_samples` is the segment length. Returns a
        new tensor or a tuple of new tensors with arbitrary shapes. The last
        dimension of each output tensor should be homogeneous to time (e.g.
        STFT frames), such that it can be padded to form mini-batches if
        needed. Usually defined as a method of the model. Default is `None`,
        i.e. no pre-processing.
    """

    def __init__(
        self,
        path: NoParse[Path],
        segment_length: float = 0.0,
        overlap_length: float = 0.0,
        fs: int = 16000,
        sources: list[str] = ['mixture', 'foreground'],
        segment_strategy: str = 'pass',
        max_segment_length: float = 0.0,
        tar: bool = True,
        transform: NoParse[Transform | None] = None,
    ):
        self.path = path
        self.segment_length = round(segment_length*fs)
        self.overlap_length = round(overlap_length*fs)
        self.fs = fs
        self.sources = sources
        self.segment_strategy = segment_strategy
        self.max_segment_length = round(max_segment_length*fs)
        if tar:
            self.archive = TarArchive(os.path.join(path, 'audio.tar'))
        else:
            self.archive = None
        self.transform = transform
        self.preloaded_data = None
        self.get_segment_info()

    def get_segment_info(self):
        file_lengths = self.get_file_lengths()
        if self.segment_length == 0.0 and self.max_segment_length != 0.0:
            max_file_length = max(file_lengths)
            if max_file_length > self.max_segment_length:
                logging.warning('Found a file longer than max_segment_length. '
                                'Setting segment_length to max_segment_length '
                                f'({self.max_segment_length}).')
                self.segment_length = self.max_segment_length
        self._segment_info = []
        if self.segment_length == 0.0:
            for file_idx, file_length in enumerate(file_lengths):
                self._segment_info.append((file_idx, (0, file_length)))
        else:
            for file_idx, file_length in enumerate(file_lengths):
                self._add_segment_info(file_idx, file_length)
        self._effective_duration = sum(
            end - start for _, (start, end) in self._segment_info
        )/self.fs

    def get_file_lengths(self):
        n_files = self.count_files()
        file_lengths = []
        logging.info('Reading file lengths...')
        for file_idx in tqdm(range(n_files)):
            source_paths = self.build_paths(file_idx)
            first_file = self.get_file(source_paths[0])
            first_metadata = torchaudio.info(first_file)
            first_length = first_metadata.num_frames
            # check all sources have the same length
            for source_path in source_paths[1:]:
                source_file = self.get_file(source_path)
                source_metadata = torchaudio.info(source_file)
                source_length = source_metadata.num_frames
                if source_length != first_length:
                    raise ValueError(
                        f'sources {file_idx} do not all have the same length'
                    )
            file_lengths.append(first_length)
        self._duration = sum(file_lengths)/self.fs
        return file_lengths

    def count_files(self):
        if self.archive is None:
            files = [
                f'audio/{file}'
                for file in os.listdir(os.path.join(self.path, 'audio'))
            ]
        else:
            files = self.archive.members
        return max(
            int(re.match(r'audio/(\d+)_.+\.flac', file).group(1))
            for file in files
        ) + 1

    def _add_segment_info(self, file_idx, file_length):
        if self.segment_strategy == 'random':
            start, end = 0, max(file_length, self.segment_length)
            self._segment_info.append((file_idx, (start, end)))
            return
        hop_length = self.segment_length - self.overlap_length
        n_segments = (file_length - self.segment_length)//hop_length + 1
        for segment_idx in range(n_segments):
            start = segment_idx*hop_length
            end = start + self.segment_length
            self._segment_info.append((file_idx, (start, end)))
        # if segment_length > file_length then we never entered the loop, so we
        # need to assign the last index of the last segment to 0, such that
        # handling of the remaining samples does not fail
        if n_segments <= 0:
            end = 0
        if self.segment_strategy == 'drop':
            pass
        elif self.segment_strategy == 'pass':
            if end != file_length:
                segment_idx = n_segments
                start = segment_idx*hop_length
                end = file_length
                self._segment_info.append((file_idx, (start, end)))
        elif self.segment_strategy == 'pad':
            if end != file_length:
                segment_idx = n_segments
                start = segment_idx*hop_length
                end = start + self.segment_length
                self._segment_info.append((file_idx, (start, end)))
        elif self.segment_strategy == 'overlap':
            if end != file_length:
                start = file_length - self.segment_length
                end = file_length
                self._segment_info.append((file_idx, (start, end)))
        else:
            raise ValueError('unrecognized segment strategy, got '
                             f'{self.segment_strategy}')

    def build_paths(self, file_idx):
        return [
            os.path.join('audio', f'{file_idx:05d}_{source}.flac')
            for source in self.sources
        ]

    def get_file(self, name):
        if self.archive is None:
            file = open(os.path.join(self.path, name), 'rb')
        else:
            file = self.archive.get_file(name.replace('\\', '/'))
        return file

    def __getitem__(self, index):
        if self.preloaded_data is not None:
            sources = self.preloaded_data[index]
        else:
            sources = self.load_segment(index)
            if self.transform is not None:
                sources = self.transform(sources)
        return sources

    def load_segment(self, index):
        file_idx, (start, end) = self._segment_info[index]
        if self.segment_strategy == 'random' and self.segment_length != 0.0:
            start = random.randint(start, end - self.segment_length)
            end = start + self.segment_length
        source_paths = self.build_paths(file_idx)
        sources = []
        for source_path in source_paths:
            source = self.load_file(source_path)
            if end > source.shape[-1]:
                if self.segment_strategy not in ['pad', 'random']:
                    raise ValueError(
                        "attempting to load a segment outside of file range "
                        "but segment strategy is not in ['pad', 'random'], "
                        f"got {self.segment_strategy}"
                    )
                source = F.pad(source, (0, end - source.shape[-1]))
            source = source[:, start:end]
            sources.append(source)
        return torch.stack(sources)

    def load_file(self, path):
        file = self.get_file(path)
        # x, _ = torchaudio.load(file)
        # torchaudio.load is BROKEN for file-like objects from FLAC files!
        # https://github.com/pytorch/audio/issues/2356
        # use soundfile instead
        x, fs = sf.read(file, dtype='float32')
        if fs != self.fs:
            raise ValueError(
                'file sampling rate does not match dataset fs attribute, got '
                f'{fs} and {self.fs}'
            )
        x = x.reshape(1, -1) if x.ndim == 1 else x.T
        return torch.from_numpy(x)

    def __len__(self):
        return len(self._segment_info)

    def get_segment_length(self, i):
        """Length of the i-th item in the dataset.

        `self._segment_into` contains tuples of the form `(file_idx, (start,
        end))` where `start` and `end` are the indices of the segment in the
        file with index `file_idx`. For the `'pad'` strategy, the `end` index
        can exceed the length of the file, but even then the segment length is
        `end - start` after padding.

        However, for the `'random'` segment strategy, `start` and `end`
        correspond to the whole file, and the true start and end indices are
        randomized on-the-fly. In that case, the segment length is always
        `self.segment_length`.
        """
        if self.segment_strategy == 'random':
            return self.segment_length
        file_idx, (start, end) = self._segment_info[i]
        return end - start

    def get_max_segment_length(self):
        """Length of the longest segment in the dataset.

        `self.max_segment_length` corresponds to the optional length limit that
        segments should not exceed. This method on the other hand returns the
        length of the longest segment in the dataset.
        """
        if self.segment_strategy == 'random':
            return self.segment_length
        return max(end - start for _, (start, end) in self._segment_info)

    def preload(self, device, tqdm_desc=None):
        if self.segment_strategy == 'random':
            raise ValueError("can't preload when segment_strategy is 'random'")
        preloaded_data = []
        for i in tqdm(range(len(self)), file=sys.stdout, desc=tqdm_desc):
            sources = self.__getitem__(i)
            if isinstance(sources, torch.Tensor):
                sources = sources.to(device)
            else:
                sources = [source.to(device) for source in sources]
            preloaded_data.append(sources)
        # set the attribute only at the end, otherwise __getitem__ will attempt
        # to access it inside the loop, causing infinite recursion
        self.preloaded_data = preloaded_data


class TarArchive:
    """Tar archive interface.

    The difficulty comes from the fact that tarfile is not thread safe, so
    when using multiple workers, each worker should have its own file handle.

    This class and only this class is inspired from code available here:

        https://github.com/jotaf98/simple-tar-dataset

    which is licensed under the following BSD 3-Clause License:

    Copyright (c) 2021 Joao F. Henriques

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

    The lines of code in this file outside of this class are not affected by
    this license.
    """

    def __init__(self, archive):
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        self.tar_obj = {worker: tarfile.open(archive)}
        self.archive = archive
        self.members = {m.name: m for m in self.tar_obj[worker].getmembers()}

    def get_file(self, name):
        # ensure a unique file handle per worker
        worker = torch.utils.data.get_worker_info()
        worker = worker.id if worker else None
        if worker not in self.tar_obj:
            self.tar_obj[worker] = tarfile.open(self.archive)
        return self.tar_obj[worker].extractfile(self.members[name])


class BreverDataLoader(torch.utils.data.DataLoader):
    """Main dataloader class.

    Implements the collating function to form batches of variable size inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn

    def set_epoch(self, epoch):
        self.batch_sampler.set_epoch(epoch)

    @staticmethod
    def _collate_fn(unbatched):
        """Collates variable size inputs.

        Variable size inputs are zero-padded to match the length of the longest
        example in the mini-batch along the last dimension. Supports an
        arbitrary number of model inputs.

        Parameters
        ----------
        unbatched : list of torch.Tensor or list of tuple of torch.Tensor
            Unbatched model inputs. The length of the list is the batch size,
            while the list elements are either tensors or tuples of tensors
            depending on the number of outputs from the dataset's
            `__getitem__`. Tensors can have a variable size along the last
            dimension, in which case they are zero-padded to match the length
            of the longest example in the mini-batch.

        Returns
        -------
        batched : list of torch.Tensor
            Batched model inputs. Same length as the number of model inputs.
        lengths : torch.Tensor
            Original input lengths along the last dimension. Can be important
            for post-processing, e.g. to ensure sample-wise losses are not
            aggregated over the zero-padded regions. Shape `(batch_size,
            n_model_inputs)`.

        Example
        -------
        >>> unbatched = [[torch.rand(2, 5), torch.rand(1)],
        ...              [torch.rand(2, 3), torch.rand(1)],
        ...              [torch.rand(2, 4), torch.rand(1)]]
        >>> unbatched
        [[tensor([[0.37, 0.09, 0.51, 0.41, 0.03],
                  [0.21, 0.25, 0.26, 0.65, 0.38]]),
          tensor([0.77])],

         [tensor([[0.99, 0.13, 0.02],
                  [0.01, 0.84, 0.48]]),
          tensor([0.14])],

         [tensor([[0.31, 0.10, 0.31, 0.57],
                  [0.29, 0.71, 0.19, 0.34]]),
          tensor([0.42])]]
        >>> batched, lengths = _collate_fn(unbatched)
        >>> [x.shape for x in batched]
        [torch.Size([3, 2, 5]), torch.Size([3, 1])]
        >>> batched
        [tensor([[[0.37, 0.09, 0.51, 0.41, 0.03],
                  [0.21, 0.25, 0.26, 0.65, 0.38]],

                  [[0.99, 0.13, 0.02, 0.00, 0.00],
                   [0.01, 0.84, 0.48, 0.00, 0.00]],

                  [[0.31, 0.10, 0.31, 0.57, 0.00],
                   [0.29, 0.71, 0.19, 0.34, 0.00]]]),
         tensor([[0.77],
                 [0.14],
                 [0.42]])]
        >>> lengths
        tensor([[5, 1],
                [3, 1],
                [4, 1]])
        """
        # convert to tuple if batch items are tensors
        inputs_are_tensors = isinstance(unbatched[0], torch.Tensor)
        unbatched = [
            (x,) if inputs_are_tensors else x for x in unbatched
        ]
        lengths = torch.tensor(
            [[x.shape[-1] for x in inputs] for inputs in unbatched],
            device=unbatched[0][0].device
        )
        batched = [
            torch.stack([
                F.pad(x, (0, max_length - x.shape[-1])) for x in inputs
            ])
            for inputs, max_length in zip(zip(*unbatched), lengths.amax(dim=0))
        ]
        # convert back to tensor if batch items were tensors
        if inputs_are_tensors:
            batched, = batched
            lengths = lengths.squeeze(-1)
        return batched, lengths


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

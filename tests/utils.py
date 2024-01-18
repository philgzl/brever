import random

import torch

from brever.data import BreverDataset
from brever.models.base import BreverBaseModel


class DummyDataset(BreverDataset):
    def __init__(self, n_examples, n_sources, n_channels, min_length,
                 max_length, transform=None):
        random_generator = random.Random(42)
        self._segment_info = [
            (i, (0, random_generator.randint(min_length, max_length)))
            for i in range(n_examples)
        ]
        torch_generator = torch.Generator().manual_seed(42)
        self.sources = [
            torch.randn((n_sources, n_channels, self._segment_info[i][1][1]),
                        generator=torch_generator)
            for i in range(n_examples)
        ]
        self.n_examples = n_examples
        self.transform = transform
        self._item_lengths = None
        self.preloaded_data = None
        self._duration = sum(x[1][1] for x in self._segment_info)/16000
        self._effective_duration = self._duration
        self.segment_strategy = 'pass'

    def __getitem__(self, index):
        if self.preloaded_data is not None:
            sources = self.preloaded_data[index]
        else:
            sources = self.sources[index]
            if self.transform is not None:
                sources = self.transform(sources)
        return sources

    def __len__(self):
        return self.n_examples


class DummyModel(BreverBaseModel):
    def __init__(self, channels=2, output_sources=1, use_transform=False,
                 criterion='snr'):
        super().__init__(criterion=criterion)
        self.conv = torch.nn.Conv1d(channels, channels*output_sources, 1)
        self.output_sources = output_sources
        self.channels = channels
        self.use_transform = use_transform
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = self.conv(x)
        return x.reshape(
            x.shape[0], self.output_sources, self.channels, x.shape[-1],
        )

    def transform(self, sources):
        # dummy pre-processing; trim inputs by factor 100
        if self.use_transform:
            sources = self.trim(sources)
        return sources

    def trim(self, x):
        return x[..., :self.transform_length(x.shape[-1])]

    def transform_length(self, segment_length):
        if self.use_transform:
            return segment_length//100
        else:
            return segment_length

    def loss(self, batch, lengths, use_amp):
        inputs, labels = batch[:, 0], batch[:, 1:]
        outputs = self(inputs)
        loss = self.criterion(outputs, labels, lengths)
        return loss.mean()

    def optimizers(self):
        return self.optimizer

    def _enhance(self, x, use_amp):
        return x.mean(-2)

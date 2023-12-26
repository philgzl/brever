import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ...modules import STFT, FeatureExtractor, MelFilterbank
from ..base import BreverBaseModel, ModelRegistry

eps = np.finfo(float).eps


@ModelRegistry.register('ffnn')
class FFNN(BreverBaseModel):
    def __init__(
        self,
        fs: int = 16000,
        features: set[str] = {'logfbe'},
        stacks: int = 5,
        decimation: int = 1,
        stft_frame_length: int = 512,
        stft_hop_length: int = 256,
        stft_window: str = 'hann',
        mel_filters: int = 64,
        hidden_layers: list[int] = [1024, 1024],
        dropout: float = 0.2,
        normalization: str = 'static',
        criterion: str = 'mse',
        optimizer: str = 'Adam',
        learning_rate: float = 0.0001,
    ):
        super().__init__(criterion=criterion)

        self.stacks = stacks
        self.decimation = decimation
        self.stft = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            window=stft_window,
        )
        self.mel_fb = MelFilterbank(
            n_filters=mel_filters,
            n_fft=stft_frame_length,
            fs=fs,
        )
        self.feature_extractor = FeatureExtractor(
            features=features,
            mel_fb=self.mel_fb,
            hop_length=stft_hop_length,
            fs=fs,
        )
        input_size = self.feature_extractor.n_features*(stacks+1)
        output_size = mel_filters
        self.ffnn = _FFNN(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        if normalization == 'static':
            self.normalization = StaticNormalizer(input_size)
        elif normalization == 'cumulative':
            self.normalization = CumulativeNormalizer()
        else:
            raise ValueError('unrecognized normalization type, got '
                             f'{normalization}')

        optimizer_cls = getattr(torch.optim, optimizer)
        self.optimizer = optimizer_cls(self.parameters(), lr=learning_rate)

    def optimizers(self):
        return self.optimizer

    def forward(self, x):
        x = self.normalization(x)
        x = self.ffnn(x)
        return x

    def transform(self, sources):
        assert sources.shape[0] == 2  # mixture, foreground
        sources = self.stft(sources)
        mix, foreground = sources
        background = mix - foreground
        foreground_mag = foreground.abs()  # (channels, freqs, frames)
        backgroung_mag = background.abs()  # (channels, freqs, frames)
        # features
        x = self.feature_extractor(mix)  # (feats, frames)
        x = self.stack(x)
        x = self.decimate(x)
        # labels
        labels = self.irm(foreground_mag, backgroung_mag)  # (labels, frames)
        labels = self.decimate(labels)
        return torch.cat([x, labels])

    def _step(self, batch, lengths):
        inputs = batch[:, :self.ffnn.input_size]
        labels = batch[:, self.ffnn.input_size:]
        outputs = self(inputs)
        loss = self.criterion(outputs, labels, lengths)
        return loss.mean()

    def train_step(self, batch, lengths, use_amp, scaler):
        self.optimizer.zero_grad()
        loss = self._step(batch, lengths)
        loss.backward()
        self.optimizer.step()
        return loss

    def val_step(self, batch, lengths, use_amp):
        return self._step(batch, lengths)

    def _enhance(self, x, use_amp):
        length = x.shape[-1]
        x = self.stft(x)
        features = self.feature_extractor(x)
        features = self.stack(features)
        features = self.normalization(features)
        mask = self.ffnn(features)
        mask_extrapolated = self.mel_fb.backward(mask)
        x = x.mean(1)  # average left and right channels
        x = self.stft.backward(x*mask_extrapolated)
        x = x[..., :length]
        return x

    def irm(self, foreground_mag, backgroung_mag):
        # (channels, freqs, frames)
        foreground_power = foreground_mag.pow(2).mean(0)  # (freqs, frames)
        backgroung_power = backgroung_mag.pow(2).mean(0)  # (freqs, frames)
        foreground_power = self.mel_fb(foreground_power)
        backgroung_power = self.mel_fb(backgroung_power)
        irm = (1 + backgroung_power/(foreground_power + eps)).pow(-0.5)
        return irm

    def stack(self, data):
        output = [data]
        for i in range(self.stacks):
            rolled = data.roll(i+1, -1)
            rolled[..., :i+1] = data[..., :1]
            output.append(rolled)
        if data.ndim == 2:  # unbatched
            cat_dim = 0
        else:  # batched
            cat_dim = 1
        return torch.cat(output, dim=cat_dim)

    def decimate(self, data):
        return data[..., ::self.decimation]

    def pre_train(self, dataset, dataloader, epochs):
        if isinstance(self.normalization, StaticNormalizer):
            logging.info('Calculating training statistics')
            mean, var = 0, 0
            for i in tqdm(range(len(dataset)), file=sys.stdout):
                data = dataset[i]
                inputs = data[:self.ffnn.input_size]
                mean += inputs.mean(-1, keepdim=True)
                var += inputs.pow(2).mean(-1, keepdim=True)
            mean, var = mean/len(dataset), var/len(dataset)
            var -= mean.pow(2)
            self.normalization.set_statistics(mean, var.sqrt())


class _FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[1024, 1024],
                 dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.module_list = nn.ModuleList()
        start_size = input_size
        for i in range(len(hidden_layers)):
            end_size = hidden_layers[i]
            self.module_list.append(nn.Linear(start_size, end_size))
            self.module_list.append(nn.ReLU())
            self.module_list.append(nn.Dropout(dropout))
            start_size = end_size
        self.module_list.append(nn.Linear(start_size, output_size))
        self.module_list.append(nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)
        for module in self.module_list:
            x = module(x)
        return x.transpose(1, 2)


class StaticNormalizer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        mean = torch.zeros((input_size, 1))
        std = torch.ones((input_size, 1))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def set_statistics(self, mean, std):
        self.mean[:], self.std[:] = mean, std

    def forward(self, x):
        return (x - self.mean)/self.std


class CumulativeNormalizer(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        cum_sum = x.cumsum(-1)
        cum_pow_sum = x.pow(2).cumsum(-1)
        count = torch.arange(1, x.shape[-1]+1, device=x.device)
        count = count.reshape(1, 1, x.shape[-1])
        cum_mean = cum_sum/count
        cum_var = cum_pow_sum/count - cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()
        return (x - cum_mean)/cum_std

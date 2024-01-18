# Copyright 2020 Szu-Wei Fu
# Copyright 2021 Peter Plantinga
# https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN
#
# Copyright 2023 Wooseok Shin
# https://github.com/wooseok-shin/MetricGAN-OKD/tree/main
#
# Copyright 2023 Panagiotis Apostolidis
# Copyright 2024 Philippe Gonzalez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn

from ...metrics import MetricRegistry
from ...modules import STFT
from ..base import BreverBaseModel, ModelRegistry


@ModelRegistry.register('metricganokd')
class MetricGANOKD(BreverBaseModel):
    # Key differences with SpeechBrain and wooseok-shin/MetricGAN-OKD:
    # - The generator and discriminator are alternatively updated for each
    # mini-batch, like in traditional GANs, instead of iterating over the
    # entire dataset once to train the discriminator, and once again to train
    # the generator for each epoch.
    # - The entire dataset is iterated for each epoch, instead of iterating
    # over a limited number of samples.
    # - No historical dataset for the discriminator. Instead we augment the
    # clean speech with noise at random SNRs.
    # - The discriminator is not trained on the clean speech, as I found this
    # makes it quickly get stuck in a mode.
    #
    # Despite my efforts, I can't get this model to work.
    def __init__(
        self,
        stft_frame_length: int = 512,
        stft_hop_length: int = 256,
        stft_n_fft: int = 512,
        stft_normalized: bool = False,
        stft_window: str = 'hamming',
        generator_lstm_hidden_size: int = 200,
        generator_lstm_num_layers: int = 2,
        generator_lstm_dropout: float = 0.1,
        generator_lstm_bidirectional: bool = True,
        generator_fc_channels: list[int] = [300],
        generator_optimizer: str = 'Adam',
        generator_learning_rate: float = 1e-4,
        discriminator_conv_channels: list[int] = [16, 32, 64, 128],
        discriminator_fc_channels: list[int] = [50, 10],
        discriminator_batchnorm: bool = True,
        discriminator_batchnorm_momentum: float = 0.01,
        discriminator_sigmoid_output: bool = False,
        discriminator_optimizer: str = 'Adam',
        discriminator_learning_rate: float = 1e-4,
        discriminator_train_clean: bool = False,
        augment: bool = True,
        augment_snr_min: float = 15.0,
        augment_snr_max: float = 55.0,
        target_metrics: list[str] = ['pesq'],
        inference_metric: str = 'pesq',
        xavier_init: bool = True,
        precompute_stft: bool = True,
        precompute_metric: bool = True,
        mag_eps: float = 1e-14,
        min_mask: float = 0.05,
        adversarial_loss: str = 'MSELoss',
        kd_loss: str = 'MSELoss',
        kd_weight: float = 10.0,
        grad_clip: float = 5.0,
        fs: int = 16000,
    ):
        super().__init__()

        self.stft_module = STFT(
            frame_length=stft_frame_length,
            hop_length=stft_hop_length,
            n_fft=stft_n_fft,
            window=stft_window,
            normalized=stft_normalized,
        )

        self.metrics = Metrics(target_metrics, fs)

        if inference_metric not in target_metrics:
            raise ValueError("inference_metric must be one of target_metrics, "
                             f"got '{inference_metric}' and {target_metrics}")
        self.inference_metric_idx = target_metrics.index(inference_metric)

        self.generators = nn.ModuleList(
            Generator(
                lstm_in_size=stft_n_fft // 2 + 1,
                lstm_hidden_size=generator_lstm_hidden_size,
                lstm_num_layers=generator_lstm_num_layers,
                lstm_dropout=generator_lstm_dropout,
                lstm_bidirectional=generator_lstm_bidirectional,
                fc_channels=generator_fc_channels,
                xavier_init=xavier_init,
            )
            for target_metric in target_metrics
        )
        self.discriminator = Discriminator(
            out_size=len(target_metrics),
            conv_channels=discriminator_conv_channels,
            fc_channels=discriminator_fc_channels,
            batchnorm=discriminator_batchnorm,
            batchnorm_momentum=discriminator_batchnorm_momentum,
            sigmoid_output=discriminator_sigmoid_output,
            xavier_init=xavier_init,
        )

        self.generator_optimizers = [
            self.init_optimizer(
                generator_optimizer,
                generator,
                lr=generator_learning_rate,
            )
            for generator in self.generators
        ]
        self.discriminator_optimizer = self.init_optimizer(
            discriminator_optimizer,
            self.discriminator,
            lr=discriminator_learning_rate,
        )

        if augment:
            self.augmentor = Augmentor(augment_snr_min, augment_snr_max)
        else:
            self.augmentor = None

        self.adversarial_loss = getattr(nn, adversarial_loss)()
        self.kd_loss = getattr(nn, kd_loss)()

        self.discriminator_train_clean = False
        self.kd_weight = kd_weight
        self.precompute_stft = precompute_stft
        self.precompute_metric = precompute_metric
        self.mag_eps = mag_eps
        self.grad_clip = grad_clip

    def optimizers(self):
        return *self.generator_optimizers, self.discriminator_optimizer

    def forward(self, noisy_mag):
        return [generator(noisy_mag) for generator in self.generators]

    def _enhance(self, noisy_wav, use_amp):
        noisy_wav = noisy_wav.mean(axis=-2)  # make monaural
        noisy_mag, noisy_phase = self.stft(noisy_wav)
        generator = self.generators[self.inference_metric_idx]
        enh_mag = generator(noisy_mag)
        return self.istft(enh_mag, noisy_phase, noisy_wav.shape[-1])

    def transform(self, sources):
        assert sources.shape[0] == 2  # mixture, foreground
        sources = sources.mean(axis=-2)  # make monaural
        output = [sources]
        if self.precompute_stft:
            output += [*self.stft(sources)]
        if self.precompute_metric:
            output.append(self.metrics(sources[0], sources[1]))
        return output

    def stft(self, x):
        x = self.stft_module(x)
        mag = torch.log1p(x.abs() + self.mag_eps)
        phase = x.angle()
        return mag, phase

    def istft(self, mag, phase, orig_length):
        mag = torch.expm1(mag)
        x = mag * torch.exp(1j * phase)
        x = self.stft_module.backward(x)
        return x[..., :orig_length]

    def load_batch(self, batch, lengths):
        if self.precompute_stft and self.precompute_metric:
            wavs, mags, phases, true_noisy_score = batch
        elif self.precompute_stft:
            wavs, mags, phases = batch
        elif self.precompute_metric:
            wavs, true_noisy_score = batch
        else:
            wavs, = batch
        if not self.precompute_stft:
            mags, phases = self.stft(wavs)
        if not self.precompute_metric:
            true_noisy_score = self.metrics(
                wavs[:, 0],
                wavs[:, 1],
                lengths=lengths[:, 0],
            )
        noisy_wav, clean_wav = wavs[:, 0], wavs[:, 1]
        noisy_mag, clean_mag = mags[:, 0], mags[:, 1]
        noisy_phase, _ = phases[:, 0], phases[:, 1]
        return noisy_wav, clean_wav, noisy_mag, clean_mag, noisy_phase, \
            true_noisy_score

    def train_step(self, batch, lengths, use_amp, scaler):
        noisy_wav, clean_wav, noisy_mag, clean_mag, noisy_phase, \
            true_noisy_score = self.load_batch(batch, lengths)
        loss_g, enh_mags = self.generator_train_step(noisy_mag, clean_mag,
                                                     scaler)
        loss_d = self.discriminator_train_step(noisy_mag, clean_mag, enh_mags,
                                               noisy_wav, clean_wav,
                                               noisy_phase, lengths, scaler)
        return dict(loss_g=torch.stack(loss_g).mean(), loss_d=loss_d)

    def val_step(self, batch, lengths, use_amp):
        noisy_wav, clean_wav, noisy_mag, clean_mag, noisy_phase, \
            true_noisy_score = self.load_batch(batch, lengths)
        loss_g, enh_mags = self.generator_loss(noisy_mag, clean_mag)
        loss_d = self.discriminator_loss(noisy_mag, clean_mag, enh_mags,
                                         noisy_wav, clean_wav, noisy_phase,
                                         lengths)
        return dict(loss_g=torch.stack(loss_g).mean(), loss_d=loss_d)

    def generator_train_step(self, noisy_mag, clean_mag, scaler):
        for generator_optimizer in self.generator_optimizers:
            generator_optimizer.zero_grad()
        losses, enh_mags = self.generator_loss(noisy_mag, clean_mag)
        for loss, generator, generator_optimizer in zip(
            losses, self.generators, self.generator_optimizers
        ):
            self.update(loss, scaler, net=generator,
                        optimizer=generator_optimizer,
                        grad_clip=self.grad_clip, retain_graph=True)
        return losses, enh_mags

    def discriminator_train_step(self, noisy_mag, clean_mag, enh_mag,
                                 noisy_wav, clean_wav, noisy_phase, lengths,
                                 scaler):
        self.discriminator_optimizer.zero_grad()
        loss = self.discriminator_loss(noisy_mag, clean_mag, enh_mag,
                                       noisy_wav, clean_wav, noisy_phase,
                                       lengths)
        self.update(loss, scaler, net=self.discriminator,
                    optimizer=self.discriminator_optimizer,
                    grad_clip=self.grad_clip)
        return loss

    def generator_loss(self, noisy_mag, clean_mag):
        enh_mags = self(noisy_mag)
        dists = self.pairwise_distances(enh_mags)
        losses = []
        for i, (generator, enh_mag) in enumerate(zip(self.generators,
                                                     enh_mags)):
            scores = self.discriminator(enh_mag, clean_mag)
            loss = self.adversarial_loss(
                scores[:, i],
                torch.ones(enh_mag.shape[0], device=enh_mag.device),
            ) + self.kd_weight * dists[i].sum()
            losses.append(loss)
        # detach for the discriminator
        enh_mags = [enh_mag.detach() for enh_mag in enh_mags]
        return losses, enh_mags

    def discriminator_loss(self, noisy_mag, clean_mag, enh_mag, noisy_wav,
                           clean_wav, noisy_phase, lengths):

        if isinstance(enh_mag, list):
            return sum(self.discriminator_loss(noisy_mag, clean_mag, x,
                                               noisy_wav, clean_wav,
                                               noisy_phase, lengths)
                       for x in enh_mag)

        enh_wav = self.istft(enh_mag, noisy_phase, noisy_wav.shape[-1])

        score_noisy = self.metrics(noisy_wav, clean_wav, lengths=lengths[:, 0])
        loss_noisy = self.adversarial_loss(
            self.discriminator(noisy_mag, clean_mag),
            score_noisy,
        )

        score_enh = self.metrics(enh_wav, clean_wav, lengths=lengths[:, 0])
        loss_enh = self.adversarial_loss(
            self.discriminator(enh_mag, clean_mag),
            score_enh,
        )

        if self.discriminator_train_clean:
            score_clean = torch.ones(clean_mag.shape[0], len(self.generators),
                                     device=clean_mag.device)
            loss_clean = self.adversarial_loss(
                self.discriminator(clean_mag, clean_mag),
                score_clean
            )
        else:
            loss_clean = 0

        if self.augmentor is not None:
            aug_wav = self.augmentor.augment(clean_wav)
            aug_mag, _ = self.stft(aug_wav)
            score_aug = self.metrics(aug_wav, clean_wav, lengths=lengths[:, 0])
            loss_aug = self.adversarial_loss(
                self.discriminator(aug_mag, clean_mag),
                score_aug,
            )
        else:
            loss_aug = 0

        return loss_noisy + loss_enh + loss_clean + loss_aug

    def pairwise_distances(self, enh_mags):
        # there is probably a more efficient way to compute this
        return torch.tensor([
            [
                self.kd_loss(enh_mag_i, enh_mag_j)
                for enh_mag_j in enh_mags
            ]
            for enh_mag_i in enh_mags
        ])


class LearnableSigmoid(nn.Module):
    def __init__(self, in_size=257, beta=1.2):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_size))
        self.beta = beta

    def forward(self, x):
        return self.beta * torch.sigmoid(self.alpha * x)


class Linear(nn.Module):
    def __init__(self, in_size, out_size, spec_norm=True, leaky_relu=True,
                 leaky_relu_slope=0.3, xavier_init=True):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        if spec_norm:
            self.fc = nn.utils.spectral_norm(self.fc)
        if xavier_init:
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
        if leaky_relu:
            self.activation = nn.LeakyReLU(leaky_relu_slope)
        else:
            self.activation = None

    def forward(self, x):
        x = self.fc(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(5, 5), spec_norm=True,
                 leaky_relu=True, leaky_relu_slope=0.3, xavier_init=True):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size)
        if spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        if xavier_init:
            nn.init.xavier_uniform_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        if leaky_relu:
            self.activation = nn.LeakyReLU(leaky_relu_slope)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Generator(nn.Module):
    def __init__(self, lstm_in_size=257, lstm_hidden_size=200,
                 lstm_num_layers=2, lstm_dropout=0.0, lstm_bidirectional=True,
                 fc_channels=[300], min_mask=0.05, xavier_init=True):
        super().__init__()
        self.min_mask = min_mask

        self.lstm = nn.LSTM(lstm_in_size, lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            dropout=lstm_dropout,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)

        if lstm_bidirectional:
            lstm_hidden_size *= 2

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_channels) + 1):
            self.fc_layers.append(Linear(
                in_size=lstm_hidden_size if i == 0 else fc_channels[i - 1],
                out_size=lstm_in_size if i == len(fc_channels) else fc_channels[i],  # noqa: E501
                leaky_relu=i != len(fc_channels),
                spec_norm=False,
                xavier_init=xavier_init,
            ))

        self.learnable_sigmoid = LearnableSigmoid()

    def predict_mask(self, x):
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.learnable_sigmoid(x)
        x = x.transpose(1, 2)
        return x

    def forward(self, noisy_mag):
        mask = self.predict_mask(noisy_mag)
        return noisy_mag * mask.clamp(min=self.min_mask)


class Discriminator(nn.Module):
    def __init__(self, out_size=1, conv_channels=[16, 32, 64, 128],
                 fc_channels=[50, 10], batchnorm=True, batchnorm_momentum=0.01,
                 sigmoid_output=False, xavier_init=True):
        super().__init__()
        in_size = 2  # input_mag, clean_mag
        self.sigmoid_output = sigmoid_output

        if batchnorm:
            self.norm = nn.BatchNorm2d(in_size, momentum=batchnorm_momentum)
        else:
            self.norm = None

        self.conv_layers = nn.ModuleList()
        for i in range(len(conv_channels)):
            self.conv_layers.append(Conv2d(
                in_size=in_size if i == 0 else conv_channels[i - 1],
                out_size=conv_channels[i],
                xavier_init=xavier_init,
            ))

        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_channels) + 1):
            self.fc_layers.append(Linear(
                in_size=conv_channels[-1] if i == 0 else fc_channels[i - 1],
                out_size=out_size if i == len(fc_channels) else fc_channels[i],
                leaky_relu=i != len(fc_channels),
                xavier_init=xavier_init,
            ))

    def forward(self, input_mag, clean_mag):
        x = torch.stack([input_mag, clean_mag], dim=1)
        x = x.transpose(2, 3)
        if self.norm is not None:
            x = self.norm(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.mean(x, (2, 3))  # average pooling
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class Augmentor:
    def __init__(self, snr_min, snr_max):
        self.snr_min = snr_min
        self.snr_max = snr_max

    def augment(self, clean_wav):
        snr = torch.rand(clean_wav.shape[0], 1, device=clean_wav.device) \
            * (self.snr_max - self.snr_min) + self.snr_min
        noise_std = clean_wav.std(dim=1, keepdim=True) * 10 ** (-snr / 20)
        return clean_wav + noise_std * torch.randn_like(clean_wav)


class Metrics:
    def __init__(self, metrics, fs):
        self.metrics = [init_metric_func(metric, fs) for metric in metrics]

    def __call__(self, x, y, lengths=None):
        output = torch.stack([
            metric(x, y, lengths=lengths)
            for metric in self.metrics
        ], dim=1).float().to(x.device)
        return output.clamp(min=0.0, max=1.0)


@ModelRegistry.register('metricganp')
class MetricGANp(MetricGANOKD):

    _is_submodel = True

    def __init__(
        self,
        generator_lstm_dropout: float = 0.0,
        discriminator_conv_channels: list[int] = [15, 15, 15, 15],
        **kwargs,
    ):
        super().__init__(
            generator_lstm_dropout=generator_lstm_dropout,
            discriminator_conv_channels=discriminator_conv_channels,
            **kwargs,
        )


def init_metric_func(metric, fs):
    raw_metric_func = MetricRegistry.get(metric)
    kwargs = {'normalized': True} if metric == 'pesq' else {}

    def metric_func(x, y, lengths=None):
        score = raw_metric_func(x, y, fs, lengths=lengths, **kwargs)
        if isinstance(score, float):
            return torch.tensor([score])
        else:
            return torch.from_numpy(score)

    return metric_func

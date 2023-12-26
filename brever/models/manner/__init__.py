import torch

from ..base import BreverBaseModel, ModelRegistry
from .config import get_config
from .models_small import MANNER as MANNER_SMALL
from .stft_loss import MultiResolutionSTFTLoss
from .time_loss import L1CharbonnierLoss, L1Loss, WeightedLoss
from .utils import args_dict


@ModelRegistry.register('manner')
class MANNER(BreverBaseModel):
    def __init__(self):
        super().__init__()
        self.args = args_dict(get_config())
        self.net = MANNER_SMALL(**self.args.manner)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.args.learning_rate,
        )
        self.criterion = self.select_loss()
        self.stft_loss = MultiResolutionSTFTLoss(
            factor_sc=self.args.stft_sc_factor,
            factor_mag=self.args.stft_mag_factor,
        )

    def forward(self, x):
        return self.net(x)

    def pre_train(self, dataset, dataloader, epochs):
        self._scheduler_kwargs = dict(
            max_lr=0.001,
            steps_per_epoch=len(dataloader),
            epochs=epochs,
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, **self._scheduler_kwargs,
        )

    def optimizers(self):
        return self.optimizer

    def transform(self, sources):
        assert sources.shape[0] == 2  # mixture, foreground
        sources = sources.mean(axis=-2)  # make monaural
        return sources

    def _enhance(self, x, use_amp):
        x = x.mean(axis=-2, keepdims=True)  # (batch_size, 1, length)
        device = x.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            x = self.forward(x)  # (batch_size, 1, length)
        return x.squeeze(1)

    def _step(self, batch, lengths, use_amp):
        mix, clean = batch[:, [0]], batch[:, 1:]
        noise = mix - clean
        device = batch.device.type
        dtype = torch.bfloat16 if device == 'cpu' else torch.float16

        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):

            clean_est = self(mix)
            noise_est = mix - clean_est

            clean = clean.squeeze(1)
            noise = noise.squeeze(1)
            clean_est = clean_est.squeeze(1)
            noise_est = noise_est.squeeze(1)

            loss = self.criterion(clean, clean_est)
            noise_loss = self.criterion(noise, noise_est)

            if self.args.stft_loss:
                sc_loss, mag_loss = self.stft_loss(clean_est, clean)
                loss += sc_loss + mag_loss
                sc_loss, mag_loss = self.stft_loss(noise_est, noise)
                noise_loss += sc_loss + mag_loss

            loss = WeightedLoss(clean, noise, loss, noise_loss)

        return loss.mean()

    def train_step(self, batch, lengths, use_amp, scaler):
        self.optimizer.zero_grad()
        loss = self._step(batch, lengths, use_amp)
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        self.scheduler.step()
        return loss

    def val_step(self, batch, lengths, use_amp):
        return self._step(batch, lengths, use_amp)

    def select_loss(self):
        if self.args.loss == 'l1':
            criterion = L1Loss()
        elif self.args.loss == 'l2':
            criterion = torch.nn.MSELoss()
        elif self.args.loss == 'ch':
            criterion = L1CharbonnierLoss()
        return criterion

    def state_dict(self):
        return {
            'net': self.net.state_dict(),
            'scheduler': {
                'state_dict': self.scheduler.state_dict(),
                'kwargs': self._scheduler_kwargs,
            }
        }

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self._scheduler_kwargs = state_dict['scheduler']['kwargs']
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, **self._scheduler_kwargs,
        )

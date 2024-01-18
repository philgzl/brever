import functools
import math

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

from ..utils import fft_freqs


class STFT:
    """Short-time Fourier transform (STFT).

    A wrapper for `torch.stft` and `torch.istft` because they have some
    limitations:
    - They do not allow `str` or callable input for `window`. See
    https://github.com/pytorch/pytorch/issues/88919
    - They use a wrong normalization factor: See
    https://github.com/pytorch/pytorch/issues/81428
    - `torch.stft` discards trailing samples if they do not fit a frame. This
    can happen even if `center=True`. This means data can be lost! See
    https://github.com/pytorch/pytorch/issues/70073
    - Using `center=False` and a non-rectangular window produces a
    RuntimeError. See https://github.com/pytorch/pytorch/issues/91309
    - They do not support arbitrary input shapes.
    Despite those limitations `torch.stft` and `torch.istft` are still used
    here due to their efficiency. As a consequence one should always use
    `center=True` to prevent complications until these issues are fixed.
    """

    def __init__(self, frame_length=512, hop_length=256, window='hann',
                 center=True, pad_mode='constant', normalized=True,
                 onesided=True, compression_factor=1, scale_factor=1,
                 n_fft=None):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.onesided = onesided
        self.compression_factor = compression_factor
        self.scale_factor = scale_factor
        self.n_fft = frame_length if n_fft is None else n_fft

        if isinstance(window, str):
            window = functools.partial(scipy.signal.get_window, window)
        if callable(window):
            window = window(frame_length)
        if isinstance(window, np.ndarray):
            window = torch.from_numpy(window)
        self.window = window

    def __call__(self, x, return_type='complex'):
        return self.forward(x, return_type=return_type)

    def forward(self, x, return_type='complex'):
        window = self.window.type(x.dtype).to(x.device)

        x = self.pad(x)

        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        x = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,  # avoid default normalization
            onesided=self.onesided,
            return_complex=True,
        )

        # apply correct normalization
        # TODO: add optional amplitude-normalization as in scipy
        # TODO: handle DC and Nyquist components
        if self.normalized:
            x /= window.pow(2).sum().sqrt()

        if self.compression_factor != 1:
            x = x.abs().pow(self.compression_factor)*torch.exp(1j*x.angle())
        x *= self.scale_factor

        x = x.view(*input_shape[:-1], *x.shape[-2:])

        if return_type == 'complex':
            return x
        elif return_type == 'real_imag':
            return x.real, x.imag
        elif return_type == 'mag_phase':
            return x.abs(), x.angle()
        else:
            raise ValueError('return_type must be complex, real_imag or '
                             f'mag_phase, got {return_type}')

    def backward(self, x, input_type='complex'):
        if input_type == 'real_imag':
            real, imag = x
            x = torch.complex(real, imag)
        elif input_type == 'mag_phase':
            mag, phase = x
            x = mag*torch.exp(1j*phase)
        elif input_type != 'complex':
            raise ValueError('input_type must be complex, real_imag or '
                             f'mag_phase, got {input_type}')

        window = self.window.type(x.real.dtype).to(x.device)

        x /= self.scale_factor
        if self.compression_factor != 1:
            x = x.abs().pow(1/self.compression_factor)*torch.exp(1j*x.angle())

        # apply correct normalization
        # TODO: add optional amplitude-normalization as in scipy
        # TODO: handle DC and Nyquist components
        if self.normalized:
            x *= window.pow(2).sum().sqrt()

        input_shape = x.shape
        x = x.reshape(-1, *input_shape[-2:])
        x = torch.istft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window=window,
            center=self.center,
            normalized=False,  # avoid default normalization
            onesided=self.onesided,
            return_complex=False,
        )

        return x.view(*input_shape[:-2], -1)

    def pad(self, x):
        samples = x.shape[-1]
        frames = self.frame_count(samples)
        padding = (frames - 1)*self.hop_length + self.frame_length - samples
        return F.pad(x, (0, padding), mode=self.pad_mode)

    def frame_count(self, samples):
        # /!\ THIS IS THE FRAME COUNT WITHOUT THE PADDING FROM TORCH.STFT
        # OF NFFT//2 LEFT AND RIGHT /!\
        return math.ceil(max(samples-self.frame_length, 0)/self.hop_length)+1


class MelFilterbank:
    def __init__(self, n_filters=64, n_fft=512, fs=16e3, fmin=50, fmax=8000):
        self.n_filters = n_filters
        self.n_fft = n_fft
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.filters, self.fc, self.scaling = self.calc_filterbank()

    def calc_filterbank(self):
        mel_min = self.freq_to_mel(self.fmin)
        mel_max = self.freq_to_mel(self.fmax)
        mel = torch.linspace(mel_min, mel_max, self.n_filters+2)
        fc = self.mel_to_freq(mel)
        f = fft_freqs(self.fs, self.n_fft)
        f = torch.from_numpy(f).float()
        filters = torch.zeros((self.n_filters, len(f)))
        for i_filt, i in enumerate(range(1, self.n_filters+1)):
            mask = (fc[i-1] <= f) & (f <= fc[i])
            filters[i_filt, mask] = (f[mask]-fc[i-1])/(fc[i]-fc[i-1])
            mask = (fc[i] <= f) & (f <= fc[i+1])
            filters[i_filt, mask] = (fc[i+1]-f[mask])/(fc[i+1]-fc[i])
        scaling = filters.sum(axis=1, keepdims=True)
        filters /= scaling
        return filters, fc, scaling

    @staticmethod
    def mel_to_freq(mel):
        return 700*(10**(mel/2595) - 1)

    @staticmethod
    def freq_to_mel(f):
        return 2595*math.log10(1 + f/700)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.matmul(self.filters.to(x.device), x)

    def backward(self, x):
        return torch.matmul(self.inverse_filters.to(x.device), x)

    @property
    def inverse_filters(self):
        filters = self.filters*self.scaling
        return filters.T


class ConvSTFT:
    def __init__(self, frame_length=512, hop_length=256, window='hann',
                 compression_factor=1, scale_factor=1, normalized=True):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.compression_factor = compression_factor
        self.scale_factor = scale_factor
        self.normalized = normalized

        if isinstance(window, str):
            window = scipy.signal.get_window(window, frame_length)**0.5
        if isinstance(window, np.ndarray):
            window = torch.from_numpy(window)
        self.window = window

        filters = torch.fft.fft(torch.eye(frame_length))
        filters = filters[:frame_length//2+1]

        # below is the correct scaling for seamless analysis-synthesis, i.e.
        # such that:
        # ```
        # y = F.conv1d(x, filters, stride=hop_length)
        # z = F.conv_transpose1d(y, filters, stride=hop_length)
        # assert torch.isclose(x, y).all()
        # ```
        filters[0, :] /= 2**0.5
        self._normalization_factor = 0.5*frame_length/hop_length**0.5
        if normalized:
            filters /= self._normalization_factor

        filters *= window
        filters = torch.cat([filters.real, filters.imag])

        filters = filters.unsqueeze(1).float()
        self.filters = filters

    def __call__(self, x, return_type='complex'):
        return self.forward(x, return_type=return_type)

    def forward(self, x, return_type='complex'):
        x = self.pad(x)
        input_shape = x.shape
        x = x.view(-1, 1, input_shape[-1])
        output = F.conv1d(x, self.filters, stride=self.hop_length)
        dim = self.frame_length//2 + 1
        real = output[:, :dim, :].view(*input_shape[:-1], dim, -1)
        imag = output[:, dim:, :].view(*input_shape[:-1], dim, -1)

        if self.compression_factor != 1:
            r = (real.pow(2) + imag.pow(2)).sqrt()
            r = r.pow(self.compression_factor)
            theta = torch.atan2(imag, real)
            real, imag = r*torch.cos(theta), r*torch.sin(theta)
        real *= self.scale_factor
        imag *= self.scale_factor

        if return_type == 'real_imag':
            return real, imag
        elif return_type == 'mag_phase':
            mag = (real.pow(2) + imag.pow(2)).pow(0.5)
            phase = torch.atan2(imag, real)
            return mag, phase
        elif return_type == 'complex':
            return torch.complex(real, imag)
        else:
            raise ValueError('return_type must be complex, real_imag or '
                             f'mag_phase, got {return_type}')

    def backward(self, x, input_type='complex'):
        if input_type == 'real_imag':
            real, imag = x
        elif input_type == 'mag_phase':
            mag, phase = x
            real = mag*torch.cos(phase)
            imag = mag*torch.sin(phase)
        elif input_type == 'complex':
            real, imag = x.real, x.imag
        else:
            raise ValueError('input_type must be complex, real_imag or '
                             f'mag_phase, got {input_type}')

        real /= self.scale_factor
        imag /= self.scale_factor
        if self.compression_factor != 1:
            r = (real.pow(2) + imag.pow(2)).sqrt()
            r = r.pow(1/self.compression_factor)
            theta = torch.atan2(imag, real)
            real, imag = r*torch.cos(theta), r*torch.sin(theta)

        x = torch.cat([real, imag], dim=-2)
        input_shape = x.shape
        x = x.view(-1, input_shape[-2], input_shape[-1])
        x = F.conv_transpose1d(x, self.filters, stride=self.hop_length)

        if not self.normalized:
            x /= self._normalization_factor**2

        # remove left and right padding for perfect reconstruction
        padding = self.frame_length - self.hop_length
        x = x[..., padding:-padding]

        return x.view(*input_shape[:-2], -1)

    def pad(self, x):
        # pad right to get integer number of frames
        samples = x.shape[-1]
        frames = self.frame_count(samples)
        padding = (frames - 1)*self.hop_length + self.frame_length - samples
        x = F.pad(x, (0, padding))
        # pad left and right for perfect reconstruction
        padding = self.frame_length - self.hop_length
        x = F.pad(x, (padding, padding))
        return x

    def frame_count(self, samples):
        return math.ceil(max(samples-self.frame_length, 0)/self.hop_length)+1

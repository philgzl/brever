import numpy as np
import scipy.signal

from ..utils import fft_freqs, pad


def rms(x, axis=0):
    """Root mean square (RMS) along a given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which to calculate. Default is `0`.

    Returns
    -------
    rms : array_like
        RMS values.
    """
    return np.mean(x**2, axis=axis)**0.5


def spatialize(x, brir):
    """Signal spatialization.

    Spatializes a single channel input with a binaural room impulse response
    (BRIR) using the overlap-add convolution method. The last samples in the
    output signal are discarded such that the length matches the input signal.

    Parameters
    ----------
    x : array_like
        Monaural audio signal to spatialize. Shape `(length,)`.
    brir : array_like
        BRIR. Shape `(brir_length, 2)`.

    Returns
    -------
    y: array_like
        Binaural audio signal. Shape `(length, 2)`.
    """
    n = len(x)
    x_left = scipy.signal.oaconvolve(x, brir[:, 0], mode='full')[:n]
    x_right = scipy.signal.oaconvolve(x, brir[:, 1], mode='full')[:n]
    return np.vstack([x_left, x_right]).T


def colored_noise(color, n_samples, seed=None):
    """Colored noise with `1/f**alpha` power spectral density.

    Parameters
    ----------
    color : {'brown', 'pink', 'white', 'blue', 'violet'}
        Noise color.
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    x: array_like
        Colored noise. Shape `(n_samples,)`.
    """
    colors = {
        'brown': 2,
        'pink': 1,
        'white': 0,
        'blue': -1,
        'violet': -2,
    }
    if color not in colors.keys():
        raise ValueError('color must be either one of %s' % colors.keys())
    alpha = colors[color]
    scaling = fft_freqs(fs=1, n_fft=n_samples)
    scaling[0] = scaling[1]
    scaling **= -alpha/2
    x = np.random.RandomState(seed).randn(n_samples)
    X = np.fft.rfft(x)
    X *= scaling
    x = np.fft.irfft(X, n_samples).real
    return x


def match_ltas(x, ltas, n_fft=512, hop_length=256):
    """Long-term-average-spectrum (LTAS) matching.

    Filters the input signal in the short-time Fourier transform (STFT) domain
    such that it presents a specific LTAS.

    Parameters
    ----------
    x : array_like
        Input signal. Shape `(n_samples, n_channels)`.
    ltas : array_like
        Desired LTAS. Shape `(n_fft,)`.
    n_fft : int, optional
        Number of FFT points. Default is `512`.
    hop_length : int, optional
        Frame shift in samples. Default is `256`.

    Returns
    -------
    y : array_like
        Output signal with LTAS equal to `ltas`.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        flat_output = True
    else:
        flat_output = False
    n = len(x)
    noverlap = n_fft-hop_length
    _, _, X = scipy.signal.stft(x, nperseg=n_fft, noverlap=noverlap, axis=0)
    ltas_X = np.mean(np.abs(X**2), axis=(1, 2))
    EQ = (ltas/ltas_X)**0.5
    X *= EQ[:, np.newaxis, np.newaxis]
    _, x = scipy.signal.istft(X, nperseg=n_fft, noverlap=noverlap, freq_axis=0)
    x = x.T
    if flat_output:
        x = x.ravel()
    return x[:n]


def split_brir(brir, reflection_boundary=50e-3, fs=16e3, max_itd=1e-3):
    """Binaural room impulse response (BRIR) splitting.

    Splits a BRIR into an early reflection component and a late reflection
    component using a reflection boundary.

    Parameters
    ----------
    brir : array_like
        Input BRIR. Shape `(brir_length, 2)`.
    reflection_boundary : float, optional
        Reflection boundary in seconds. This is the limit between early and
        late reflections. Default is `50e-3`.
    fs : int or float, optional
        Sampling frequency. Default is `16e3`.
    max_itd : float, optional
        Maximum interaural time difference in seconds. Used to correct the
        location of the impulse peak in each channel if the initial estimates
        are more than `max_itd` seconds apart. Default is `1e-3`.

    Returns
    -------
    brir_early: array_like
        Early reflection part of the input BRIR. Shape `(brir_length, 2)`.
    brir_late: array_like
        Late reflection part of the input BRIR. Shape `(brir_length, 2)`.
    """
    peak_i = np.argmax(np.abs(brir), axis=0)
    peak_val = np.max(np.abs(brir), axis=0)
    max_delay = round(max_itd*fs)
    if peak_val[0] > peak_val[1]:
        segment = np.abs(brir[peak_i[0]:peak_i[0]+max_delay, 1])
        peak_i[1] = peak_i[0] + np.argmax(segment)
    else:
        segment = np.abs(brir[peak_i[1]:peak_i[1]+max_delay, 0])
        peak_i[0] = peak_i[1] + np.argmax(segment)
    win_early = np.zeros(brir.shape)
    win_early[:peak_i[0] + round(reflection_boundary*fs), 0] = 1
    win_early[:peak_i[1] + round(reflection_boundary*fs), 1] = 1
    win_late = 1 - win_early
    brir_early = win_early*brir
    brir_late = win_late*brir
    return brir_early, brir_late


def adjust_snr(signal, noise, snr, slice_=None):
    """Signal-to-noise ratio (SNR) adjustment.

    Scales a noise signal given a target signal and a desired SNR.

    Parameters
    ----------
    signal: array_like
        Target signal. Shape `(n_samples, n_channels)`.
    noise: array
        Noise signal. Shape `(n_samples, n_channels)`.
    snr:
        Desired SNR.
    slice_: slice or None, optional
        Slice of the target and noise signals from which the SNR should be
        calculated. Default is `None`, which means the energy of the entire
        signals are calculated.

    Returns
    -------
    noise_scaled: array_like
        Scaled noise. The SNR between the target signal and the new scaled
        noise is equal to `snr`.
    """
    if slice_ is None:
        slice_ = np.s_[:]
    energy_signal = np.sum(signal[slice_].mean(axis=1)**2)
    energy_noise = np.sum(noise[slice_].mean(axis=1)**2)
    if energy_signal == 0:
        raise ValueError('cannot scale noise signal if target signal is 0')
    if energy_noise == 0:
        raise ValueError('cannot scale noise signal if it equals 0')
    gain = (10**(-snr/10)*energy_signal/energy_noise)**0.5
    noise_scaled = gain*noise
    return noise_scaled, gain


def adjust_rms(signal, rms_dB):
    """Root-mean-square (RMS) adjustment.

    Scales a signal to a desired RMS in dB. Note that the dB value is relative
    to 1, which means a signal with an RMS of 0 dB has an absolute RMS of 1.
    In the case of white noise, this leads to a signal with unit variance, with
    values likely to be outside the [-1, 1] range.

    Parameters
    ----------
    signal: array_like
        Input signal.
    rms_dB: float or int
        Desired RMS in dB.

    Returns
    -------
    signal_scaled: array_like
        Scaled signal.
    gain: float
        Gain used to scale the signal.
    """
    rms_max = rms(signal).max()
    gain = 10**(rms_dB/20)/rms_max
    signal_scaled = gain*signal
    return signal_scaled, gain


class Mixture:
    """Main mixture class.

    A convenience class for creating a mixture and accessing its different
    components. The different components are:
        - early_speech: early reflections of the target speech
        - late_speech: late reflections of the target speech
        - speech: early_speech + late_speech
        - dir_noise: sum of reverberant noises (optional)
        - diffuse: diffuse noise (optional)
        - noise: dir_noise + diffuse
        - foreground: same as early_speech
        - background: late_speech + noise
        - mixture: speech + noise
    """

    def __init__(self):
        self.early_speech = None
        self.late_speech = None
        self.dir_noise = None
        self.diffuse = None
        self.speech_idx = None

    @property
    def mixture(self):
        return self.speech + self.noise

    @property
    def speech(self):
        return self.early_speech + self.late_speech

    @property
    def noise(self):
        output = np.zeros(self.shape)
        if self.dir_noise is not None:
            output += self.dir_noise
        if self.diffuse is not None:
            output += self.diffuse
        return output

    @property
    def foreground(self):
        return self.early_speech

    @property
    def background(self):
        return self.late_speech + self.noise

    @property
    def shape(self):
        return self.early_speech.shape

    def __len__(self):
        return len(self.early_speech)

    def add_speech(self, x, brir, reflection_boundary, padding, fs):
        brir_early, brir_late = split_brir(brir, reflection_boundary, fs)
        n_pad = round(padding*fs)
        self.speech_idx = (n_pad, n_pad+len(x))
        x = pad(x, n_pad, where='both')
        self.early_speech = spatialize(x, brir_early)
        self.late_speech = spatialize(x, brir_late)
        self.early_speech = pad(self.early_speech, n_pad, where='both')
        self.late_speech = pad(self.late_speech, n_pad, where='both')

    def add_noises(self, xs, brirs):
        if len(xs) != len(brirs):
            raise ValueError('xs and brirs must have same number of elements')
        if not xs:
            raise ValueError('xs and brirs cannot be empty')
        self.dir_noise = np.zeros(self.shape)
        for x, brir in zip(xs, brirs):
            self.dir_noise += spatialize(x, brir)

    def add_diffuse_noise(self, brirs, color, ltas=None):
        if not brirs:
            raise ValueError('brirs cannot be empty')
        self.diffuse = np.zeros(self.shape)
        for brir in brirs:
            noise = colored_noise(color, len(self))
            self.diffuse += spatialize(noise, brir)
        if ltas is not None:
            self.diffuse = match_ltas(self.diffuse, ltas)

    def set_ndr(self, ndr):
        self.diffuse, _ = adjust_snr(
            self.dir_noise,
            self.diffuse,
            ndr,
        )

    def set_snr(self, snr):
        _, gain = adjust_snr(
            self.foreground,
            self.background,
            snr,
            slice(*self.speech_idx)
        )
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse is not None:
            self.diffuse *= gain

    def set_rms(self, rms_dB):
        _, gain = adjust_rms(self.mixture, rms_dB)
        self.early_speech *= gain
        self.late_speech *= gain
        if self.dir_noise is not None:
            self.dir_noise *= gain
        if self.diffuse is not None:
            self.diffuse *= gain

    def set_tmr(self, tmr):
        target_energy = np.sum(self.foreground.mean(axis=1)**2)
        new_masker_energy = target_energy*(1/tmr-1)
        old_masker_energy = np.sum(self.background.mean(axis=1)**2)
        gain = (new_masker_energy/old_masker_energy)**0.5
        self.scale_background(gain)

    def get_rms(self):
        rms_dB = 20*np.log10(rms(self.mixture).max())
        return rms_dB

    def transform(self, func):
        for attr_name in [
            'early_speech',
            'late_speech',
            'noise',
            'diffuse',
        ]:
            attr_val = getattr(self, attr_name)
            if attr_val is not None:
                setattr(self, attr_name, func(attr_val))

    def get_long_term_label(self, label='tmr'):
        target = self.early_speech
        if label == 'tmr':
            masker = self.late_speech + self.noise
        elif label == 'tnr':
            masker = self.noise
        elif label == 'trr':
            masker = self.late_speech
        else:
            raise ValueError(f'label must be tmr, tnr or trr, got {label}')
        slice_ = slice(*self.speech_idx)
        energy_target = np.sum(target[slice_].mean(axis=-1)**2)
        energy_masker = np.sum(masker[slice_].mean(axis=-1)**2)
        label = energy_target / (energy_target + energy_masker)
        return label

    def scale_background(self, gain):
        self.late_speech = gain*self.late_speech
        if self.dir_noise is not None:
            self.dir_noise = gain*self.dir_noise
        if self.diffuse is not None:
            self.diffuse = gain*self.diffuse


class BRIRDecay:
    """Adds an extra decaying tail to binaural room impulse responses."""

    def __init__(self, rt60, drr, delay, color, fs):
        self.rt60 = rt60
        self.drr = drr
        self.delay = delay
        self.color = color
        self.fs = fs

    def __call__(self, brir, seed=None):
        if self.rt60 == 0:
            return brir
        n = max(int(round(2*(self.rt60+self.delay)*self.fs)), len(brir))
        offset = min(np.argmax(abs(brir), axis=0))
        i_start = int(round(self.delay*self.fs)) + offset
        brir_padded = np.zeros((n, 2))
        brir_padded[:len(brir)] = brir
        t = np.arange(n-i_start).reshape(-1, 1)/self.fs
        noise = colored_noise(self.color, n-i_start, seed).reshape(-1, 1)
        decaying_tail = np.zeros((n, 2))
        decaying_tail[i_start:] = np.exp(-t/self.rt60*3*np.log(10))*noise
        decaying_tail, _ = adjust_snr(brir_padded, decaying_tail, self.drr)
        return brir_padded + decaying_tail

import math

import torch
from scipy.special import expi

from ...registry import Registry

SDERegistry = Registry('sde')


class _BaseSDE:
    def probability_flow(self, x, y, score, t):
        return self.f(x, y, t) - 0.5*self.g(t)**2*score

    def reverse_step(self, x, y, score, t, dt):
        noise = self.g(t)*(-dt)**0.5*torch.randn(x.shape, device=x.device)
        return (self.f(x, y, t) - self.g(t)**2*score)*dt + noise

    def prior(self, y):
        t = torch.tensor(1, device=y.device)
        sigma = self.s(t)*self.sigma(t)
        return y + sigma*torch.randn_like(y)

    def s(self, t):
        raise NotImplementedError

    def sigma(self, t):
        raise NotImplementedError

    def f(self, x, y, t):
        raise NotImplementedError

    def g(self, t):
        raise NotImplementedError

    def sigma_inv(self, t):
        raise NotImplementedError


class _BaseOUVESDE(_BaseSDE):
    def __init__(self, stiffness, sigma_min, sigma_max, **kwargs):
        self.stiffness = stiffness
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._sigma_p = sigma_max/sigma_min
        self._log_sigma_p = math.log(sigma_max/sigma_min)

    def s(self, t):
        return (-self.stiffness * t).exp()

    def f(self, x, y, t):
        return self.stiffness * (y - x)


@SDERegistry.register('richter-ouve')
class RichterOUVESDE(_BaseOUVESDE):
    def sigma(self, t):
        return self.sigma_min * (
            ((self._sigma_p**t / self.s(t))**2 - 1)
            / (1 + self.stiffness / self._log_sigma_p)
        )**0.5

    def g(self, t):
        return self.sigma_min * self._sigma_p**t * (2 * self._log_sigma_p)**0.5

    def sigma_inv(self, sigma):
        return 0.5 * (
            1 + (1 + self.stiffness/self._log_sigma_p)
            * (sigma / self.sigma_min)**2
        ).log() / (self.stiffness + self._log_sigma_p)


@SDERegistry.register('brever-ouve')
class BreverOUVESDE(_BaseOUVESDE):
    def sigma(self, t):
        return self.sigma_min * (self._sigma_p**(2*t) - 1)**0.5

    def g(self, t):
        return self.s(t) * self.sigma_min * self._sigma_p**t \
            * (2 * self._log_sigma_p)**0.5

    def sigma_inv(self, sigma):
        return 0.5 * ((sigma/self.sigma_min)**2 + 1).log() / self._log_sigma_p


class _BaseVPSDE(_BaseSDE):
    def s(self, t):
        return (-self.stiffness * t).exp() / (1 + self.sigma(t)**2)**0.5

    def f(self, x, y, t):
        return (self.stiffness + 0.5 * self.beta(t)) * (y - x)

    def g(self, t):
        return (-self.stiffness * t).exp() * self.beta(t)**0.5


@SDERegistry.register('brever-ouvp')
class BreverOUVPSDE(_BaseVPSDE):
    def __init__(self, stiffness, beta_min, beta_max, **kwargs):
        self.stiffness = stiffness
        self.beta_min = beta_min
        self.beta_max = beta_max
        self._beta_d = beta_max - beta_min

    def beta(self, t):
        return self.beta_min + self._beta_d * t

    def sigma(self, t):
        return ((0.5*self._beta_d*t**2 + self.beta_min*t).exp() - 1)**0.5

    def sigma_inv(self, sigma):
        return (
            (self.beta_min**2 + 2*self._beta_d*(sigma ** 2 + 1).log())**0.5
            - self.beta_min
        ) / self._beta_d


@SDERegistry.register('brever-oucosine')
class BreverOUCosineSDE(_BaseVPSDE):
    def __init__(self, stiffness, lambda_min, lambda_max, shift, beta_clamp,
                 **kwargs):
        self.stiffness = stiffness
        self.shift = shift
        self.lambda_min = lambda_min + shift
        self.lambda_max = lambda_max + shift
        self.t_min = self.lambda_inv(self.lambda_min)
        self.t_max = self.lambda_inv(self.lambda_max)
        self.t_d = self.t_min - self.t_max
        self.beta_clamp = beta_clamp

    def lambda_(self, t):
        return -2 * (math.pi * t / 2).tan().log() + self.shift

    def lambda_inv(self, lambda_):
        if isinstance(lambda_, torch.Tensor):
            return 2 / math.pi * ((-lambda_ + self.shift)/2).exp().atan()
        else:
            return 2 / math.pi * math.atan(math.exp((-lambda_ + self.shift)/2))

    def lambda_tilde(self, t):
        return self.lambda_(self.t_max + self.t_d*t)

    def lambda_tilde_inv(self, lambda_):
        return (self.lambda_inv(lambda_) - self.t_max) / self.t_d

    def beta(self, t):
        pi_t_half = math.pi * (self.t_max + self.t_d*t) / 2
        return (
            math.pi * self.t_d
            / pi_t_half.cos()**2
            * pi_t_half.tan()
            / (math.exp(self.shift) + pi_t_half.tan()**2)
        ).clamp(max=self.beta_clamp)

    def sigma(self, t):
        return (-self.lambda_tilde(t) / 2).exp()

    def sigma_inv(self, sigma):
        return self.lambda_tilde_inv(-2 * sigma.log())


class _BaseBBSDE(_BaseSDE):
    def clamp(self, t):
        return t * self.t_max

    def s(self, t):
        return 1 - self.clamp(t)

    def f(self, x, y, t):
        return (y - x) / (1 - self.clamp(t))


@SDERegistry.register('bbed')
class BBEDSDE(_BaseBBSDE):
    def __init__(self, scaling=0.1, k=10, **kwargs):
        self.scaling = scaling
        self.t_max = 0.999
        self.k = k
        self._k2 = k**2
        self._logk2 = 2*math.log(k)

    def g(self, t):
        t = self.clamp(t)
        return self.scaling * self.k**t

    def sigma(self, t):
        t = self.clamp(t)
        return self.scaling * (
                self._k2 * self._logk2 * (
                    expi((t.cpu() - 1)*self._logk2).to(t.device)
                    - expi(-self._logk2)
                )
                - self._k2**t / (t - 1) - 1
        )**0.5


@SDERegistry.register('bbcd')
class BBCD(_BaseBBSDE):
    def __init__(self, scaling=0.1, **kwargs):
        self.scaling = scaling
        self.t_max = 0.999

    def g(self, t):
        return self.scaling

    def sigma(self, t):
        t = self.clamp(t)
        return self.scaling * (t / (1 - t))**0.5

    def sigma_inv(self, sigma):
        return sigma**2 / (self.scaling**2 + sigma**2) / self.t_max


@SDERegistry.register('bbls')
class BBLS(_BaseBBSDE):
    def __init__(self, scaling=0.1, **kwargs):
        self.scaling = scaling
        self.t_max = 0.999

    def g(self, t):
        t = self.clamp(t)
        return self.scaling * (1 - t) * (2 * t)**0.5

    def sigma(self, t):
        t = self.clamp(t)
        return self.scaling * t

    def sigma_inv(self, sigma):
        return sigma / (self.scaling * self.t_max)

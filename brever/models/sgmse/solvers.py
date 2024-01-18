import torch

from ...registry import Registry

SolverRegistry = Registry('solver')


@SolverRegistry.register('edm')
class EDMSolver:
    def __init__(self, num_steps, schurn, smin, smax, snoise, **kwargs):
        self.num_steps = num_steps
        self.schurn = schurn
        self.smin = smin
        self.smax = smax
        self.snoise = snoise
        self._gamma = min(schurn/num_steps, 2**0.5 - 1)

    def __call__(self, sde, y, model):
        t = torch.linspace(1, 0, self.num_steps+1, device=y.device)
        sigma = sde.sigma(t)
        x = sde.prior(y)

        for i in range(self.num_steps):
            # stochastic step
            eps = self.snoise*torch.randn_like(x)
            gamma = self._gamma if self.smin <= sigma[i] <= self.smax else 0
            sigma_hat = sigma[i]*(1 + gamma)
            t_hat = sde.sigma_inv(sigma_hat)
            x_hat = sde.s(t_hat)/sde.s(t[i])*(x - y) + y \
                + sde.s(t_hat)*(sigma_hat**2 - sigma[i]**2)**0.5*eps

            # deterministic step
            x_tilde = (x_hat - y)/sde.s(t_hat)  # undo scaling and shifting
            score = model.score(x_tilde, y, sigma_hat, t_hat)
            d_hat = sde.probability_flow(x_hat, y, score, t_hat)
            x = x_hat + (t[i+1] - t_hat)*d_hat
            if i < self.num_steps - 1:
                x_tilde = (x - y)/sde.s(t[i+1])  # undo scaling and shifting
                score = model.score(x_tilde, y, sigma[i+1], t[i+1])
                d_next = sde.probability_flow(x, y, score, t[i+1])
                x = x_hat + 0.5*(t[i+1] - t_hat)*(d_hat + d_next)

        nfe = 2*self.num_steps
        return x, nfe


@SolverRegistry.register('pc')
class PCSolver:
    def __init__(self, num_steps, corrector_steps, corrector_snr, **kwargs):
        self.num_steps = num_steps
        self.corrector_steps = corrector_steps
        self.corrector_snr = corrector_snr

    def __call__(self, sde, y, model):
        dt = -1/self.num_steps
        t = torch.arange(1, 0, dt, device=y.device)
        sigma = sde.sigma(t)
        x = sde.prior(y)
        eps = 2*(self.corrector_snr*sde.s(t)*sigma)**2

        for i in range(self.num_steps):
            # corrector step
            for _ in range(self.corrector_steps):
                x_tilde = (x - y)/sde.s(t[i])  # undo scaling and shifting
                score = model.score(x_tilde, y, sigma[i], t[i])
                x += eps[i]*score + (2*eps[i])**0.5*torch.randn_like(x)

            # predictor step
            x_tilde = (x - y)/sde.s(t[i])  # undo scaling and shifting
            score = model.score(x_tilde, y, sigma[i], t[i])
            if i < self.num_steps - 1:
                x += sde.reverse_step(x, y, score, t[i], dt)
            else:  # don't add noise on the last step
                x += dt*sde.probability_flow(x, y, score, t[i])

        nfe = self.num_steps * (self.corrector_steps + 1)
        return x, nfe

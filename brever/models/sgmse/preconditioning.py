import torch
import torch.nn as nn


class Preconditioning(nn.Module):
    def __init__(self, raw_net, sde, cskip, cout, cin, cshift, cnoise, weight,
                 sigma_data):
        super().__init__()
        self.net = raw_net
        self.sde = sde

        _preconditionings = {
            'richter': dict(
                cskip=lambda sigma: 1,
                cout=lambda sigma, scaling, t: - scaling * sigma**2 / t,
                cin=lambda sigma, scaling: scaling,
                cshift=lambda y, cin, scaling: y,
                cnoise=lambda sigma, t: t.log(),
                weight=lambda sigma: 1 / sigma**2,
            ),
            'edm': dict(
                cskip=lambda sigma: sigma_data**2 / (sigma**2 + sigma_data**2),
                cout=lambda sigma, scaling, t: sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5,  # noqa: E501
                cin=lambda sigma, scaling: 1 / (sigma**2 + sigma_data**2)**0.5,
                cshift=lambda y, cin, scaling: 0,
                cnoise=lambda sigma, t: sigma.log()/4,
                weight=lambda sigma: (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2,  # noqa: E501
            ),
            'edm-scaled-shift': dict(
                cshift=lambda y, cin, scaling: cin * y / scaling,
            )
        }

        for arg in ['cskip', 'cout', 'cin', 'cshift', 'cnoise', 'weight']:
            val = eval(arg)  # richter or edm
            if val not in _preconditionings:
                raise ValueError(f'Invalid preconditioning {arg}: {val}')
            setattr(self, arg, _preconditionings[val][arg])

    def forward(self, x, y, sigma, t):
        scaling = self.sde.s(t)

        cskip = self.cskip(sigma)
        cout = self.cout(sigma, scaling, t)
        cin = self.cin(sigma, scaling)
        cshift = self.cshift(y, cin, scaling)
        cnoise = self.cnoise(sigma, t)

        x_in = cin*x + cshift

        net_in = torch.cat([x_in.real, x_in.imag, y.real, y.imag], dim=1)
        net_out = self.net(net_in, cnoise)
        net_out = torch.complex(net_out[:, 0], net_out[:, 1]).unsqueeze(1)

        return cskip*x + cout*net_out

    def score(self, x, y, sigma, t):
        return (self(x, y, sigma, t) - x) / (self.sde.s(t) * sigma**2)

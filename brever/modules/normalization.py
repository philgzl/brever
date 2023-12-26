import torch
import torch.nn as nn


class CausalGroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups, time_dim=-1, eps=1e-10):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self._check_time_dim(time_dim)

        self.num_groups = num_groups
        self.time_dim = time_dim
        self.eps = eps

        self.gain = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        time_dim = list(range(x.ndim))[self.time_dim]
        self._check_time_dim(time_dim)

        orig_shape = x.shape
        new_shape = [
            x.shape[0],
            self.num_groups,
            x.shape[1]//self.num_groups,
            *[x.shape[i] for i in range(2, x.ndim)],
        ]
        x = x.reshape(new_shape)
        time_dim += 1

        sum_dims = [i for i in range(x.ndim) if i not in [0, 1, time_dim]]
        count = torch.ones(x.shape, device=x.device)
        count = count.sum(sum_dims, keepdims=True).cumsum(time_dim)
        mean = x.sum(sum_dims, keepdims=True).cumsum(time_dim)
        mean = mean/count
        var = x.pow(2).sum(sum_dims, keepdims=True).cumsum(time_dim)
        var = var/count - mean.pow(2)
        x = (x - mean)/(var + self.eps).sqrt()

        x = x.reshape(orig_shape)

        param_shape = [1 if i != 1 else x.shape[1] for i in range(x.ndim)]
        return x*self.gain.view(*param_shape) + self.bias.view(*param_shape)

    def _check_time_dim(self, time_dim):
        if time_dim == 0:
            raise ValueError('time_dim cannot be the batch dimension (0)')
        elif time_dim == 1:
            raise ValueError('time_dim cannot be the channel dimension (1)')


class CausalLayerNorm(CausalGroupNorm):
    def __init__(self, num_channels, time_dim=-1, eps=1e-10):
        super().__init__(
            num_channels=num_channels,
            num_groups=1,
            time_dim=time_dim,
            eps=eps,
        )


class CausalInstanceNorm(CausalGroupNorm):
    def __init__(self, num_channels, time_dim=-1, eps=1e-10):
        super().__init__(
            num_channels=num_channels,
            num_groups=num_channels,
            time_dim=time_dim,
            eps=eps,
        )

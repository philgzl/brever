# MIT License
#
# Copyright (c) 2019 Ivan Nazarov
# https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            # weight contains W_rr, W_ri=Wir, W_ii
            self.weight = torch.nn.Parameter(torch.empty(3, num_features))
            self.bias = torch.nn.Parameter(torch.empty(2, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.empty(2, num_features))
            self.register_buffer('running_var',
                                 torch.empty(2, 2, num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.copy_(torch.eye(2, 2).unsqueeze(-1))
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.copy_(torch.tensor([[1], [0], [1]]))
            torch.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.ndim != 4:
            raise ValueError(f'input must be 4 dimensional, got {input.ndim}')

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return complex_batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            exponential_average_factor,
            self.eps,
        )


def complex_batch_norm(input, running_mean, running_var, weight=None,
                       bias=None, training=True, momentum=0.1, eps=1e-05):
    assert ((running_mean is None and running_var is None)
            or (running_mean is not None and running_var is not None))
    assert ((weight is None and bias is None)
            or (weight is not None and bias is not None))

    real, imag = torch.chunk(input, 2, dim=1)
    x = torch.stack([real, imag], dim=0)

    z = whiten2x2(x, training=training, running_mean=running_mean,
                  running_cov=running_var, momentum=momentum, eps=eps)

    if weight is not None and bias is not None:
        shape = 1, x.shape[2], *([1] * (x.ndim - 3))
        weight = weight.reshape(3, *shape)
        z = torch.stack([
            z[0] * weight[0] + z[1] * weight[1],
            z[0] * weight[1] + z[1] * weight[2],
        ], dim=0) + bias.reshape(2, *shape)

    return torch.cat([z[0], z[1]], dim=1)


def whiten2x2(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, eps=1e-5):
    # input must have shape (2, batch_size, channels, ...)
    assert tensor.ndim >= 3

    # the shape to reshape statistics to for broadcasting with real and
    # imaginary parts separately: (1, channels, 1, ..., 1)
    tail = 1, tensor.shape[2], *([1] * (tensor.ndim - 3))

    # the axes along which to average, i.e. all but 0 and 2
    axes = 1, *range(3, tensor.ndim)

    # compute batch mean with shape (2, channels) and center the batch
    if training or running_mean is None:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)
    else:
        mean = running_mean
    tensor = tensor - mean.reshape(2, *tail)
    # compute covariance matrix with shape (2, 2, channels)
    if training or running_cov is None:
        var = (tensor * tensor).mean(dim=axes) + eps
        cov_uu, cov_vv = var[0], var[1]
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack([
                cov_uu.data, cov_uv.data,
                cov_vu.data, cov_vv.data,
            ], dim=0).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)
    else:
        cov_uu, cov_uv, cov_vu, cov_vv = running_cov.reshape(4, -1)

    # compute inverse square root R = [[p, q], [r, s]] of covariance matrix
    #
    # using Cholesky decomposition L.L^T=V seems to 'favour' the first
    # dimension i.e. the real part, so Trabelsi et al. (2018) used an explicit
    # inverse square root calculation as follows:
    #
    # for M = [[a, b], [c, d]] we have
    #     \sqrt{M} = \frac{1}{t} [[a+s, b], [c, d+s]]
    # where
    #     s = \sqrt{\det{M}} and t = \sqrt{\trace{M} + 2*s}
    # moreover it can be easily shown that
    #     \det{\sqrt{M}} = s
    # therefore using the formula of the inverse of a 2-by-2 matrix we have
    #     \inv{\sqrt{M}} = \frac{1}{ts} [[d+s, -b], [-c, a+s]]
    s = torch.sqrt(cov_uu*cov_vv - cov_uv*cov_vu)
    t = torch.sqrt(cov_uu + cov_vv + 2*s)
    denom = t*s
    p, q = (cov_vv+s)/denom, -cov_uv/denom
    r, s = -cov_vu/denom, (cov_uu+s)/denom

    # apply R = [[p, q], [r, s]] to input
    out = torch.stack([
        tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
        tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
    ], dim=0)
    return out

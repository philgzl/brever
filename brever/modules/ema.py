# Traditional EMA:
# Copyright (c) 2019 Samuel G. Fadel <samuelfadel@gmail.com>
# https://github.com/fadel/pytorch_ema

# Post-hoc EMA and modifications to traditional EMA:
# Copyright (c) 2024 Philippe Gonzalez <hello@philgzl.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os

import numpy as np
import torch


class _BaseEMA:
    @property
    def params(self):
        return self.model.parameters()

    def update(self, ema_params, beta):
        with torch.no_grad():
            for param, ema_param in zip(self.params, ema_params):
                ema_param += (1 - beta) * (param - ema_param)

    def store(self):
        self.stored_params = [p.clone() for p in self.params]

    def restore(self):
        if self.stored_params is None:
            raise RuntimeError('no stored parameters')
        for param, stored_param in zip(self.params, self.stored_params):
            param.data.copy_(stored_param.data)
        self.stored_params = None

    def apply(self, ema_params):
        for param, ema_param in zip(self.params, ema_params):
            param.data.copy_(ema_param.data)

    def state_dict(self):
        return {attr: getattr(self, attr) for attr in self._state_dict_attrs}

    def load_state_dict(self, state_dict):
        assert set(state_dict) == set(self._state_dict_attrs)
        for attr, value in state_dict.items():
            setattr(self, attr, value)


class EMA(_BaseEMA):
    """Traditional exponential moving average (EMA)."""

    def __init__(self, model, beta=0.999):
        assert 0.0 < beta < 1.0
        self.model = model
        self.beta = beta
        self.ema_params = [p.clone().detach() for p in model.parameters()]
        self.stored_params = None
        self._state_dict_attrs = ['ema_params']

    def update(self):
        super().update(self.ema_params, self.beta)

    def apply(self):
        super().apply(self.ema_params)


class EMAKarras(_BaseEMA):
    """Post-hoc exponential moving average (EMA) as proposed in [1]_.

    References
    ----------
    .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S.
           Laine, "Analyzing and Improving the Training Dynamics of
           Diffusion Models", in CoRR, abs/2312.02696, 2023.
    """

    def __init__(self, model, sigma_rels=[0.05, 0.1]):
        assert all(0.0 < sigma_rel < 1.0 for sigma_rel in sigma_rels)
        self.model = model
        self.sigma_rels = sigma_rels
        self.ema_params = {
            sigma_rel: [p.clone().detach() for p in model.parameters()]
            for sigma_rel in sigma_rels
        }
        self.stored_params = None
        self._num_updates = 0
        self._gammas = {
            sigma_rel: self.sigma_rel_to_gamma(sigma_rel)
            for sigma_rel in sigma_rels
        }
        self._state_dict_attrs = ['ema_params', '_num_updates', '_gammas']

    def update(self):
        self._num_updates += 1
        for sigma_rel in self.sigma_rels:
            ema_params = self.ema_params[sigma_rel]
            gamma = self._gammas[sigma_rel]
            beta = (1 - 1 / self._num_updates) ** (gamma + 1)
            super().update(ema_params, beta)

    def apply(self, sigma_rel):
        ema_params = self.ema_params[sigma_rel]
        super().apply(ema_params)

    @staticmethod
    def sigma_rel_to_gamma(sigma_rel):
        """Algorithm 2 in [1_].

        References
        ----------
        .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S.
               Laine, "Analyzing and Improving the Training Dynamics of
               Diffusion Models", in CoRR, abs/2312.02696, 2023.
        """
        t = sigma_rel ** -2
        gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
        return gamma

    @staticmethod
    def solve_weights(t_i, gamma_i, t_r, gamma_r):
        """Algorithm 3 in [1_].

        References
        ----------
        .. [1] T. Karras, M. Aittala, J. Lehtinen, J. Hellsten, T. Aila and S.
               Laine, "Analyzing and Improving the Training Dynamics of
               Diffusion Models", in CoRR, abs/2312.02696, 2023.
        """
        def p_dot_p(t_a, gamma_a, t_b, gamma_b):
            t_ratio = t_a / t_b
            t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
            t_max = np.maximum(t_a, t_b)
            num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
            den = (gamma_a + gamma_b + 1) * t_max
            return num / den

        rv = lambda x: np.float64(x).reshape(-1, 1)  # noqa: E731
        cv = lambda x: np.float64(x).reshape(1, -1)  # noqa: E731
        A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
        B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
        X = np.linalg.solve(A, B)
        return X

    def post_hoc_ema(self, ckpts_or_ckpt_dir, sigma_rel_r, t_r=None,
                     extension='.ckpt', state_dict_key=None, apply=True):
        """Apply arbitrary EMA profiles to model parameters from checkpoints.

        Parameters
        ----------
        ckpts_or_ckpt_dir : str or list of str
            Path to the checkpoint directory or list of checkpoint paths.
        sigma_rel_r : float or list of float
            Target `sigma_rel` value for each profile.
        t_r : int or list of int, optional
            Target update step for each profile. If not provided, the latest
            update step will be used.
        extension : str, optional
            Checkpoint file extension.
        state_dict_key : str, optional
            Key to access the EMA state dict within each checkpoint file. If
            not provided, the entire state dict is assumed to be the EMA state.
        apply : bool, optional
            Whether to apply the EMA profile to the attached model. If `True`,
            then `sigma_rel_r` must be a single value, since the EMA profile to
            apply to the model would be ambiguous otherwise.

        Returns
        -------
        list of list of torch.Tensor or list of torch.Tensor
            Averaged parameters for each profile. Same length as `sigma_rel_r`
            if `sigma_rel_r` is a list, otherwise a single list of averaged
            parameters.
        """
        if isinstance(ckpts_or_ckpt_dir, str):
            ckpts = [
                os.path.join(ckpts_or_ckpt_dir, f)
                for f in os.listdir(ckpts_or_ckpt_dir)
                if f.endswith(extension)
            ]
            if not ckpts:
                raise ValueError(f'no {extension} file in {ckpts_or_ckpt_dir}')
        else:
            ckpts = ckpts_or_ckpt_dir

        sigma_rel_r_was_list = isinstance(sigma_rel_r, list)
        t_r_was_list = isinstance(t_r, list)

        if not sigma_rel_r_was_list:
            if t_r_was_list:
                sigma_rel_r = [sigma_rel_r] * len(t_r)
            else:
                sigma_rel_r = [sigma_rel_r]
        if not all(isinstance(s, float) for s in sigma_rel_r):
            raise TypeError('sigma_rel_r must be a float or a list of floats')
        if not all(0.0 < s < 1.0 for s in sigma_rel_r):
            raise ValueError('sigma_rel_r values must be strictly in [0, 1]')

        if t_r is not None:
            if not t_r_was_list:
                if sigma_rel_r_was_list:
                    t_r = [t_r] * len(sigma_rel_r)
                else:
                    t_r = [t_r]
            if not all(isinstance(t, int) for t in t_r):
                raise TypeError('t_r must be an int or a list of ints')
            if len(t_r) != len(sigma_rel_r):
                raise ValueError('gamma_r and t_r must have the same length')

        if apply and len(sigma_rel_r) > 1:
            raise ValueError('cannot apply multiple EMA profiles to the model')

        ema_params = []
        t_i = []
        gamma_i = []

        for ckpt in ckpts:

            state_dict = torch.load(ckpt)

            if state_dict_key is not None:
                if state_dict_key in state_dict:
                    state_dict = state_dict[state_dict_key]
                else:
                    raise ValueError(f"no '{state_dict_key}' key in {ckpt}")

            for sigma_rel in self.sigma_rels:

                if sigma_rel not in state_dict['ema_params']:
                    raise ValueError('no averaged parameters for '
                                     f'sigma_rel={sigma_rel} in {ckpt}')

                ema = state_dict['ema_params'][sigma_rel]
                ema_params.append(ema)
                t_i.append(state_dict['_num_updates'])
                gamma_i.append(state_dict['_gammas'][sigma_rel])

        if t_r is None:
            t_r = [max(t_i)] * len(sigma_rel_r)

        gamma_r = [self.sigma_rel_to_gamma(s) for s in sigma_rel_r]
        X = self.solve_weights(t_i, gamma_i, t_r, gamma_r)

        with torch.no_grad():
            ema_params = [
                [
                    sum(x.item() * param for x, param in zip(X[:, i], params))
                    for params in zip(*ema_params)
                ]
                for i in range(X.shape[1])
            ]

        if apply:
            assert len(ema_params) == 1
            super().apply(ema_params[0])

        if not sigma_rel_r_was_list and not t_r_was_list:
            ema_params = ema_params[0]

        return ema_params

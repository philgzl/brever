import itertools
import random
import tempfile

import pytest
import torch
import torch.nn.functional as F

from brever.modules import STFT, ConvSTFT

from brever.modules import (  # isort: skip
    # CausalGroupNorm,  # TODO: test causal group norm
    CausalLayerNorm,
    CausalInstanceNorm,
    Resample,
    Upsample,
    Downsample,
    EMA,
    EMAKarras,
)


class TestResample():
    kernel = (1, 1)
    x_even = torch.randn(1, 1, 16, 16)
    x_odd = torch.randn(1, 1, 17, 17)
    x_odd_output_shape = {
        'up': (1, 1, 34, 34),
        'down': (1, 1, 9, 9),
    }

    def test_resample(self):
        resample = Resample(self.kernel)

        resample_func = lambda x: resample(x, 'up')  # noqa: E731
        self._test_resample(resample_func, 'up')
        self._test_shape(resample_func, 'up')

        resample_func = lambda x: resample(x, 'down')  # noqa: E731
        self._test_resample(resample_func, 'down')
        self._test_shape(resample_func, 'down')

        # test the buffer_padding option for seamless down-up sampling
        resample = Resample(self.kernel, buffer_padding=True)
        skips = []
        for i in range(10):
            H, W = random.randint(2, 100), random.randint(2, 100)
            x = torch.randn(1, 1, H, W)
            y = resample(x, 'down')
            skips.append((x, y))
        for i in range(10):
            x, y = skips.pop()
            z = resample(y, 'up')
            assert z.shape == x.shape

    def test_upsample(self):
        upsample = Upsample(self.kernel)
        self._test_resample(upsample, 'up')
        self._test_shape(upsample, 'up')

    def test_downsample(self):
        downsample = Downsample(self.kernel)
        self._test_resample(downsample, 'down')
        self._test_shape(downsample, 'down')

    def _test_resample(self, resample_func, which):
        # check that the implementation of FIR up/down-sampling from
        # huggingface/diffusers is the same as ours for even input shapes and
        # even kernel lengths
        y = resample_func(self.x_even)
        z = self.resample_2d(self.x_even, which, self.kernel)
        assert torch.isclose(y, z).all()

    def _test_shape(self, resample_func, which):
        # check output shape when using odd shaped input
        y = resample_func(self.x_odd)
        assert y.shape == self.x_odd_output_shape[which]

    def resample_2d(self, x, which, kernel=(1, 3, 3, 1), factor=2, gain=1):
        # set up arguments for call to upfirdn2d
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        pad = kernel.shape[0] - factor

        if which == 'up':
            kernel *= gain * (factor ** 2)
            pad = ((pad + 1) // 2 + factor - 1, pad // 2)
            up, down = factor, 1
        elif which == 'down':
            kernel *= gain
            pad = ((pad + 1) // 2, pad // 2)
            up, down = 1, factor
        else:
            raise ValueError(f'which must be up or down, got {which}')

        return self.upfirdn2d(
            x,
            kernel.to(device=x.device),
            up=up,
            down=down,
            pad=pad,
        )

    def upfirdn2d(self, tensor, kernel, up=1, down=1, pad=(0, 0)):
        # This function and only this function was copied from:
        #
        #     https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py
        #
        # which is under the following license:
        #
        # Copyright 2023 The HuggingFace Team. All rights reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #      http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
        # implied. See the License for the specific language governing
        # permissions and limitations under the License.
        #
        # The lines of code in this file outside of this function are not
        # affected by this license.
        up_x = up_y = up
        down_x = down_y = down
        pad_x0 = pad_y0 = pad[0]
        pad_x1 = pad_y1 = pad[1]

        _, channel, in_h, in_w = tensor.shape
        tensor = tensor.reshape(-1, in_h, in_w, 1)

        _, in_h, in_w, minor = tensor.shape
        kernel_h, kernel_w = kernel.shape

        out = tensor.view(-1, in_h, 1, in_w, 1, minor)
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
        out = out.view(-1, in_h * up_y, in_w * up_x, minor)

        out = F.pad(
            out,
            [
                0, 0,
                max(pad_x0, 0),
                max(pad_x1, 0),
                max(pad_y0, 0),
                max(pad_y1, 0),
            ]
        )
        out = out[
            :,
            max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
            :,
        ]

        out = out.permute(0, 3, 1, 2)
        out = out.reshape([
            -1,
            1,
            in_h * up_y + pad_y0 + pad_y1,
            in_w * up_x + pad_x0 + pad_x1,
        ])
        w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
        out = F.conv2d(out, w)
        out = out.reshape(
            -1,
            minor,
            in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
            in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
        )
        out = out.permute(0, 2, 3, 1)
        out = out[:, ::down_y, ::down_x, :]

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

        return out.view(-1, channel, out_h, out_w)


class TestNormalization():
    batch_size, channels, freqs, frames = 16, 3, 64, 100
    shape = batch_size, channels, freqs, frames
    x = torch.randn(shape)

    def test_causal_layer_norm(self):
        norm = CausalLayerNorm(self.channels)
        norm.eval()
        self._test_normalization(norm, aggregated_dims=(1, 2, 3))
        self._test_causality(norm)

    def test_causal_instance_norm(self):
        norm = CausalInstanceNorm(self.channels)
        norm.eval()
        self._test_normalization(norm, aggregated_dims=(2, 3))
        self._test_causality(norm)

    def _test_normalization(self, norm, aggregated_dims):
        y = norm(self.x)
        for i in range(1, self.frames):
            mean = self.x[..., :i+1].mean(aggregated_dims, keepdims=True)
            std = self.x[..., :i+1].std(aggregated_dims, keepdims=True,
                                        unbiased=False)
            assert torch.isclose(y[..., [i]], (self.x[..., [i]] - mean)/std,
                                 atol=1e-7).all()

    def _test_causality(self, norm):
        for i in range(1, self.frames):
            x = torch.randn(*self.shape)
            x[..., i] = float('nan')
            y = norm(x)
            assert not y[..., :i].isnan().any()


class TestEMA:
    def init_model(self):
        return torch.nn.Linear(4, 4, bias=False)

    @pytest.mark.parametrize(
        'ema_cls, ema_kwargs',
        [
            [EMA, {'beta': 0.99}],
            [EMAKarras, {'sigma_rels': [0.05, 0.1]}],
        ],
    )
    def test_state_dict(self, ema_cls, ema_kwargs):
        first_val = 1.0
        second_val = 0.0
        third_val = 0.5

        # init the model and simulate an update
        model = self.init_model()
        with torch.no_grad():
            model.weight.fill_(first_val)
        ema = ema_cls(model, **ema_kwargs)
        with torch.no_grad():
            model.weight.fill_(second_val)
        ema.update()

        # simulate saving and loading the state dict
        with tempfile.TemporaryFile() as f:
            torch.save(ema.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)

        # create a new model and load the state dict
        model = self.init_model()
        ema = ema_cls(model, **ema_kwargs)
        ema.load_state_dict(state_dict)

        # check that the loaded state dict is correct
        if isinstance(ema, EMA):
            beta = ema_kwargs['beta']
            ema_param = ema.ema_params[0]
            target = beta * first_val + (1 - beta) * second_val
            assert torch.allclose(ema_param, torch.tensor(target))
        else:
            target = second_val
            for sigma_rel in ema_kwargs['sigma_rels']:
                ema_param = ema.ema_params[sigma_rel][0]
                assert torch.allclose(ema_param, torch.tensor(second_val))

        # try another update
        with torch.no_grad():
            model.weight.fill_(third_val)
        ema.update()
        if isinstance(ema, EMA):
            target = beta * target + (1 - beta) * third_val
            assert torch.allclose(ema_param, torch.tensor(target))
        else:
            # TODO: check the actual target value for Karras EMA
            for sigma_rel in ema_kwargs['sigma_rels']:
                ema_param = ema.ema_params[sigma_rel][0]
                assert not torch.allclose(ema_param, torch.tensor(second_val))

    def test_post_hoc_ema(self):
        model = self.init_model()
        ema = EMAKarras(model, sigma_rels=[0.05, 0.1])
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as d:
            for i in range(10):
                x = torch.randn(16, 4)
                y = model(x)
                loss = y.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                ema.update()
                torch.save(ema.state_dict(), f'{d}/{i}.ckpt')

            sigma_rel_r = 0.2
            ema.post_hoc_ema(d, sigma_rel_r)


@pytest.mark.parametrize(
    'stft_kwargs',
    [
        dict(zip([
            'frame_length',
            'hop_length',
            'compression_factor',
            'scale_factor',
            'normalized',
            'onesided',
        ], vals)) for vals in itertools.product(
            [512],
            [256, 128],
            [1.0, 0.5],
            [1.0, 0.15],
            [False, True],
            [False, True],
        )
    ]
)
def test_stft(stft_kwargs):
    stft = STFT(**stft_kwargs)
    torch_generator = torch.Generator().manual_seed(42)
    x = torch.randn(4096, generator=torch_generator)
    y = stft.backward(stft(x))
    assert torch.allclose(x, y, rtol=0, atol=1e-6)
    assert torch.allclose(x, y, rtol=2e-3, atol=0)


@pytest.mark.parametrize(
    'stft_kwargs',
    [
        dict(zip([
            'frame_length',
            'hop_length',
            'compression_factor',
            'scale_factor',
            'normalized',
        ], vals)) for vals in itertools.product(
            [512],
            [256, 128],
            [1.0, 0.5],
            [1.0, 0.15],
            [False, True],
        )
    ]
)
def test_conv_stft(stft_kwargs):
    stft = ConvSTFT(**stft_kwargs)
    torch_generator = torch.Generator().manual_seed(42)
    x = torch.randn(4096, generator=torch_generator)
    y = stft.backward(stft(x))
    assert torch.allclose(x, y, rtol=1e-1, atol=1e-1)

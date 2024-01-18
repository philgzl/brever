import random

import torch

from brever.models import ModelRegistry, count_params


class _TestModel:
    default_model_kwargs = {}
    forward_model_kwargs = None
    enhance_model_kwargs = None
    latency_model_kwargs = None
    is_source_separation_network = False

    def init_model(self, model_kwargs):
        if model_kwargs is None:
            model_kwargs = self.default_model_kwargs
        net = ModelRegistry.get(self.model_name)(**model_kwargs)
        net.eval()
        return net

    def test_forward(self):
        net = self.init_model(self.forward_model_kwargs)
        net(*self.forward_args)

    def test_enhance(self):
        net = self.init_model(self.enhance_model_kwargs)
        input_ = torch.randn(2, 2000)
        output = net.enhance(input_)
        if self.is_source_separation_network:
            assert output.ndim == 2
        else:
            assert output.ndim == 1
        assert input_.shape[-1] == output.shape[-1]

    def test_latency(self):
        if self.latency is None:
            return
        net = self.init_model(self.latency_model_kwargs)
        latency = self.latency(net)
        n = 100
        min_length = max(1600, latency+1)
        max_length = 3200
        random.seed(0)
        for i in range(n):
            length = random.randint(min_length, max_length)
            input_ = torch.randn(2, length)
            if i == 0:
                # try special case where the first nan is exactly at j=latency
                nan_start = latency
            else:
                nan_start = random.randint(latency, input_.shape[-1]-1)
            input_[..., nan_start:] = float('nan')
            output = net.enhance(input_)
            j = next(
                k for k in range(output.shape[-1])
                if output[..., k].isnan().any()
            )
            assert j >= nan_start-latency+1

    def test_n_params(self):
        if self.n_params is None:
            return
        net = self.init_model(None)
        assert count_params(net) == self.n_params


class TestFFNN(_TestModel):
    model_name = 'ffnn'
    forward_args = [torch.randn(1, 384, 100)]
    latency = lambda self, net: net.stft.frame_length  # noqa: E731
    n_params = 1_509_440


class TestConvTasNet(_TestModel):
    model_name = 'convtasnet'
    forward_args = [torch.randn(1, 16000)]
    latency = lambda self, net: net.encoder.filter_length  # noqa: E731
    default_model_kwargs = {'output_sources': 1, 'causal': True}
    is_source_separation_network = True
    # n_params = 5_066_929  # same as naplab/Conv-TasNet with output_sources=2 and if the last residual convolution is not removed  # noqa: E501
    n_params = 4_935_217


class TestDCCRN(_TestModel):
    model_name = 'dccrn'
    forward_args = [torch.randn(1, 16000)]
    latency = lambda self, net: net.latency  # noqa: E731
    # n_params = 3_671_917  # same as https://github.com/huyanxin/DeepComplexCRN/issues/4 with use_complex_batchnorm=False  # noqa: E501
    n_params = 3_671_053  # with use_complex_batchnorm=True


class TestSGMSE(_TestModel):
    model_name = 'sgmsepm'
    forward_args = [
        torch.randn(4, 1, 256, 32, dtype=torch.cfloat),
        torch.randn(4, 1, 256, 32, dtype=torch.cfloat),
        torch.rand(4, 1, 1, 1),
        torch.rand(4, 1, 1, 1),
    ]
    latency = None  # TODO
    enhance_model_kwargs = {'solver_num_steps': 3}
    n_params = 27_756_186  # same as NCSN++M in sp-uhh/sgmse/tree/icassp_2023


class TestMetricGANOKD(_TestModel):
    model_name = 'metricganokd'
    forward_args = [torch.randn(4, 257, 4)]
    latency = None  # TODO
    # n_params = 1_914_524  # same as SpeechBrain with discriminator_conv_channels=[15, 15, 15, 15]  # noqa: E501
    n_params = 2_172_329  # same as wooseok-shin/MetricGAN-OKD


class TestMANNER(_TestModel):
    model_name = 'manner'
    forward_args = [torch.randn(1, 1, 16000)]
    latency = None  # TODO
    n_params = 21_253_921

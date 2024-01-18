import random
import tempfile

import numpy as np
import pytest
import torch
from utils import DummyDataset, DummyModel

from brever.models import ModelRegistry
from brever.training import BreverTrainer

FS = 16000
CHANNELS = 2
N_TRAIN_EXAMPLES = 16
N_VAL_EXAMPLES = 4
MIN_LENGTH = int(FS*0.5)
MAX_LENGTH = FS*4
EPOCHS = 2
BATCH_SAMPLER = 'bucket'
DYNAMIC_BATCH_SIZE = True
BATCH_SIZE = (MAX_LENGTH*2)/FS if DYNAMIC_BATCH_SIZE else 2


@pytest.mark.parametrize(
    'model, model_kwargs, sources, parameter_values',
    [
        [
            'dummy',
            dict(
                channels=CHANNELS,
                output_sources=2,
            ),
            3,
            torch.tensor([
                0.0007474505,
                0.3613473773,
                -0.5639899373,
                -0.5023828149,
                -0.2544076741,
                0.1717012972,
                -0.0008943576,
                0.5426732302,
                -0.0451891944,
                0.1692180037,
            ]),
        ],
        [
            'ffnn',
            dict(
                stft_frame_length=32,
                stft_hop_length=16,
                mel_filters=8,
                hidden_layers=[16],
            ),
            2,
            torch.tensor([
                0.0006858980,
                0.0791954845,
                -0.1170329824,
                -0.1047649980,
                -0.0573415197,
                0.0369523726,
                -0.0020908122,
                0.1159501597,
                -0.0136796404,
                0.0373229906,
            ]),
        ],
        [
            'convtasnet',
            dict(
                filters=4,
                filter_length=2,
                bottleneck_channels=1,
                hidden_channels=1,
                skip_channels=1,
                kernel_size=1,
                layers=1,
                repeats=1,
                output_sources=2,
            ),
            3,
            torch.tensor([
                -0.0230164211,
                0.3616594672,
                -0.5642961860,
                -0.5377520323,
                -0.2545245290,
                0.1720197797,
                0.0037516195,
                0.5429784060,
                -0.0448859259,
                0.1693844944,
            ]),
        ],
        [
            'dccrn',
            dict(
                stft_frame_length=2048,
                stft_hop_length=1024,
                channels=[1, 1],
                kernel_size=(1, 1),
                stride=(2, 1),
                padding=(2, 0),
                output_padding=(1, 0),
                lstm_channels=1,
                lstm_layers=1,
            ),
            2,
            torch.tensor([
                -0.0064911707,
                0.5360093117,
                -0.8240072131,
                -0.7364782691,
                0.9982165694,
                0.9982484579,
                -0.0017675193,
                -0.0017992775,
                0.2517986000,
                -0.3869318366,
            ]),
        ],
        [
            'sgmsepm',
            dict(
                stft_frame_length=512,
                stft_hop_length=256,
                net_base_channels=4,
                net_channel_mult=[1, 1, 1, 1],
                net_num_res_blocks=0,
                net_noise_channels=4,
                net_emb_channels=4,
                net_fir_kernel=[1, 1],
                net_attn_resolutions=[0],
                net_attn_bottleneck=True,
                solver_num_steps=1,
            ),
            2,
            torch.tensor([
                -0.1917882711,
                0.1353184879,
                -0.0095984694,
                0.3959043622,
                -0.0444973484,
                0.1326071620,
                -0.1499995887,
                -0.0981456935,
                -0.4782881439,
                -0.3310792744,
            ]),
        ],
        [
            'metricganokd',
            dict(
                generator_lstm_hidden_size=1,
                generator_lstm_num_layers=1,
                generator_lstm_bidirectional=False,
                generator_lstm_dropout=0.0,
                generator_fc_channels=[1],
                discriminator_conv_channels=[1, 1],
                discriminator_fc_channels=[1, 1],
                target_metrics=['stoi'],
                inference_metric='stoi',
            ),
            2,
            None,  # TODO
        ],
        [
            'manner',
            dict(),
            2,
            torch.tensor([
                -0.0035240455,
                0.1714718491,
                -0.2553732991,
                -0.2305837423,
                -0.1228022128,
                0.0827351138,
                -0.0056101158,
                0.2437470108,
                -0.0336221606,
                0.0873370022,
            ]),
        ]
    ],
    ids=[
        'dummy',
        'ffnn',
        'convtasnet',
        'dccrn',
        'sgmse',
        'metricganokd',
        'manner',
    ],
)
def test_model_training(model, model_kwargs, sources, parameter_values):
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cls = DummyModel if model == 'dummy' else ModelRegistry.get(model)
    model = model_cls(**model_kwargs)

    def init_dataset(**kwargs):
        return DummyDataset(
            n_sources=sources,
            n_channels=CHANNELS,
            min_length=MIN_LENGTH,
            max_length=MAX_LENGTH,
            **kwargs,
        )

    train_dataset = init_dataset(
        n_examples=N_TRAIN_EXAMPLES,
        transform=model.transform,
    )
    val_dataset = init_dataset(
        n_examples=N_VAL_EXAMPLES,
        transform=None,
    )

    parameter_before_training = sample_parameters(model)

    with tempfile.TemporaryDirectory() as tempdir:

        def init_trainer(**kwargs):
            return BreverTrainer(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model_dirpath=tempdir,
                batch_sampler=BATCH_SAMPLER,
                batch_size=BATCH_SIZE,
                dynamic_batch_size=DYNAMIC_BATCH_SIZE,
                fs=FS,
                ema=True,
                **kwargs,
            )

        trainer = init_trainer(
            model=model,
            epochs=EPOCHS,
            device='cpu',
            preload=True,
        )
        trainer.run()

        # check that parameters have changed after training
        parameter_after_training = sample_parameters(model)
        assert not torch.allclose(
            parameter_before_training,
            parameter_after_training,
        )

        # torch.set_printoptions(precision=10, linewidth=10)
        # print(parameter_after_training)
        # breakpoint()

        # check the parameter values after training
        if parameter_values is not None:
            assert torch.allclose(
                parameter_after_training,
                parameter_values,
            )

        # test resume from checkpoint and on cuda if cuda is available
        trainer = init_trainer(
            model=model.cuda() if torch.cuda.is_available() else model,
            epochs=EPOCHS+1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            preload=False,
        )
        trainer.run()

        # check that parameters have changed again
        parameter_after_training_again = sample_parameters(model)
        assert not torch.allclose(
            parameter_after_training,
            parameter_after_training_again,
        )


def sample_parameters(net, n=10):
    parameters = []
    numel = 0
    for next_parameters in net.parameters():
        if numel == n:
            break
        for param in next_parameters.flatten():
            parameters.append(param.item())
            numel += 1
            if numel == n:
                break
    return torch.tensor(parameters)

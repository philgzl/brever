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
                0.0011939593,
                0.3613566458,
                -0.5639830232,
                -0.5024077892,
                -0.2543744147,
                0.1717016846,
                -0.0019191481,
                0.5426709652,
                -0.0451210514,
                0.1691833586,
            ])
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
                0.0006956795,
                0.0792052671,
                -0.1170097962,
                -0.1047872975,
                -0.0571516603,
                0.0370130911,
                -0.0026739689,
                0.1157489568,
                -0.0136005739,
                0.0374020599,
            ])
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
                -0.0230649039,
                0.3616523147,
                -0.5642771125,
                -0.5375614762,
                -0.2545236349,
                0.1718905419,
                0.0037576971,
                0.5429673791,
                -0.0449336246,
                0.1693678647,
            ])
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
                -0.0068854718,
                0.5356806517,
                -0.8236098289,
                -0.7355660200,
                0.9982261658,
                0.9982483983,
                -0.0017647517,
                -0.0017971768,
                0.2517972887,
                -0.3869321942,
            ])
        ],
        [
            'sgmsepm',
            dict(
                stft_frame_length=512,
                stft_hop_length=256,
                net_base_channels=4,
                net_channel_mult=[1, 1, 1, 1],
                net_num_blocks_per_res=1,
                net_noise_channel_mult=1,
                net_emb_channel_mult=1,
                net_fir_kernel=[1, 1],
                net_attn_resolutions=[0],
                net_attn_bottleneck=False,
                solver_num_steps=1,
            ),
            2,
            torch.tensor([
                -0.1922940910,
                0.1330814660,
                -0.0099604866,
                0.3955351412,
                -0.0439126305,
                0.1317846030,
                -0.1510771811,
                -0.0984570533,
                -0.4786233008,
                -0.3303738832,
            ])
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
            torch.tensor([
                -0.0075165564,
                0.5364242792,
                -0.8230714202,
                -0.7359706163,
                -0.3851906657,
                0.2681215703,
                -0.0198521800,
                0.7928522825,
                -0.0887797624,
                0.2645742893,
            ])
        ],
        [
            'manner',
            dict(),
            2,
            torch.tensor([
                -0.0043031317,
                0.1695238799,
                -0.2573905587,
                -0.2253300399,
                -0.1264770925,
                0.0862908363,
                -0.0084038256,
                0.2472611815,
                -0.0289198831,
                0.0871178061,
            ])
        ],
        [
            'tfgridnet',
            dict(
                n_srcs=2,
                n_layers=1,
                lstm_hidden_units=1,
                attn_n_head=1,
                attn_approx_qk_dim=1,
                emb_dim=1,
            ),
            3,
            torch.tensor([
                0.0166356694,
                0.0712037086,
                -0.1547482908,
                -0.1049334109,
                -0.0812901407,
                0.0616331883,
                -0.0212811977,
                0.1498976648,
                -0.0321449488,
                0.0574254245,
            ])
        ],
    ],
    ids=[
        'dummy',
        'ffnn',
        'convtasnet',
        'dccrn',
        'sgmse',
        'metricganokd',
        'manner',
        'tfgridnet',
    ],
)
def test_model_training(model, model_kwargs, sources, parameter_values):
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_is_manner = model == 'manner'
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
        # TODO: Fix MANNER raising error when resuming from checkpoint due to
        # mismatching number of epochs EPOCHS vs. EPOCHS+1
        if model_is_manner:
            return
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

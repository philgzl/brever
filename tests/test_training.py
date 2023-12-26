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
N_TRAIN_EXAMPLES = 32
N_VAL_EXAMPLES = 8
MIN_LENGTH = int(FS*0.5)
MAX_LENGTH = FS*5
EPOCHS = 2
BATCH_SAMPLER = 'bucket'
DYNAMIC_BATCH_SIZE = True
BATCH_SIZE = (MAX_LENGTH*2)/FS if DYNAMIC_BATCH_SIZE else 2


@pytest.mark.parametrize(
    'model, model_kwargs, sources',
    [
        [
            'dummy',
            dict(
                channels=CHANNELS,
                output_sources=2,
            ),
            3,
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
        ],
        [
            'sgmse',
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
        ],
        [
            'manner',
            dict(),
            2,
        ]
    ],
    ids=[
        'dummy',
        'ffnn',
        'convtasnet',
        'dccrn',
        'sgmse',
        'manner',
    ],
)
def test_model_training(model, model_kwargs, sources):
    model_cls = DummyModel if model == 'dummy' else ModelRegistry.get(model)
    model = model_cls(**model_kwargs)

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

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

        # test resume from checkpoint and on cuda if cuda is available
        trainer = init_trainer(
            model=model.cuda() if torch.cuda.is_available() else model,
            epochs=EPOCHS+1,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            preload=False,
        )
        trainer.run()

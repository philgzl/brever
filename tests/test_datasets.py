import math
import os
import subprocess
import tarfile
import tempfile

import numpy as np
import pytest
import soundfile as sf
import yaml

from brever.batching import BatchSamplerRegistry
from brever.config import get_config
from brever.data import BreverDataLoader, BreverDataset

FS = 16000
N_MIXTURES = 100
MIXTURE_LENGTH = 100
SEGMENT_LENGTH = MIXTURE_LENGTH*(3/4)
SEGMENT_RATIO = MIXTURE_LENGTH/SEGMENT_LENGTH


@pytest.fixture(scope="session")
def dummy_dset(tmp_path_factory):
    tempdir = tmp_path_factory.mktemp("data-dummy")

    mix_dirname = 'audio'
    archive_path = os.path.join(tempdir, f'{mix_dirname}.tar')
    archive = tarfile.open(archive_path, 'w')

    for i in range(N_MIXTURES):
        for name in ['mixture', 'foreground', 'background']:
            filename = f'{i:05d}_{name}.flac'
            temp = tempfile.NamedTemporaryFile(
                prefix='brever_',
                suffix='.flac',
                delete=False,
            )
            x = np.random.randn(MIXTURE_LENGTH, 2)
            sf.write(temp, x, FS)
            temp.close()
            arcname = os.path.join(mix_dirname, filename)
            archive.add(temp.name, arcname=arcname)
            os.remove(temp.name)

    archive.close()

    return tempdir


@pytest.fixture(scope="session")
def real_dset(tmp_path_factory):
    tempdir = tmp_path_factory.mktemp("data-real")

    config = get_config('config/dataset.yaml')
    config.update_from_dict({
        'duration': 30,
        'rmm': {
            'speakers': {'libri_.*', 'vctk_.*'},
            'noises': {'dcase_.*', 'demand_.*'},
            'rooms': {'surrey_.*', 'ash_.*'},
        }
    })

    config_path = os.path.join(tempdir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    subprocess.check_call([
        'python',
        'scripts/create_dataset.py',
        tempdir,
        '-f'
    ])

    subprocess.check_call([
        'python',
        'scripts/create_dataset.py',
        tempdir,
        '-f',
        '--no_tar'
    ])

    return tempdir


@pytest.mark.parametrize(
    'segment_strat, dset_length',
    [
        ['drop', math.floor(SEGMENT_RATIO)*N_MIXTURES],
        ['pass', math.ceil(SEGMENT_RATIO)*N_MIXTURES],
        ['pad', math.ceil(SEGMENT_RATIO)*N_MIXTURES],
        ['overlap', math.ceil(SEGMENT_RATIO)*N_MIXTURES],
    ]
)
def test_segment_strat_on_dummy_dset(dummy_dset, segment_strat, dset_length):
    dataset = BreverDataset(
        dummy_dset,
        segment_strategy=segment_strat,
        segment_length=SEGMENT_LENGTH/FS,
    )
    assert len(dataset) == dset_length
    for inputs in dataset:
        break


@pytest.mark.parametrize(
    'segment_strat, dset_length',
    [
        ['drop', 31],
        ['pass', 37],
        ['pad', 37],
        ['overlap', 37],
    ]
)
@pytest.mark.parametrize('tar', [True, False])
def test_segment_strat_on_real_dset(real_dset, segment_strat, dset_length,
                                    tar):
    dataset = BreverDataset(
        real_dset,
        segment_strategy=segment_strat,
        segment_length=1.0,
        tar=tar,
    )
    assert len(dataset) == dset_length
    for inputs in dataset:
        break


@pytest.mark.parametrize(
    'segment_length, dset_length',
    [
        [math.ceil(N_MIXTURES*(1/4))/FS, 400],
        [math.ceil(N_MIXTURES*(1/3))/FS, 300],
        [math.ceil(N_MIXTURES*(1/2))/FS, 200],
        [math.ceil(N_MIXTURES*(2/3))/FS, 200],
    ]
)
def test_segment_len_on_dummy_dset(dummy_dset, segment_length, dset_length):
    dataset = BreverDataset(
        dummy_dset,
        segment_length=segment_length,
    )
    assert len(dataset) == dset_length
    for inputs in dataset:
        break


@pytest.mark.parametrize(
    'segment_length, dset_length',
    [
        [0.25, 137],
        [0.50, 71],
        [1.00, 37],
        [2.00, 21],
    ]
)
@pytest.mark.parametrize('tar', [True, False])
def test_segment_len_on_real_dset(real_dset, segment_length, dset_length, tar):
    dataset = BreverDataset(
        real_dset,
        segment_length=segment_length,
        tar=tar,
    )
    assert len(dataset) == dset_length
    for inputs in dataset:
        break


@pytest.mark.parametrize('segment_length', [4.0])
@pytest.mark.parametrize('overlap_length', [1.0])
@pytest.mark.parametrize('segment_strategy', [
    'pass',
])
@pytest.mark.parametrize('sampler', ['bucket'])
@pytest.mark.parametrize('dynamic_batch_size, batch_size', [
    [True, 16],
])
@pytest.mark.parametrize('drop_last', [False])
@pytest.mark.parametrize('shuffle', [True])
@pytest.mark.parametrize('dynamic_mixtures_per_epoch', [8])
def test_dynamic_mixing(real_dset, segment_length, overlap_length,
                        segment_strategy, sampler, batch_size, drop_last,
                        shuffle, dynamic_batch_size,
                        dynamic_mixtures_per_epoch):
    dataset = BreverDataset(
        real_dset,
        segment_length=segment_length,
        overlap_length=overlap_length,
        segment_strategy=segment_strategy,
        dynamic_mixing=True,
        dynamic_mixtures_per_epoch=dynamic_mixtures_per_epoch,
    )
    sampler = BatchSamplerRegistry.get(sampler)(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        dynamic=dynamic_batch_size,
    )
    dataloader = BreverDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
    )
    for batch in dataloader:
        pass
    dataloader.set_epoch(1)
    for batch in dataloader:
        pass

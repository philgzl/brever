import json
import math
import os
import tarfile
import tempfile

import numpy as np
import pytest
import soundfile as sf

from brever.data import BreverDataset

FS = 16000
N_MIXTURES = 100
MIXTURE_LENGTH = 100
SEGMENT_LENGTH = MIXTURE_LENGTH*(3/4)
SEGMENT_RATIO = MIXTURE_LENGTH/SEGMENT_LENGTH


@pytest.fixture(scope="session")
def dummy_dset(tmp_path_factory):
    tempdir = tmp_path_factory.mktemp("data")

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

    mix_info_path = os.path.join(tempdir, 'mixture_info.json')
    with open(mix_info_path, 'w') as f:
        json.dump(list(range(N_MIXTURES)), f)

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
        ['drop', 25],
        ['pass', 37],
        ['pad', 37],
        ['overlap', 37],
    ]
)
def test_segment_strat_on_real_dset(dummy_dset, segment_strat, dset_length):
    dataset = BreverDataset(
        'tests/test_dataset',
        segment_strategy=segment_strat,
        segment_length=1.0,
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
        [0.25, 129,],
        [0.50, 68],
        [1.00, 37],
        [2.00, 22],
    ]
)
def test_segment_len_on_real_dset(segment_length, dset_length):
    dataset = BreverDataset(
        'tests/test_dataset',
        segment_length=segment_length,
    )
    assert len(dataset) == dset_length
    for inputs in dataset:
        break

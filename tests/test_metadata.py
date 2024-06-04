import pytest

from brever.mixture.io import AudioFileLoader
from brever.mixture.metadata import MixtureMetadata


@pytest.fixture(scope="session")
def loader():
    loader_obj = AudioFileLoader()
    loader_obj.scan_material({'libri_.*'}, {'dcase_.*'}, {'surrey_.*'})
    return loader_obj


def test_get(loader):
    meta_obj = MixtureMetadata(loader)

    meta_obj.roll()
    meta_dict = meta_obj.get()
    assert isinstance(meta_dict, dict)
    assert meta_dict != {}

    meta_obj.roll()
    meta_dict_2 = meta_obj.get()
    assert isinstance(meta_dict_2, dict)
    assert meta_dict_2 != {}
    assert meta_dict != meta_dict_2


def test_seed(loader):
    meta_obj = MixtureMetadata(loader, seed=0)
    meta_dicts = []
    for _ in range(10):
        meta_obj.roll()
        meta_dicts.append(meta_obj.get())

    meta_obj_2 = MixtureMetadata(loader, seed=0)
    meta_dicts_2 = []
    for _ in range(10):
        meta_obj_2.roll()
        meta_dicts_2.append(meta_obj_2.get())

    assert meta_dicts == meta_dicts_2


def test_noises(loader):
    meta_obj = MixtureMetadata(loader, noise_num=(1, 3), seed=0)
    meta_dicts = []
    for _ in range(10):
        meta_obj.roll()
        meta_dicts.append(meta_obj.get())

    meta_obj_2 = MixtureMetadata(loader, noise_num=(1, 2), seed=0)
    meta_dicts_2 = []
    for _ in range(10):
        meta_obj_2.roll()
        meta_dicts_2.append(meta_obj_2.get())

    all_same_length = True
    for meta_dict, meta_dict_2 in zip(meta_dicts, meta_dicts_2):
        for noise, noise_2 in zip(meta_dict['noises'], meta_dict_2['noises']):
            assert noise == noise_2
        all_same_length &= (meta_dict['noises'] == meta_dict_2['noises'])
        meta_dict.pop('noises')
        meta_dict_2.pop('noises')
        assert meta_dict == meta_dict_2
    assert not all_same_length


@pytest.mark.parametrize('diffuse', [False, True])
@pytest.mark.parametrize('noise_num', [(0, 0), (1, 1)])
def test_ndr(loader, diffuse, noise_num):
    meta_obj = MixtureMetadata(
        loader,
        noise_num=noise_num,
        diffuse=diffuse,
        seed=0,
    )
    meta_obj.roll()
    assert ('ndr' in meta_obj.get()) == (diffuse and noise_num == (1, 1))


@pytest.mark.parametrize('diffuse', [False, True])
@pytest.mark.parametrize('noise_num', [(0, 0), (1, 1)])
def test_snr(loader, diffuse, noise_num):
    meta_obj = MixtureMetadata(
        loader,
        noise_num=noise_num,
        diffuse=diffuse,
        seed=0,
    )
    meta_obj.roll()
    assert ('snr' in meta_obj.get()) == (diffuse or noise_num == (1, 1))


def test_diffuse(loader):
    meta_obj = MixtureMetadata(
        loader,
        diffuse=False,
        seed=0
    )
    meta_obj.roll()
    meta_dict = meta_obj.get()
    assert 'diffuse' not in meta_dict
    assert 'ndr' not in meta_dict

    meta_obj_2 = MixtureMetadata(
        loader,
        diffuse=True,
        seed=0
    )
    meta_obj_2.roll()
    meta_dict_2 = meta_obj_2.get()
    meta_dict_2.pop('diffuse')
    meta_dict_2.pop('ndr')

    assert meta_dict == meta_dict_2


def test_uniform_tmr(loader):
    meta_obj = MixtureMetadata(
        loader,
        uniform_tmr=False,
        seed=0
    )
    meta_obj.roll()
    meta_dict = meta_obj.get()
    assert 'tmr' not in meta_dict

    meta_obj_2 = MixtureMetadata(
        loader,
        uniform_tmr=True,
        seed=0
    )
    meta_obj_2.roll()
    meta_dict_2 = meta_obj_2.get()
    meta_dict_2.pop('tmr')

    assert meta_dict == meta_dict_2

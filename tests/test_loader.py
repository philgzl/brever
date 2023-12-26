import os

import pytest

from brever.io import AudioFileLoader


@pytest.fixture(scope="session")
def audio_file_loader():
    return AudioFileLoader()


def test_errors(audio_file_loader):
    with pytest.raises(ValueError):
        audio_file_loader.get_speech_files('dcase_bus')
    with pytest.raises(ValueError):
        audio_file_loader.get_noise_files('libri_.*')
    with pytest.raises(ValueError):
        audio_file_loader.get_angles('babble')


def test_speakers(audio_file_loader):
    speakers = audio_file_loader.get_speakers('timit')
    assert len(speakers) == 630
    speakers = audio_file_loader.get_speakers('libri')
    assert len(speakers) == 251


@pytest.mark.parametrize(
    'speaker, file_count',
    [
        ['timit_.*', 6300],
        ['libri_.*', 28539],
        ['wsj0_.*', 34738],
        ['clarity_.*', 11352],
        ['vctk_.*', 44454],
    ]
)
def test_target(audio_file_loader, speaker, file_count):
    files = audio_file_loader.get_speech_files(speaker)
    assert len(files) == file_count


@pytest.mark.parametrize(
    'prefix, suffixes',
    [
        [
            'dcase',
            [
                'airport',
                'bus',
                'metro',
                'metro_station',
                'park',
                'public_square',
                'shopping_mall',
                'street_pedestrian',
                'street_traffic',
                'tram',
                '.*',
            ]
        ],
        [
            'noisex',
            [
                'babble',
                'buccaneer1',
                'buccaneer2',
                'destroyerengine',
                'destroyerops',
                'f16',
                'factory1',
                'factory2',
                'hfchannel',
                'leopard',
                'm109',
                'machinegun',
                'pink',
                'volvo',
                'white',
                '.*',
            ]
        ],
        [
            'icra',
            [
                '01',
                '02',
                '03',
                '04',
                '05',
                '06',
                '07',
                '08',
                '09',
                '.*',
            ]
        ],
        [
            'demand',
            []
        ],
        [
            'arte',
            []
        ]
    ]
)
def test_noises(audio_file_loader, prefix, suffixes):
    if suffixes:
        for suffix in suffixes:
            audio_file_loader.get_noise_files(f'{prefix}_{suffix}')
    else:
        audio_file_loader.get_noise_files(prefix)


@pytest.mark.parametrize(
    'prefix, suffix_count_pairs',
    [
        [
            'surrey',
            [
                # ['anechoic', 37],
                ['room_a', 37],
                ['room_b', 37],
                ['room_c', 37],
                ['room_d', 37],
                ['.*', 148]
            ]
        ],
        [
            'ash',
            [
                ['r01', 24],
                ['r02', 24],
                ['r03', 24],
                ['r04', 24],
                ['r05a', 24],
                ['r05b', 20],
                ['r06', 20],
                ['r07', 24],
                ['r08', 9],
                ['r09', 9],
                ['r10', 18],
                ['r11', 18],
                ['r12', 9],
                ['r13', 9],
                ['r14', 9],
                ['r15', 9],
                ['r16', 18],
                # ['r17', 10],
                ['r18', 16],
                ['r19', 7],
                # ['r20', 5],
                ['r21', 7],
                # ['r22', 11],
                ['r23', 16],
                ['r24', 16],
                ['r25', 14],
                ['r26', 16],
                # ['r27', 5],
                ['r28', 14],
                ['r29', 14],
                ['r30', 14],
                ['r31', 14],
                # ['r32', 7],
                ['r33', 14],
                ['r34', 14],
                ['r35', 14],
                ['r36', 14],
                ['r37', 14],
                ['r38', 14],
                ['r39', 14],
                ['.*', 538]
            ]
        ],
        [
            'catt',
            [
                ['00', 37],
                ['01', 37],
                ['02', 37],
                ['03', 37],
                ['04', 37],
                ['05', 37],
                ['06', 37],
                ['07', 37],
                ['08', 37],
                ['09', 37],
                ['10', 37],
                ['.*', 407]
            ]
        ],
        [
            'avil',
            [
                ['anechoic', 24],
                ['high', 24],
                ['low', 24],
                ['medium', 24],
                ['.*', 96]
            ]
        ],
        [
            'bras',
            [
                ['cr2', 45],
                ['cr3', 45],
                ['cr4', 45],
                ['rs5', 45],
                ['.*', 180]
            ]
        ]
    ]
)
def test_brirs(audio_file_loader, prefix, suffix_count_pairs):
    dirpath = audio_file_loader.get_path(prefix, raise_=False)
    if not os.path.exists(dirpath):
        pytest.skip()
    for suffix, count in suffix_count_pairs:
        room_alias = f'{prefix}_{suffix}'
        rooms = audio_file_loader.get_rooms(room_alias)
        n = 0
        for room in rooms:
            brirs, _ = audio_file_loader.load_brirs(room)
            n += len(brirs)
        assert n == count

import argparse
import json
import os
import tarfile
import tempfile
from io import BytesIO
from zipfile import ZipFile

import soundfile as sf
from tqdm import tqdm

from brever.config import get_config
from brever.io import resample

VAL_SPEAKERS = [
    'p226',
    'p287',
]


def filter_func(x, train_val_test):
    assert train_val_test in ['train', 'val', 'test']
    if not x.endswith('.wav'):
        return False
    if train_val_test == 'test':
        return True
    if any(os.path.basename(x).startswith(spk) for spk in VAL_SPEAKERS):
        return train_val_test == 'val'
    else:
        return train_val_test == 'train'


def main():
    dsets_dir = get_config('config/paths.yaml').DATASETS
    for train_val_test, suffix in zip(
        ['train', 'val', 'test'],
        ['trainset_28spk', 'trainset_28spk', 'testset'],
    ):
        output_dir = os.path.join(dsets_dir, train_val_test, 'vbdemand')
        archive_path = os.path.join(output_dir, 'audio.tar')
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        if args.force:
            archive = tarfile.open(archive_path, 'w')
        else:
            try:
                archive = tarfile.open(archive_path, 'a')
            except tarfile.ReadError:
                print('output archive is corrupted, recreating...')
                archive = tarfile.open(archive_path, 'w')

        try:
            arcnames = archive.getnames()
            wav_names = []

            with ZipFile(args.vbdemand_path, 'r') as zip_file:
                for noisy_or_clean, mixture_or_foreground, first_iteration in [
                    ('noisy', 'mixture', True),
                    ('clean', 'foreground', False),
                ]:
                    zip_name = f'{noisy_or_clean}_{suffix}_wav.zip'
                    print(f'reading {zip_name}...')
                    zip_data = BytesIO(zip_file.read(zip_name))
                    print('done.')
                    with ZipFile(zip_data) as inner_zip_file:
                        names = list(filter(
                            lambda x: filter_func(x, train_val_test),
                            inner_zip_file.namelist()
                        ))
                        for i, wav_name in tqdm(list(enumerate(names))):
                            if first_iteration:
                                wav_names.append(wav_name)
                            else:
                                assert os.path.basename(wav_name) \
                                    == os.path.basename(wav_names[i])

                            filename = f'{i:05d}_{mixture_or_foreground}.flac'
                            arcname = os.path.join('audio', filename)
                            if arcname in arcnames:
                                continue

                            x, fs = sf.read(
                                BytesIO(inner_zip_file.read(wav_name))
                            )
                            temp = tempfile.NamedTemporaryFile(
                                prefix='brever_',
                                suffix='.flac',
                                delete=False,
                            )
                            x = resample(x, fs, 16000)
                            sf.write(temp, x, 16000)
                            temp.close()
                            archive.add(temp.name, arcname=arcname)
                            os.remove(temp.name)
        finally:
            archive.close()

        metadata_file = os.path.join(output_dir, 'mixture_info.json')
        with open(metadata_file, 'w') as f:
            json.dump(wav_names, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vbdemand_path')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    main()

import os
import subprocess

import yaml

from brever.config import get_config


def main():
    config = get_config('config/dataset.yaml')
    config.update_from_dict({
        'duration': 30,
        'rmm': {
            'speakers': {'timit_.*', 'clarity_.*'},
            'noises': {'dcase_.*', 'noisex_.*'},
            'rooms': {'surrey_.*', 'ash_.*'},
        }
    })

    dataset_dir = 'tests/test_dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    config_path = os.path.join(dataset_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    subprocess.call([
        'python',
        'scripts/create_dataset.py',
        dataset_dir,
        '-f'
    ])


if __name__ == '__main__':
    main()

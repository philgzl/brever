import os

import yaml

from brever.args import DatasetArgParser
from brever.config import get_config, get_dataset_default_config


def main():
    paths = get_config('config/paths.yaml')

    config = get_dataset_default_config()
    config.update_from_args(args, parser.arg_map())
    if args.name is None:
        dataset_id = config.get_hash()
    else:
        dataset_id = args.name

    dataset_dir = os.path.join(paths.DATASETS, args.kind, dataset_id)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    config_path = os.path.join(dataset_dir, 'config.yaml')
    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(f'dataset already exists: {config_path} ')
    else:
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, sort_keys=False)
        print(f'Initialized {config_path}')


if __name__ == '__main__':
    parser = DatasetArgParser(description='initialize a dataset')
    parser.add_argument('kind', choices=['train', 'val', 'test'],
                        help='dump in train or test subdir')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('-n', '--name', help='dataset name')
    parser.add_argument('--all_databases', action='store_true',
                        help='use all databases')
    args = parser.parse_args()

    if args.all_databases:
        from brever.cross import CrossCorpusExperiment
        ccexp = CrossCorpusExperiment([])
        for key, databases in ccexp.databases.items():
            setattr(args, key, set(databases))

    # set different seeds for train, val and test by default
    # this is to prevent having datasets that are too similar
    if args.seed is None:
        args.seed = {
            'train': 0,
            'val': 1337,
            'test': 42,
        }[args.kind]
    # also set default speech, noise and room file limits
    for attr in ['speech_files', 'noise_files']:
        if getattr(args, attr) is None:
            setattr(args, attr, {
                'train': (0.0, 0.8),
                'val': (0.0, 0.8),
                'test': (0.8, 1.0),
            }[args.kind])
    if args.room_files is None:
        args.room_files = {
            'train': 'even',
            'val': 'even',
            'test': 'odd',
        }[args.kind]
    # weight by average length the train set but not the val and test set
    if args.weight_by_avg_length is None:
        args.weight_by_avg_length = {
            'train': True,
            'val': False,
            'test': False,
        }[args.kind]

    main()

import os
import subprocess

from brever.args import DatasetArgParser
from brever.config import DatasetFinder, ModelFinder, get_config
from brever.inspect import Path


def main():
    finder = DatasetFinder()
    matching_dsets, _ = finder.find_from_args(args)

    if args.unused:
        used_dsets = set()
        models, _ = ModelFinder().find()
        for model in models:
            cfg = get_config(os.path.join(model, 'config.yaml'))
            if hasattr(cfg, 'train_path'):
                used_dsets.add(Path(cfg.train_path))
            if hasattr(cfg, 'val_path'):
                used_dsets.add(Path(cfg.val_path))

    dsets = []
    for dset in matching_dsets:
        mix_info_file = os.path.join(dset, 'mixture_info.json')

        if args.created is not None:
            if not args.created and os.path.exists(mix_info_file):
                continue
            if args.created and not os.path.exists(mix_info_file):
                continue
        if args.unused and dset in used_dsets:
            continue

        dsets.append(dset)

    if not args.exec:
        for dset in dsets:
            print(dset)

    if dsets and args.exec:
        for dset in dsets:
            subprocess.call(args.exec.split(' ') + [dset])


if __name__ == '__main__':
    parser = DatasetArgParser(description='find datasets')
    parser.add_argument('kind', choices=['train', 'val', 'test'], nargs='?',
                        help='scan train or test subdir')
    parser.add_argument('--created', action='store_true', dest='created',
                        help='find created datasets', default=None)
    parser.add_argument('--uncreated', action='store_false', dest='created',
                        help='find uncreated datasets', default=None)
    parser.add_argument('--unused', action='store_true',
                        help='find datasets used by no model for training or '
                             'validation')
    parser.add_argument('--exec', help='command to run for each dataset found')
    args = parser.parse_args()
    main()

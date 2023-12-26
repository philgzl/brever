import argparse
import os

import yaml

from brever.data import BreverDataset
from brever.inspect import get_func_spec
from brever.mixture import RandomMixtureMaker
from brever.models import ModelRegistry
from brever.training import BreverTrainer


class NoAliasDumper(yaml.Dumper):
    # see https://github.com/yaml/pyyaml/issues/103
    def ignore_aliases(self, data):
        return True


def build_default_config(keys, classes):

    def _handle_required(x):
        x = x.copy()
        _parser = argparse.ArgumentParser()
        _arg = x.pop('arg')
        x.pop('required')
        _parser.add_argument(_arg, **x)
        _args = _parser.parse_args(['none'])
        return getattr(_args, _arg)

    def _subconfig(cls_):
        return {
            x['arg']: _handle_required(x) if x['required'] else x['default']
            for x in get_func_spec(cls_)
        }

    return {key: _subconfig(cls_) for key, cls_ in zip(keys, classes)}


def main(which):
    if which == 'dataset':
        config = {
            'duration': 36000,
            'sources': ['mixture', 'foreground'],
            **build_default_config(['rmm'], [RandomMixtureMaker])
        }
        path = f'config/{which}.yaml'
    else:
        config = {
            'arch': which,
            'seed': 0,
            'train_path': 'none',
            'val_path': 'none',
            **build_default_config(
                ['dataset', 'trainer', 'model'],
                [BreverDataset, BreverTrainer, ModelRegistry.get(which)]
            ),
        }
        path = f'config/models/{which}.yaml'

    if not args.update:
        if os.path.exists(path) and not args.force:
            print(f'Config file already exists: {path}')
            if input('Overwrite? y/n') != 'y':
                print('Aborting')
                return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f, Dumper=NoAliasDumper, sort_keys=False)
        print(f'Initialized {path}')
    else:
        if not os.path.exists(path):
            print(f'Config file does not exist: {path}')
            print('Aborting')
            return
        with open(path, 'r') as f:
            old_config = yaml.load(f, Loader=yaml.Loader)
        updated = recursive_update(old_config, config, path)
        if updated:
            with open(path, 'w') as f:
                yaml.dump(old_config, f, Dumper=NoAliasDumper, sort_keys=False)
            print(f'Updated {path}')


def recursive_update(d, u, path, suffix=''):
    updated = False
    for k, v in u.items():
        if k not in d:
            if input(f'Add {suffix}{k}={v} to {path}? [y/n]') == 'y':
                d[k] = v
                updated = True
        else:
            if isinstance(v, dict) and isinstance(d[k], dict):
                updated = recursive_update(d[k], v, path, f'{suffix}{k}.') \
                    or updated
            else:
                if d[k] == v or v == 'none':
                    continue
                if input(
                    f'Update {suffix}{k} from {d[k]} to {v} in {path}? [y/n]'
                ) == 'y':
                    d[k] = v
                    updated = True
    return updated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='write default config from class signatures'
    )
    parser.add_argument('which', nargs='*',
                        help='which default config to write')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite without confirming')
    parser.add_argument('-a', '--all', action='store_true',
                        help='write all default configs')
    parser.add_argument('-u', '--update', action='store_true',
                        help='detect and apply changes instead of overwriting')
    args = parser.parse_args()

    # manually implement choices since it doesn't work with nargs='*'
    choices = ['dataset', *ModelRegistry.keys()]
    for which in args.which:
        if which not in choices:
            choices = ', '.join(f"'{choice}'" for choice in choices)
            raise ValueError(
                f"invalid argument which: {which} (choose from {choices})"
            )

    if args.all:
        args.which = choices
    elif not args.which:
        raise ValueError('either --all or which must be specified')

    for which in args.which:
        main(which)

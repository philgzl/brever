import argparse
import os

import yaml

from brever.config import get_config


def recursive_update(d, u, path, updated=False):
    global yes_to_all
    for k, v in u.items():
        if k not in d:
            while True:
                if yes_to_all:
                    r = 'y'
                else:
                    r = input(
                        f'Add {k}={v} to {path}? [y/n/yes-all]'
                    )
                if r.lower() in ['y', 'yes-all']:
                    if r.lower() == 'yes-all':
                        yes_to_all = True
                    d[k] = v
                    updated = True
                    break
                elif r.lower() == 'n':
                    break
                else:
                    print('Could not interpret answer')
        else:
            if d[k] == v or v == 'none':
                continue
            if isinstance(v, dict):
                updated = recursive_update(d[k], v, path, updated)
    for k, v in d.copy().items():
        if k not in u:
            while True:
                if yes_to_all:
                    r = 'y'
                else:
                    r = input(
                        f'Remove {k}={v} from {path}? [y/n/yes-all]'
                    )
                if r.lower() in ['y', 'yes-all']:
                    if r.lower() == 'yes-all':
                        yes_to_all = True
                    del d[k]
                    updated = True
                    break
                elif r.lower() == 'n':
                    break
                else:
                    print('Could not interpret answer')
    return updated


def update_models():
    def_cfg = {}
    for arch in os.listdir('config/models'):
        arch = arch.split('.')[0]  # remove extension
        with open(f'config/models/{arch}.yaml') as f:
            def_cfg[arch] = yaml.load(f, Loader=yaml.Loader)
    models_dir = get_config('config/paths.yaml').MODELS
    for model in os.listdir(models_dir):
        model_dir = os.path.join(models_dir, model)
        cfg_path = os.path.join(model_dir, 'config.yaml')
        if not os.path.exists(cfg_path):
            print(f'no config.yaml in {model_dir}, skipping')
            continue
        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        if 'arch' not in cfg:
            print(f"no 'arch' field in {cfg_path}, skipping")
            continue
        arch = cfg['arch']
        updated = recursive_update(cfg, def_cfg[arch], cfg_path)
        if updated:
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)


def update_datasets():
    with open('config/dataset.yaml') as f:
        def_cfg = yaml.load(f, Loader=yaml.Loader)
    dsets_dir = get_config('config/paths.yaml').DATASETS
    for dirname in ['train', 'val', 'test']:
        dsets_subdir = os.path.join(dsets_dir, dirname)
        for dset in os.listdir(dsets_subdir):
            dset_dir = os.path.join(dsets_subdir, dset)
            cfg_path = os.path.join(dset_dir, 'config.yaml')
            if not os.path.exists(cfg_path):
                print(f'no config.yaml in {dset_dir}, skipping')
                continue
            with open(cfg_path) as f:
                cfg = yaml.load(f, Loader=yaml.Loader)
            updated = recursive_update(cfg, def_cfg, cfg_path)
            if updated:
                with open(cfg_path, 'w') as f:
                    yaml.dump(cfg, f, sort_keys=False)


def main():
    if args.models:
        update_models()
        return
    if args.datasets:
        update_datasets()
        return
    update_models()
    update_datasets()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='update configs')
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--datasets', action='store_true')
    args = parser.parse_args()

    yes_to_all = False
    main()

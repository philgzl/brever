import argparse
import itertools
import os

from brever.config import get_config


def main():
    yes_to_all = False

    if not args.inputs:
        models_dir = get_config('config/paths.yaml').MODELS
        dsets_dir = get_config('config/paths.yaml').DATASETS
        args.inputs = itertools.chain(
            [
                os.path.join(models_dir, model_id)
                for model_id in os.listdir(models_dir)
            ],
            *[
                [
                    os.path.join(dsets_dir, subdir, dset_id)
                    for dset_id in os.listdir(os.path.join(dsets_dir,  subdir))
                ]
                for subdir in ['train', 'val', 'test']
            ],
        )

    for input_ in args.inputs:
        input_id = os.path.basename(os.path.normpath(input_))

        config_path = os.path.join(input_, 'config.yaml')
        if os.path.exists(config_path):
            config = get_config(config_path)
        else:
            print(f'{config_path} does not exist, skipping')
            continue
        new_id = config.get_hash()

        if new_id != input_id:
            print(f'{input_} has wrong ID!')
            while True:
                if yes_to_all:
                    r = 'y'
                else:
                    r = input('Would you like to rename it? [y/n/yes-all]')
                if r.lower() in ['y', 'yes-all']:
                    if r.lower() == 'yes-all':
                        yes_to_all = True
                    new_input = os.path.join(
                        os.path.dirname(os.path.normpath(input_)), new_id
                    )
                    os.rename(input_, new_input)
                    print(f'Renamed {input_} to {new_input}')
                    break
                elif r.lower() == 'n':
                    print(f'{input_} was not renamed')
                    break
                else:
                    print('Could not interpret answer')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check sanity of model and '
                                                 'dataset directories')
    parser.add_argument('inputs', nargs='*',
                        help='input models and datasets, all models and all '
                             'datasets are checked by default')
    args = parser.parse_args()
    main()

import os

import yaml

from brever.args import ModelArgParser
from brever.config import get_config, get_model_default_config


def main():
    paths = get_config('config/paths.yaml')

    config = get_model_default_config(args.arch)
    config.update_from_args(args, parser.arg_map(args.arch))
    if args.name is not None:
        model_id = args.name
    else:
        model_id = config.get_hash()

    model_dir = os.path.join(paths.MODELS, model_id)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = os.path.join(model_dir, 'config.yaml')
    if os.path.exists(config_path) and not args.force:
        raise FileExistsError(f'model already exists: {config_path} ')
    else:
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, sort_keys=False)
        print(f'Initialized {config_path}')


if __name__ == '__main__':
    parser = ModelArgParser(description='initialize a model')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite config file if already exists')
    parser.add_argument('-n', '--name', help='model name')
    args = parser.parse_args()
    main()

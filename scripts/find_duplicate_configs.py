import os

from brever.config import get_config


def main():
    models_dir = get_config('config/paths.yaml').MODELS

    model_dirs = [
        os.path.join(models_dir, model_id)
        for model_id in os.listdir(models_dir)
    ]

    cfgs = {}

    for model_dir in model_dirs:
        model_id = os.path.basename(os.path.normpath(model_dir))

        config_path = os.path.join(model_dir, 'config.yaml')
        if os.path.exists(config_path):
            config = get_config(config_path)
        else:
            print(f'Model {model_id} has no config.yaml!')
            continue
        cfg_id = config.get_hash()

        if cfg_id not in cfgs:
            cfgs[cfg_id] = []
        cfgs[cfg_id].append(model_dir)

    for cfg_id, model_dirs in cfgs.items():
        if len(model_dirs) > 1:
            print(' '.join(model_dirs))


if __name__ == '__main__':
    main()

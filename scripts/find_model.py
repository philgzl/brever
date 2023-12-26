import os
import subprocess

from brever.args import ModelArgParser
from brever.config import ModelFinder, get_config


def main():
    finder = ModelFinder()
    matching_models, _ = finder.find_from_args(args)

    models = []
    for model in matching_models:
        loss_file = os.path.join(model, 'losses.npz')
        score_file = os.path.join(model, 'scores.hdf5')

        if args.trained is not None:
            if not args.trained and os.path.exists(loss_file):
                continue
            if args.trained and not os.path.exists(loss_file):
                continue
        if args.tested is not None:
            if not args.tested and os.path.exists(score_file):
                continue
            if args.tested and not os.path.exists(score_file):
                continue

        if args.trainable:
            cfg = get_config(os.path.join(model, 'config.yaml'))
            mix_json = 'mixture_info.json'
            if (
                not hasattr(cfg, 'train_path')
                or not hasattr(cfg, 'val_path')
                or not os.path.exists(os.path.join(cfg.train_path, mix_json))
                or not os.path.exists(os.path.join(cfg.val_path, mix_json))
            ):
                continue

        models.append(model)

    if not args.exec:
        for model in models:
            print(model)

    if models and args.exec:
        if '{}' not in args.exec:
            raise ValueError('--exec must contain a placeholder {}')
        for model in models:
            subprocess.call(args.exec.format(model), shell=True)


if __name__ == '__main__':
    parser = ModelArgParser(required=False, description='find models')
    parser.add_argument('--trained', action='store_true', dest='trained',
                        help='find trained models', default=None)
    parser.add_argument('--untrained', action='store_false', dest='trained',
                        help='find untrained models', default=None)
    parser.add_argument('--tested', action='store_true', dest='tested',
                        help='find tested models', default=None)
    parser.add_argument('--untested', action='store_false', dest='tested',
                        help='find untested models', default=None)
    parser.add_argument('--trainable', action='store_true',
                        help='find models whose training dataset is created')
    parser.add_argument('--exec', help='command to run for each model found')
    args = parser.parse_args()
    main()

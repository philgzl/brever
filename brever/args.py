import argparse
import warnings

from .data import BreverDataset
from .inspect import OriginAction, Path, get_func_spec
from .mixture import RandomMixtureMaker
from .models import ModelRegistry
from .training import BreverTrainer

ALLOWED_DUPLICATE_ARGS = ['fs']


class BaseArgParser(argparse.ArgumentParser):

    extra_args = {}

    @classmethod
    def _add_args(cls, func, parser, add_defaults=False, required=True):
        for item in get_func_spec(func):
            arg = item.pop('arg')
            if not add_defaults:
                item['default'] = None
            if not required:
                item['required'] = False
            parser.add_argument(f'--{arg}', **item)

    @classmethod
    def add_extra_args(cls, parser, new_group=True, required=False):
        if new_group:
            parser = parser.add_argument_group('extra options')
        for arg, kwargs in cls.extra_args.items():
            kwargs['required'] = kwargs.get('required', False) and required
            parser.add_argument(f'--{arg}', **kwargs)

    @classmethod
    def build_argmap(cls, prefixes, classes):
        arg_map = {}
        for prefix, cls_ in zip(prefixes, classes):
            for item in get_func_spec(cls_):
                arg = item['arg']
                if arg not in arg_map:
                    arg_map[arg] = []
                key_list = [item['arg']]
                if prefix:
                    key_list = [prefix] + key_list
                arg_map[arg].append(key_list)
        for arg, key_list_list in arg_map.items():
            if len(key_list_list) > 1 and arg not in ALLOWED_DUPLICATE_ARGS:
                warnings.warn(
                    f'Argument --{arg} matches more than one '
                    'configuration field: '
                    f'{", ".join(".".join(x) for x in key_list_list)}. '
                    'These will be set to the same value.'
                )
        return arg_map


class DatasetArgParser(BaseArgParser):

    extra_args = {
        'duration': dict(type=int),
        'sources': dict(action=OriginAction(list, str)),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_args(self)
        self.add_extra_args(self)

    @classmethod
    def add_args(cls, parser, new_group=True):
        if new_group:
            parser = parser.add_argument_group('random mixture maker options')
        cls._add_args(RandomMixtureMaker, parser)

    @classmethod
    def arg_map(cls):
        return {
            **{arg: [[arg]] for arg in cls.extra_args.keys()},
            **cls.build_argmap(['rmm'], [RandomMixtureMaker])
        }


class ModelArgParser(BaseArgParser):

    extra_args = {
        'seed': dict(type=int),
        'train_path': dict(type=Path, required=True),
        'val_path': dict(type=Path, required=True),
    }

    def __init__(self, required=True, *args, **kwargs):
        super().__init__(*args, conflict_handler='resolve', **kwargs)

        self.add_dataset_args(self, required=required)
        self.add_trainer_args(self, required=required)
        self.add_extra_args(self, required=required)

        subs = self.add_subparsers(
            help='model architecture',
            dest='arch',
            parser_class=argparse.ArgumentParser,
            required=required,
        )
        for model in ModelRegistry.keys():
            sub = subs.add_parser(model, conflict_handler='resolve')
            self.add_model_args(sub, model)

    @classmethod
    def add_model_args(cls, parser, model, new_group=True, required=False):
        if new_group:
            parser = parser.add_argument_group('model options')
        cls._add_args(ModelRegistry.get(model), parser, required=required)

    @classmethod
    def add_dataset_args(cls, parser, new_group=True, required=False):
        if new_group:
            parser = parser.add_argument_group('dataset options')
        cls._add_args(BreverDataset, parser, required=required)

    @classmethod
    def add_trainer_args(cls, parser, new_group=True, required=False):
        if new_group:
            parser = parser.add_argument_group('trainer options')
        cls._add_args(BreverTrainer, parser, required=required)

    @classmethod
    def trainer_arg_map(cls):
        return {
            **{arg: [[arg]] for arg in cls.extra_args.keys()},
            **cls.build_argmap(
                ['dataset', 'trainer'],
                [BreverDataset, BreverTrainer],
            )
        }

    @classmethod
    def arg_map(cls, model_key):
        return {
            **{arg: [[arg]] for arg in cls.extra_args.keys()},
            **cls.build_argmap(
                ['dataset', 'trainer', 'model'],
                [BreverDataset, BreverTrainer, ModelRegistry.get(model_key)],
            )
        }

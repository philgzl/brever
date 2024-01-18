import hashlib
import os
import warnings

import yaml

from .args import DatasetArgParser, ModelArgParser
from .inspect import Path, get_func_spec
from .models import ModelRegistry


def get_config(path):
    with open(path) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config = BreverConfig(config_dict)
    return config


def get_model_default_config(model_key):
    path = f'config/models/{model_key}.yaml'
    with open(path) as f:
        file_config = yaml.load(f, Loader=yaml.Loader)
    spec = get_func_spec(ModelRegistry.get(model_key))
    spec_config = {arg: x['default'] for arg, x in spec.items()}
    if file_config['model'] != spec_config:
        warnings.warn(f'Default config file {path} does not match default '
                      'arguments from model __init__ signature')
    config = BreverConfig(file_config)
    return config


class BreverConfig:
    def __init__(self, dict_):
        for key, value in dict_.items():
            if isinstance(value, dict):
                super().__setattr__(key, BreverConfig(value))
            else:
                super().__setattr__(key, value)

    def __setattr__(self, attr, value):
        class_name = self.__class__.__name__
        raise AttributeError(f'{class_name} objects are immutable')

    def to_dict(self):
        dict_ = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BreverConfig):
                dict_[key] = value.to_dict()
            else:
                dict_[key] = value
        return dict_

    def to_json(self):
        dict_ = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BreverConfig):
                dict_[key] = value.to_json()
            elif isinstance(value, set):
                dict_[key] = sorted(value)
            else:
                dict_[key] = value
        return dict_

    def get_hash(self, length=8):

        def sorted_dict(input_dict):
            output_dict = {}
            for key, value in sorted(input_dict.items()):
                if isinstance(value, dict):
                    output_dict[key] = sorted_dict(value)
                elif isinstance(value, set):
                    output_dict[key] = sorted(value)
                else:
                    output_dict[key] = value
            return output_dict

        dict_ = self.to_dict()
        dict_ = sorted_dict(dict_)
        str_ = str(dict_.items())
        hash_ = hashlib.sha256(str_.encode()).hexdigest()
        return hash_[:length]

    def get_field(self, key_list):
        attr = getattr(self, key_list[0])
        if len(key_list) == 1:
            return attr
        else:
            return attr.get_field(key_list[1:])

    def set_field(self, key_list, value):
        if len(key_list) == 1:
            key = key_list[0]
            attr = getattr(self, key)
            if not isinstance(value, type(attr)):
                type_a = attr.__class__.__name__
                type_v = value.__class__.__name__
                msg = f'attribute {key} must be {type_a}, got {type_v}'
                raise TypeError(msg)
            super().__setattr__(key, value)
        else:
            config = self.get_field(key_list[:-1])
            config.set_field(key_list[-1:], value)

    def update_from_args(self, args, arg_map):
        for arg_name, key_list_list in arg_map.items():
            value = getattr(args, arg_name)
            if value is not None:
                for key_list in key_list_list:
                    self.set_field(key_list, value)

    def update_from_dict(self, dict_, parent_keys=[]):

        def flatten_dict(dict_, parent_keys=[]):
            for key, value in dict_.items():
                key_list = parent_keys + [key]
                if isinstance(value, dict):
                    yield from flatten_dict(value, key_list)
                else:
                    yield key_list, value

        for key_list, value in flatten_dict(dict_):
            self.set_field(key_list, value)


class ModelFinder:
    def __init__(self):
        self.models = None
        self.configs = None

    def find(self, arch=None, **kwargs):
        if self.models is None:
            self.models = {}
            paths = get_config('config/paths.yaml')
            models_dir = paths.MODELS
            if os.path.exists(models_dir):
                for model in os.listdir(models_dir):
                    cfg_path = os.path.join(models_dir, model, 'config.yaml')
                    if os.path.exists(cfg_path):
                        cfg = get_config(cfg_path)
                        self.models[os.path.join(models_dir, model)] = cfg

        models = []
        configs = []
        for model, cfg in self.models.items():
            valid = True
            if arch is not None \
                    and (not hasattr(cfg, 'arch') or cfg.arch != arch):
                valid = False
            else:
                if kwargs:
                    if not hasattr(cfg, 'arch'):
                        valid = False
                    else:
                        arg_map = ModelArgParser.arg_map(cfg.arch)
                        for key, value in kwargs.items():
                            key_list_list = arg_map[key]
                            for key_list in key_list_list:
                                try:
                                    if cfg.get_field(key_list) != value:
                                        valid = False
                                        break
                                except AttributeError:
                                    valid = False
                                    break
                            if not valid:
                                break
            if valid:
                models.append(model)
                configs.append(cfg)

        return models, configs

    def find_from_args(self, args):
        if args.arch is None:
            arg_map = ModelArgParser.trainer_arg_map()
        else:
            arg_map = ModelArgParser.arg_map(args.arch)
        kwargs = {}
        for key in arg_map.keys():
            val = getattr(args, key)
            if val is not None:
                kwargs[key] = val
        return self.find(args.arch, **kwargs)


class DatasetFinder:
    def __init__(self):
        self.dsets = None
        self.configs = None

    def find(self, kind=None, **kwargs):
        if self.dsets is None:
            self.dsets = {}
            paths = get_config('config/paths.yaml')
            kinds = ['train', 'val', 'test'] if kind is None else [kind]
            for kind in kinds:
                dsets_dir = os.path.join(paths.DATASETS, kind)
                if os.path.exists(dsets_dir):
                    for dset in os.listdir(dsets_dir):
                        cfg_path = os.path.join(dsets_dir, dset, 'config.yaml')
                        if os.path.exists(cfg_path):
                            cfg = get_config(cfg_path)
                            self.dsets[os.path.join(dsets_dir, dset)] = cfg

        arg_map = DatasetArgParser.arg_map()
        dsets = []
        configs = []
        for dset, config in self.dsets.items():
            valid = True
            for key, value in kwargs.items():
                key_list_list = arg_map[key]
                for key_list in key_list_list:
                    try:
                        if config.get_field(key_list) != value:
                            valid = False
                            break
                    except AttributeError:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                dsets.append(dset)
                configs.append(config)

        return dsets, configs

    def find_from_args(self, args):
        arg_map = DatasetArgParser.arg_map()
        kwargs = {}
        for key in arg_map.keys():
            val = getattr(args, key)
            if val is not None:
                kwargs[key] = val
        return self.find(args.kind, **kwargs)


class ModelInitializer:
    def __init__(self, batch_mode=False):
        self.dir_ = get_config('config/paths.yaml').MODELS
        self.batch_mode = batch_mode
        self._default_cfg_path = lambda arch: f'config/models/{arch}.yaml'

    def init_from_args(self, args):
        config = get_config(self._default_cfg_path(args.arch))
        config.update_from_args(args, ModelArgParser.arg_map(args.arch))
        return self.write_config(config, args.force)

    def init_from_kwargs(self, arch, force=False, model_id=None, **kwargs):
        config = self.get_config_from_kwargs(arch, **kwargs)
        return self.write_config(config, force=force, model_id=model_id)

    def get_config_from_kwargs(self, arch, **kwargs):
        config = get_config(self._default_cfg_path(arch))
        arg_map = ModelArgParser.arg_map(arch)
        for key, val in kwargs.items():
            for key_list in arg_map[key]:
                config.set_field(key_list, val)
        return config

    def get_path_from_kwargs(self, arch, **kwargs):
        config = self.get_config_from_kwargs(arch, **kwargs)
        model_id = config.get_hash()
        model_dir = os.path.join(self.dir_, model_id)
        return Path(model_dir)

    def write_config(self, config, force=False, model_id=None):
        if model_id is None:
            model_id = config.get_hash()

        model_dir = os.path.join(self.dir_, model_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        config_path = os.path.join(model_dir, 'config.yaml')
        if os.path.exists(config_path) and not force:
            msg = f'model already exists: {config_path}'
            if self.batch_mode:
                print(msg)
            else:
                raise FileExistsError(msg)
        else:
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f)
            print(f'Initialized {config_path}')

        return Path(model_dir)


class DatasetInitializer:
    def __init__(self, batch_mode=False):
        self.dir_ = get_config('config/paths.yaml').DATASETS
        self.batch_mode = batch_mode
        self._default_cfg_path = 'config/dataset.yaml'

    def init_from_args(self, args):
        config = get_config(self._default_cfg_path)
        config.update_from_args(args, DatasetArgParser.arg_map())
        return self.write_config(args.kind, config, args.force)

    def init_from_kwargs(self, kind, force=False, **kwargs):
        config = self.get_config_from_kwargs(**kwargs)
        return self.write_config(kind, config, force=force)

    def get_config_from_kwargs(self, **kwargs):
        config = get_config(self._default_cfg_path)
        arg_map = DatasetArgParser.arg_map()
        for key, val in kwargs.items():
            for key_list in arg_map[key]:
                config.set_field(key_list, val)
        return config

    def get_path_from_kwargs(self, kind, **kwargs):
        config = self.get_config_from_kwargs(**kwargs)
        dataset_id = config.get_hash()
        dataset_dir = os.path.join(self.dir_, kind, dataset_id)
        return Path(dataset_dir)

    def write_config(self, kind, config, force=False):
        dataset_id = config.get_hash()

        dataset_dir = os.path.join(self.dir_, kind, dataset_id)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        config_path = os.path.join(dataset_dir, 'config.yaml')
        if os.path.exists(config_path) and not force:
            msg = f'dataset already exists: {config_path}'
            if self.batch_mode:
                print(msg)
            else:
                raise FileExistsError(msg)
        else:
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f)
            print(f'Initialized {config_path}')

        return Path(dataset_dir)

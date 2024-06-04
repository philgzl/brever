import pytest

from brever.args import DatasetArgParser, ModelArgParser
from brever.config import get_dataset_default_config, get_model_default_config
from brever.data import BreverDataset
from brever.inspect import get_func_spec
from brever.mixture import RandomMixtureMaker
from brever.models import ModelRegistry
from brever.training import BreverTrainer


def _build_command(func, parser_cls, command=[]):
    spec = get_func_spec(func)
    for arg, x in spec.items():
        arg = f'--{arg}='
        if isinstance(x['default'], (list, tuple, set)):
            arg += ','.join(str(y) for y in x['default'])
        else:
            arg += str(x['default'])
        command.append(arg)
    return command


def test_dataset_args():
    parser = DatasetArgParser()

    assert len(parser._actions) == len(parser.arg_map()) + 1

    arg_cmd = [
        '--duration=36000',
        '--sources=mixture,foreground'
    ]
    _build_command(RandomMixtureMaker, DatasetArgParser, arg_cmd)
    args = parser.parse_args(arg_cmd)

    assert all(arg is not None for arg in args.__dict__.values())

    config = get_dataset_default_config()
    config.update_from_args(args, parser.arg_map())


@pytest.mark.parametrize(
    'model', ModelRegistry.keys()
)
def test_model_args(model):
    arg_cmd = [
        '--seed=0',
        '--train_path=foo',
        '--val_path=bar',
    ]
    _build_command(BreverDataset, ModelArgParser, arg_cmd)
    _build_command(BreverTrainer, ModelArgParser, arg_cmd)
    arg_cmd.append(model)
    _build_command(ModelRegistry.get(model), ModelArgParser, arg_cmd)

    parser = ModelArgParser()
    args = parser.parse_args(arg_cmd)

    for arg in args.__dict__.keys():
        assert args.__dict__[arg] is not None, f'{arg} not set'

    config = get_model_default_config(model)
    config.update_from_args(args, parser.arg_map(model))

    model_cls = ModelRegistry.get(config.arch)
    model = model_cls(**config.model.to_dict())

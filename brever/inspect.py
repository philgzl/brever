import argparse
import inspect
from types import UnionType
from typing import Generic, TypeVar, Union, get_args, get_origin

T = TypeVar('T')


class NoParse(Generic[T]):
    pass


class Parse(Generic[T]):
    pass


class Path:
    def __new__(cls, s):
        return s.replace('\\', '/').rstrip('/')


class Bool:
    def __new__(cls, s):
        if s.lower() in ['true', 'yes', '1']:
            return True
        elif s.lower() in ['false', 'no', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError(f'expected bool value, got {s}')


def get_func_spec(func):
    def raise_bad_typing():
        raise ValueError(f'unsupported typing for argument {arg}, got {type_}')

    def raise_bad_default():
        raise ValueError(f'default value of argument {arg} does not match '
                         f'typing, got {default} and {type_}')

    def raise_ambiguous_union():
        raise ValueError(f'ambiguous union typing for argument {arg}, got '
                         f'{type_}; use Parse or NoParse to avoid ambiguity')

    spec = inspect.getfullargspec(func)

    if spec.defaults:
        defaults = dict(zip(reversed(spec.args), reversed(spec.defaults)))
    else:
        defaults = {}

    output = []
    for arg in spec.args:
        if arg in ['self', 'return']:
            continue
        if arg not in spec.annotations:
            raise ValueError(f'missing type hint for argument {arg}')

        type_ = spec.annotations[arg]
        default = defaults.get(arg)
        action = None

        origin = get_origin(type_)
        if origin is None:
            if default is not None and not isinstance(default, type_):
                raise_bad_default()
        else:
            if origin is NoParse:
                continue
            type_args = get_args(type_)
            if origin in [list, set, tuple]:
                if origin is tuple:
                    if not all(type_args[0] == t for t in type_args):
                        raise_bad_typing()
                    if default is not None and len(default) != len(type_args):
                        raise_bad_default()
                else:
                    if len(type_args) != 1:
                        raise_bad_typing()
                if default is not None:
                    if not isinstance(default, origin):
                        raise_bad_default()
                    if not all(isinstance(d, type_args[0]) for d in default):
                        raise_bad_default()
                type_ = str
                action = OriginAction(origin, type_args[0])
            elif origin in [Union, UnionType]:
                parse, not_no_parse = [], []
                for t in type_args:
                    if get_origin(t) is Parse:
                        parse.append(t)
                    elif get_origin(t) is not NoParse:
                        not_no_parse.append(t)
                if len(parse) > 1:
                    raise_ambiguous_union()
                elif len(parse) == 1:
                    parse, = parse
                    type_, = get_args(parse)
                elif len(not_no_parse) > 1:
                    raise_ambiguous_union()
                elif len(not_no_parse) == 1:
                    not_no_parse, = not_no_parse
                    type_, = get_args(not_no_parse)
                else:
                    raise_bad_typing()
                if default is not None and not isinstance(default, type_):
                    raise_bad_default()
            else:
                raise_bad_typing()

        if type_ not in [str, int, float, bool, Path]:
            raise_bad_typing()

        if type_ is bool:
            type_ = Bool

        output.append(dict(
            arg=arg,
            type=type_,
            action=action,
            default=default,
            required=arg not in defaults,
        ))

    return output


class OriginAction:
    def __init__(self, origin, type_):
        self.origin = origin
        self.type_ = type_

    def __call__(oself, *args, **kwargs):

        class _Action(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None):
                values = [oself.type_(s) for s in values.split(',') if s != '']
                setattr(namespace, self.dest, oself.origin(values))

        return _Action(*args, **kwargs)

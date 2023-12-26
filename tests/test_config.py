import pytest

from brever.config import BreverConfig


@pytest.mark.parametrize(
    'config_1, config_2, result',
    [
        [
            {'foo': 0, 'bar': 1},
            {'bar': 1, 'foo': 0},
            True
        ],
        [
            {'foo': 0},
            {'foo': 1},
            False
        ],
        [
            {'foo': [0, 1]},
            {'foo': [1, 0]},
            False
        ],
        [
            {'foo': {0, 1}},
            {'foo': {1, 0}},
            True
        ],
    ]
)
def test_hash(config_1, config_2, result):
    config_1 = BreverConfig(config_1)
    config_2 = BreverConfig(config_2)
    assert (config_1.get_hash() == config_2.get_hash()) == result


def test_attribute_assignement():
    config = BreverConfig({'foo': 0})

    with pytest.raises(AttributeError):
        config.foo = 1

    with pytest.raises(TypeError):
        config.update_from_dict({'foo': 'bar'})

    config.update_from_dict({'foo': 1})

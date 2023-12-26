import pytest

from brever.mixture import (  # isort: skip
    BaseRandGen,
    ChoiceRandGen,
    DistRandGen,
    MultiChoiceRandGen,
    AngleRandomizer,
    TargetFileRandomizer,
    NoiseFileRandomizer,
)


def test_roll_error():
    for rand in [
        BaseRandGen(),
        ChoiceRandGen(['foo', 'bar']),
        ChoiceRandGen(['foo', 'bar'], weights=[0.3, 0.7]),
        ChoiceRandGen(['foo', 'bar'], size=2, replace=True),
        ChoiceRandGen(['foo', 'bar'], size=2, replace=False),
        DistRandGen('uniform', [0.0, 1.0]),
        DistRandGen('logistic', [0.0, 4.3429448190325175]),
        MultiChoiceRandGen({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        AngleRandomizer({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        TargetFileRandomizer({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        NoiseFileRandomizer({'dcase': ['a', 'b', 'c']}, size=3),
    ]:
        if isinstance(rand, NoiseFileRandomizer):
            def get():
                rand.get('dcase', 0)
        elif isinstance(rand, (
            MultiChoiceRandGen,
            AngleRandomizer,
            TargetFileRandomizer,
        )):
            def get():
                rand.get('foo')
        else:
            get = rand.get
        with pytest.raises(ValueError):
            get()
        rand.roll()
        get()
        with pytest.raises(ValueError):
            get()
        rand.roll()
        get()


def test_base():
    # test seeding
    rand1 = BaseRandGen(seed=0)
    rand2 = BaseRandGen(seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get() == rand2.get()

    rand1 = BaseRandGen(seed=0)
    rand2 = BaseRandGen(seed=42)
    rand1.roll(), rand2.roll()
    assert rand1.get() != rand2.get()


def test_choice():
    # test error
    rand = ChoiceRandGen(['foo', 'bar'], size=3, replace=False, seed=0)
    with pytest.raises(ValueError):
        rand.roll()

    rand = ChoiceRandGen(['foo', 'bar'], size=10, weights=[0, 1], seed=0)
    rand.roll()
    x = rand.get()
    assert 'foo' not in x

    rand = ChoiceRandGen(['foo', 'bar'], size=10, weights=[1, 1], seed=0)
    rand.roll()
    x = rand.get()
    assert 'foo' in x and 'bar' in x

    rand = ChoiceRandGen(['foo', 'bar'], size=10, seed=0)
    rand.roll()
    x = rand.get()
    assert 'foo' in x and 'bar' in x

    # test seeding
    rand1 = ChoiceRandGen(list(range(10)), size=2, seed=0)
    rand2 = ChoiceRandGen(list(range(10)), size=3, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get() == rand2.get()[:2]

    rand1 = ChoiceRandGen(list(range(10)), seed=0)
    rand2 = ChoiceRandGen(list(range(10)), seed=42)
    rand1.roll(), rand2.roll()
    assert rand1.get() != rand2.get()


def test_dist():
    rand1 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    rand2 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get() == rand2.get()

    rand1 = DistRandGen('uniform', [0.0, 1.0], seed=0)
    rand2 = DistRandGen('uniform', [0.0, 1.0], seed=42)
    rand1.roll(), rand2.roll()
    assert rand1.get() != rand2.get()


def test_multi_choice():
    pool_dict = {
        'foo': list(range(10)),
        'bar': list(range(42)),
    }
    rand1 = MultiChoiceRandGen(pool_dict=pool_dict, seed=0)
    rand2 = MultiChoiceRandGen(pool_dict=pool_dict, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get('foo') == rand2.get('foo')
        rand1.roll(), rand2.roll()
        assert rand1.get('bar') == rand2.get('bar')
        rand1.roll(), rand2.roll()
        assert rand1.get(['foo', 'bar']) == rand2.get(['foo', 'bar'])

    rand1 = MultiChoiceRandGen(pool_dict=pool_dict, seed=0)
    rand2 = MultiChoiceRandGen(pool_dict=pool_dict, seed=42)
    rand1.roll(), rand2.roll()
    assert rand1.get('foo') != rand2.get('foo')
    rand1.roll(), rand2.roll()
    assert rand1.get('bar') != rand2.get('bar')
    rand1.roll(), rand2.roll()
    assert rand1.get(['foo', 'bar']) != rand2.get(['foo', 'bar'])


def test_angle():
    pool_dict = {
        'surrey': list(range(-45, 45+5, 5)),
        'ash': list(range(-90, 90+10, 10)),
    }
    target_angle_min, target_angle_max = -15, 15
    noise_angle_min, noise_angle_max = -30, 30

    # test generated angles are within the limits
    rand = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=0,
                           target_angle=[target_angle_min, target_angle_max],
                           noise_angle=[noise_angle_min, noise_angle_max])
    for i in range(10):
        rand.roll()
        target_angle, noise_angles = rand.get('surrey')
        assert target_angle_min <= target_angle <= target_angle_max
        for noise_angle in noise_angles:
            assert noise_angle_min <= noise_angle <= noise_angle_max

    # test generated angles can be outside the limits if they are not provided
    rand = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=0)

    def experiment():
        rand.roll()
        target_angle, _ = rand.get('ash')
        return not (target_angle_min <= target_angle <= target_angle_max)

    assert any(experiment() for i in range(10))

    def experiment():
        rand.roll()
        _, noise_angles = rand.get('surrey')
        for noise_angle in noise_angles:
            if not (noise_angle_min <= noise_angle <= noise_angle_max):
                return True
        return False

    assert any(experiment() for i in range(10))

    # test seeding
    rand1 = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=0)
    rand2 = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        target_angle_1, noise_angles_1 = rand1.get('ash')
        target_angle_2, noise_angles_2 = rand2.get('ash')
        assert target_angle_1 == target_angle_2
        assert noise_angles_1 == noise_angles_2

    rand1 = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=0)
    rand2 = AngleRandomizer(pool_dict=pool_dict, noise_num=3, seed=42)
    rand1.roll(), rand2.roll()
    target_angle_1, noise_angles_1 = rand1.get('ash')
    target_angle_2, noise_angles_2 = rand2.get('ash')
    assert target_angle_1 != target_angle_2
    assert noise_angles_1 != noise_angles_2


def test_target_file():
    # test limits
    pool_dict = {
        'foo': list(range(10)),
        'bar': list(range(42)),
    }
    lims = [0.0, 0.3]
    rand = TargetFileRandomizer(pool_dict=pool_dict, lims=lims, seed=0)
    for i in range(10):
        rand.roll()
        x1, x2 = rand.get(['foo', 'bar'])
        assert x1 in pool_dict['foo'][0:3]
        assert x2 in pool_dict['bar'][0:14]

    lims = [0.7, 1.0]
    rand = TargetFileRandomizer(pool_dict=pool_dict, lims=lims, seed=0)
    for i in range(10):
        rand.roll()
        x1, x2 = rand.get(['foo', 'bar'])
        assert x1 in pool_dict['foo'][-3:]
        assert x2 in pool_dict['bar'][-14:]

    # test seeding
    rand1 = TargetFileRandomizer(pool_dict=pool_dict, seed=0)
    rand2 = TargetFileRandomizer(pool_dict=pool_dict, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get('foo') == rand2.get('foo')
        rand1.roll(), rand2.roll()
        assert rand1.get('bar') == rand2.get('bar')
        rand1.roll(), rand2.roll()
        assert rand1.get(['foo', 'bar']) == rand2.get(['foo', 'bar'])

    rand1 = TargetFileRandomizer(pool_dict=pool_dict, seed=0)
    rand2 = TargetFileRandomizer(pool_dict=pool_dict, seed=42)
    rand1.roll(), rand2.roll()
    assert rand1.get('foo') != rand2.get('foo')
    rand1.roll(), rand2.roll()
    assert rand1.get('bar') != rand2.get('bar')
    rand1.roll(), rand2.roll()
    assert rand1.get(['foo', 'bar']) != rand2.get(['foo', 'bar'])


def test_noise_file():
    # test limits
    pool_dict = {
        'dcase': list(range(10)),
    }
    lims = [0.0, 0.3]
    rand = NoiseFileRandomizer(pool_dict=pool_dict, lims=lims, seed=0, size=3)
    for i in range(10):
        rand.roll()
        x = rand.get('dcase', 0)
        assert x in pool_dict['dcase'][0:3]

    lims = [0.7, 1.0]
    rand = NoiseFileRandomizer(pool_dict=pool_dict, lims=lims, seed=0, size=3)
    for i in range(10):
        rand.roll()
        x = rand.get('dcase', 0)
        assert x in pool_dict['dcase'][-3:]

    # test seeding
    rand1 = NoiseFileRandomizer(pool_dict=pool_dict, seed=0, size=3)
    rand2 = NoiseFileRandomizer(pool_dict=pool_dict, seed=0, size=3)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get('dcase', 0) == rand2.get('dcase', 0)

    rand1 = NoiseFileRandomizer(pool_dict=pool_dict, seed=0, size=3)
    rand2 = NoiseFileRandomizer(pool_dict=pool_dict, seed=42, size=3)
    rand1.roll(), rand2.roll()
    assert rand1.get('dcase', 0) != rand2.get('dcase', 0)

    # test multiple get calls
    rand = NoiseFileRandomizer(pool_dict=pool_dict, seed=0, size=3)
    rand.roll()
    rand.get('dcase', 0)
    rand.get('dcase', 1)
    rand.get('dcase', 2)
    with pytest.raises(ValueError):
        rand.get('dcase', 2)
    rand.roll()
    rand.get('dcase', 0)
    rand.get('dcase', 1)
    rand.get('dcase', 2)
    with pytest.raises(ValueError):
        rand.get('dcase', 2)

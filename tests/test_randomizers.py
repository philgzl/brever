import pytest

from brever.mixture.random import (  # isort: skip
    BaseRandGen,
    ChoiceRandGen,
    DistRandGen,
    MultiDistRandGen,
    MultiChoiceRandGen,
    AngleRandGen,
    TargetFileRandGen,
    NoiseFileRandGen,
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
        MultiDistRandGen('uniform', [0.0, 1.0]),
        MultiChoiceRandGen({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        AngleRandGen({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        TargetFileRandGen({'foo': [0, 1], 'bar': ['x', 'y', 'z']}),
        NoiseFileRandGen({'dcase': ['a', 'b', 'c']}, size=3),
    ]:
        if isinstance(rand, NoiseFileRandGen):
            def get():
                rand.get('dcase', 0)
        elif isinstance(rand, (
            MultiChoiceRandGen,
            AngleRandGen,
            TargetFileRandGen,
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


def test_multi_dist():
    rand1 = MultiDistRandGen('uniform', [0.0, 1.0], size=2, seed=0)
    rand2 = MultiDistRandGen('uniform', [0.0, 1.0], size=3, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get() == rand2.get()[:2]

    rand1 = MultiDistRandGen('uniform', [0.0, 1.0], size=3, seed=0)
    rand2 = MultiDistRandGen('uniform', [0.0, 1.0], size=3, seed=42)
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
    angle_min, angle_max = -30, 30

    # test generated angles are within the limits
    rand = AngleRandGen(pool_dict=pool_dict, size=3, seed=0,
                        lims=[angle_min, angle_max])
    for i in range(10):
        rand.roll()
        angles = rand.get('surrey')
        for angle in angles:
            assert angle_min <= angle <= angle_max

    # test generated angles can be outside the limits if they are not provided
    rand = AngleRandGen(pool_dict=pool_dict, size=1, seed=0)

    def experiment():
        rand.roll()
        angle = rand.get('ash')
        return not (angle_min <= angle <= angle_max)

    assert any(experiment() for i in range(10))

    # same but with squeeze=False
    rand = AngleRandGen(pool_dict=pool_dict, size=1, seed=0, squeeze=False)

    def experiment():
        rand.roll()
        angles = rand.get('surrey')
        for angle in angles:
            if not (angle_min <= angle <= angle_max):
                return True
        return False

    assert any(experiment() for i in range(10))

    # test seeding
    rand_1 = AngleRandGen(pool_dict=pool_dict, size=3, seed=0)
    rand_2 = AngleRandGen(pool_dict=pool_dict, size=3, seed=0)
    for i in range(10):
        rand_1.roll(), rand_2.roll()
        angles_1 = rand_1.get('ash')
        angles_2 = rand_2.get('ash')
        assert angles_1 == angles_2

    rand_1 = AngleRandGen(pool_dict=pool_dict, size=3, seed=0)
    rand_2 = AngleRandGen(pool_dict=pool_dict, size=3, seed=42)
    rand_1.roll(), rand_2.roll()
    angles_1 = rand_1.get('ash')
    angles_2 = rand_2.get('ash')
    assert angles_1 != angles_2


def test_target_file():
    # test limits
    pool_dict = {
        'foo': list(range(10)),
        'bar': list(range(42)),
    }
    lims = [0.0, 0.3]
    rand = TargetFileRandGen(pool_dict=pool_dict, lims=lims, seed=0)
    for i in range(10):
        rand.roll()
        x1, x2 = rand.get(['foo', 'bar'])
        assert x1 in pool_dict['foo'][0:3]
        assert x2 in pool_dict['bar'][0:14]

    lims = [0.7, 1.0]
    rand = TargetFileRandGen(pool_dict=pool_dict, lims=lims, seed=0)
    for i in range(10):
        rand.roll()
        x1, x2 = rand.get(['foo', 'bar'])
        assert x1 in pool_dict['foo'][-3:]
        assert x2 in pool_dict['bar'][-14:]

    # test seeding
    rand1 = TargetFileRandGen(pool_dict=pool_dict, seed=0)
    rand2 = TargetFileRandGen(pool_dict=pool_dict, seed=0)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get('foo') == rand2.get('foo')
        rand1.roll(), rand2.roll()
        assert rand1.get('bar') == rand2.get('bar')
        rand1.roll(), rand2.roll()
        assert rand1.get(['foo', 'bar']) == rand2.get(['foo', 'bar'])

    rand1 = TargetFileRandGen(pool_dict=pool_dict, seed=0)
    rand2 = TargetFileRandGen(pool_dict=pool_dict, seed=42)
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
    rand = NoiseFileRandGen(pool_dict=pool_dict, lims=lims, seed=0, size=3)
    for i in range(10):
        rand.roll()
        x = rand.get('dcase', 0)
        assert x in pool_dict['dcase'][0:3]

    lims = [0.7, 1.0]
    rand = NoiseFileRandGen(pool_dict=pool_dict, lims=lims, seed=0, size=3)
    for i in range(10):
        rand.roll()
        x = rand.get('dcase', 0)
        assert x in pool_dict['dcase'][-3:]

    # test seeding
    rand1 = NoiseFileRandGen(pool_dict=pool_dict, seed=0, size=3)
    rand2 = NoiseFileRandGen(pool_dict=pool_dict, seed=0, size=3)
    for i in range(10):
        rand1.roll(), rand2.roll()
        assert rand1.get('dcase', 0) == rand2.get('dcase', 0)

    rand1 = NoiseFileRandGen(pool_dict=pool_dict, seed=0, size=3)
    rand2 = NoiseFileRandGen(pool_dict=pool_dict, seed=42, size=3)
    rand1.roll(), rand2.roll()
    assert rand1.get('dcase', 0) != rand2.get('dcase', 0)

    # test multiple get calls
    rand = NoiseFileRandGen(pool_dict=pool_dict, seed=0, size=3)
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

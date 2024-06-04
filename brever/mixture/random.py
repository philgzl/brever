import random

import numpy as np

from .io import is_long_recording


class Seeder:
    """Random integer generator to seed other random generators."""

    def __init__(self, seed, max_seed=2**16):
        self.random = random.Random(seed)
        self.max_seed = max_seed

    def __call__(self):
        return self.random.randrange(self.max_seed)


class BaseRandGen:
    """Base class for all random generators.

    The `__init__` and `roll` methods should be overwritten to obtain desired
    behaviors.

    The subsequent classes are useful to generate random datasets that differ
    only along specific hyperparameters. For the same seed, we want two
    datasets to be identical along dimensions with common hyperparameter
    values (e.g. if the set of noise types is the same, the random noise
    recordings should be the same). Using the same random generator for the
    randomization of all dimensions would break this.
    """

    def __init__(self, seed=None):
        self.random = np.random.RandomState(seed)
        self._to_yield = None

    def roll(self):
        self._to_yield = self.random.rand()

    def get(self):
        if self._to_yield is None:
            raise ValueError('must call roll() before calling get()')
        output = self._to_yield
        self._to_yield = None
        return output


class ChoiceRandGen(BaseRandGen):
    """Pool of elements to randomly draw from.

    Supports weights for non-uniform probability distribution. Supports
    multiple draws, with or without replacement.

    When drawing more than one element, each extra element is drawn from a
    dedicated random generator. This means that for the same seed, drawing
    more elements will not change the sequence of elements drawn. This is why
    we can't use `np.random.choice` with `size > 1`.

    E.g.: Pool is `[1, 2, 3]`. First experiment draws 2 elements twice and
    obtains `[1, 3]` and `[1, 2]`. Second experiment has the same seed but
    draws 3 elements twice. Obtains `[1, 3, 3]` and `[1, 2, 1]`. Notice the
    first two elements in each draw are the same. If we were using the same
    random generator for every single number, the third number in the first
    draw would have changed the seed, and the second draw would have thus been
    completely different.
    """

    def __init__(self, pool, size=1, weights=None, replace=True, seed=None,
                 squeeze=True):
        super().__init__(seed)
        self.random = [
            np.random.RandomState(
                seed if seed is None else seed+i
            )
            for i in range(size)
        ]
        if isinstance(pool, set):
            self.pool = sorted(pool)
            if weights is not None:
                if not isinstance(weights, dict):
                    raise ValueError('weights must be dict when pool is set')
                if set(weights.keys()) != pool:
                    raise ValueError('weights keys do not match pool')
                weights = [weights[x] for x in self.pool]
        else:
            self.pool = pool
            if weights is not None:
                if not isinstance(weights, list):
                    raise ValueError('weights must be list when pool is list')
                if len(weights) != len(pool):
                    raise ValueError('weights and pool must have same length')
        if weights is not None:
            weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.replace = replace
        self.squeeze = squeeze

    def roll(self):
        self._to_yield = []
        current_pool = self.pool.copy()
        for rand in self.random:
            val = rand.choice(current_pool, p=self.weights).item()
            self._to_yield.append(val)
            if not self.replace:
                current_pool.remove(val)
        if len(self._to_yield) == 1 and self.squeeze:
            self._to_yield, = self._to_yield


class DistRandGen(BaseRandGen):
    """For arbitrary distributions as provided by `np.random`."""

    def __init__(self, dist_name, dist_args, seed=None):
        super().__init__(seed)
        self.dist_name = dist_name
        self.dist_args = dist_args

    def roll(self):
        dist_func = getattr(self.random, self.dist_name)
        self._to_yield = dist_func(*self.dist_args)


class MultiDistRandGen(BaseRandGen):
    """A list of `DistRandGen` objects."""

    def __init__(self, dist_name, dist_args, size=1, seed=None):
        self.random = []
        for i in range(size):
            self.random.append(
                DistRandGen(
                    dist_name=dist_name,
                    dist_args=dist_args,
                    seed=seed if seed is None else seed+i,
                )
            )
        self._to_yield = None

    def roll(self):
        self._to_yield = []
        for rand in self.random:
            rand.roll()
            self._to_yield.append(rand.get())

    def get(self):
        if self._to_yield is None:
            raise ValueError('must call roll() before calling get()')
        output = self._to_yield
        self._to_yield = None
        return output


class MultiChoiceRandGen(BaseRandGen):
    """Dictionary of `ChoiceRandGen` objects."""

    def __init__(self, pool_dict, size=1, replace=True, seed=None,
                 squeeze=True):
        self.random = {}
        if not pool_dict:
            raise ValueError('pool_dict cannot be empty')
        # !!! IMPORTANT: ITERATE OVER SORTED pool_dict KEYS, OTHERWISE THE
        # STATES OF THE ChoiceRandGens WILL NOT BE DETERMINISTIC !!!
        for i, key in enumerate(sorted(pool_dict.keys())):
            self.random[key] = ChoiceRandGen(
                pool=pool_dict[key],
                size=size,
                replace=replace,
                seed=seed if seed is None else seed+i,
                squeeze=squeeze,
            )
        self._to_yield = None

    def roll(self):
        self._to_yield = {}
        for key, rand in self.random.items():
            rand.roll()
            self._to_yield[key] = rand.get()

    def get(self, key):
        if self._to_yield is None:
            raise ValueError('must call roll() before calling get()')
        if isinstance(key, list):
            list_input = True
        else:
            key = [key]
            list_input = False
        output = [self._to_yield[key] for key in key]
        self._to_yield = None
        if not list_input:
            output, = output
        return output


class AngleRandGen(MultiChoiceRandGen):
    """Target and noise angle randomizer."""

    def __init__(self, pool_dict, size=1, replace=False, lims=None,
                 parity='all', seed=None, squeeze=True):
        pool_dict = {
            room: self.filter_angles(angles, lims, parity)
            for room, angles in pool_dict.items()
        }
        super().__init__(
            pool_dict=pool_dict,
            size=size,
            replace=replace,
            seed=seed,
            squeeze=squeeze,
        )

    def filter_angles(self, angles, lims, parity):
        angles = sorted(angles)
        if parity == 'all':
            pass
        elif parity == 'even' or parity == 'odd':
            even_angles = angles[::2]
            odd_angles = angles[1::2]
            if 0 not in even_angles:
                even_angles, odd_angles = odd_angles, even_angles
            if parity == 'even':
                angles = even_angles
            else:
                angles = odd_angles
        else:
            raise ValueError(f'parity must be all, odd or even, got {parity}')
        if lims is not None:
            a_min, a_max = lims
            angles = [a for a in angles if a_min <= a <= a_max]
        return angles


class TargetFileRandGen(MultiChoiceRandGen):
    """Target file randomizer.

    Basically a `MultiChoiceRandGen` with file limit support.
    """

    def __init__(self, pool_dict, *args, lims=[0.0, 1.0], **kwargs):
        super().__init__(
            pool_dict=self.make_pool_dict(pool_dict, lims),
            *args,
            **kwargs,
        )

    def make_pool_dict(self, pool_dict, lims):
        output = {}
        for key, files in pool_dict.items():
            n = len(files)
            i_min, i_max = round(n*lims[0]), round(n*lims[1])
            output[key] = files[i_min:i_max]
        return output


class NoiseFileRandGen(MultiChoiceRandGen):
    """Noise file randomizer.

    Basically a `MultiChoiceRandGen` with file limit support and a sensibly
    different `get` method which can be called as many times as the roll size.
    """

    def __init__(self, pool_dict, *args, lims=[0.0, 1.0], size=1, **kwargs):
        super().__init__(
            pool_dict=self.make_pool_dict(pool_dict, lims),
            *args,
            size=size,
            **kwargs,
        )
        # set replace=True for ChoiceRandGen objects of colored noise
        for key, rand in self.random.items():
            if key.startswith('colored_'):
                rand.replace = True
        self.size = size
        self.counter = [False]*self.size

    def make_pool_dict(self, pool_dict, lims):
        output = {}
        for key, files in pool_dict.items():
            if not key.startswith('colored_') and not is_long_recording(key):
                n = len(files)
                i_min, i_max = round(n*lims[0]), round(n*lims[1])
                files = files[i_min:i_max]
            output[key] = files
        return output

    def roll(self):
        super().roll()
        self.counter = [False]*self.size

    def get(self, noise, idx):
        if self._to_yield is None or self.counter[idx]:
            raise ValueError('must call roll() before calling get()')
        output = self._to_yield[noise][idx]
        self.counter[idx] = True
        if all(self.counter):
            self._to_yield = None
            self.counter = [False]*self.size
        return output

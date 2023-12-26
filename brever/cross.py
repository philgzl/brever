import itertools
import os

import h5py
import numpy as np

from .config import DatasetInitializer, ModelInitializer
from .inspect import Path


class CrossCorpusExperiment:
    """Cross-corpus experiment initializer and score loader.

    Initializes models and datasets given speech, noise and rooms databases,
    specifications for training, validation and test datasets, and model
    architectures. Can also be used to load and gather the scores once the
    models are trained and tested.

    Parameters
    ----------
    model_archs : list
        List of model architectures.
    databases : dict, optional
        Mapping from acoustic dimensions `'speakers'`, `'noises'` and `'rooms'`
        to lists of database regexps. The lists of database regexps must have
        the same length. Default is `None`, which means some default databases
        are used.
    dset_spec : dict, optional
        Mapping from dataset types `'train'`, `'val'` or `'test'` to
        specifications for the datasets. Each dataset specification is a dict
        with keys `'seed'`, `'duration'`, `'speech_files'`, `'noise_files'`,
        `'room_files'` and `'weight_by_avg_length'`. Default is `None`, which
        means some default specifications are used.
    metrics : list, optional
        List of metric names. Used to load scores. Default is `['pesq',
        'estoi', 'snr']`.
    delta_scores : bool, optional
        Whether to compute delta scores when loading scores. Default is `True`.
    model_seeds : list, optional
        Model seeds. Default is `[0]`, i.e. only one model per training
        database is initialized.
    model_kwargs : dict, optional
        Extra model keyword arguments. Default is `None`.
    """

    def __init__(
        self,
        model_archs,
        databases=None,
        dset_spec=None,
        metrics=['pesq', 'estoi', 'snr'],
        delta_scores=True,
        model_seeds=[0],
        model_kwargs={},
    ):
        if databases is None:
            databases = dict(
                speakers=[
                    'timit_.*',
                    'libri_.*',
                    'wsj0_.*',
                    'clarity_.*',
                    'vctk_.*',
                ],
                noises=[
                    'dcase_.*',
                    'noisex_.*',
                    'icra_.*',
                    'demand',
                    'arte',
                ],
                rooms=[
                    'surrey_.*',
                    'ash_.*',
                    'bras_.*',
                    'catt_.*',
                    'avil_.*',
                ],
            )

        if dset_spec is None:
            dset_spec = dict(
                train=dict(
                    seed=0,
                    duration=30*60*60,
                    speech_files=(0.0, 0.8),
                    noise_files=(0.0, 0.8),
                    room_files='even',
                    weight_by_avg_length=True,
                ),
                val=dict(
                    seed=1337,
                    duration=30*60,
                    speech_files=(0.0, 0.8),
                    noise_files=(0.0, 0.8),
                    room_files='even',
                    weight_by_avg_length=False,
                ),
                test=dict(
                    seed=42,
                    duration=60*60,
                    speech_files=(0.8, 1.0),
                    noise_files=(0.8, 1.0),
                    room_files='odd',
                    weight_by_avg_length=False,
                ),
            )

        self.databases = databases
        self.dset_spec = dset_spec
        self.metrics = metrics
        self.delta_scores = delta_scores
        self.model_seeds = model_seeds
        self.model_archs = model_archs
        self.model_kwargs = model_kwargs

        self.dset_init = DatasetInitializer(batch_mode=True)
        self.model_init = ModelInitializer(batch_mode=True)

        self._main_models = {1: set(), self.n_db - 1: set()}
        self._dsets = set()
        self._evals = dict()

    def apply_dset_func(self, func, kind, speakers, noises, rooms):
        return func(
            kind=kind,
            speakers=speakers,
            noises=noises,
            rooms=rooms,
            **self.dset_spec[kind],
        )

    def get_dset(self, *args, **kwargs):
        return self.apply_dset_func(self.dset_init.get_path_from_kwargs,
                                    *args, **kwargs)

    def get_train_dset(self, *args, **kwargs):
        return self.get_dset('train', *args, **kwargs)

    def get_val_dset(self, *args, **kwargs):
        return self.get_dset('val', *args, **kwargs)

    def get_test_dset(self, *args, **kwargs):
        return self.get_dset('test', *args, **kwargs)

    def init_dset(self, *args, **kwargs):
        return self.apply_dset_func(self.dset_init.init_from_kwargs,
                                    *args, **kwargs)

    def init_train_dset(self, *args, **kwargs):
        return self.init_dset('train', *args, **kwargs)

    def init_val_dset(self, *args, **kwargs):
        return self.init_dset('val', *args, **kwargs)

    def init_test_dset(self, *args, **kwargs):
        return self.init_dset('test', *args, **kwargs)

    def apply_model_func(self, func, arch, train_path, val_path, seed):
        return func(
            arch=arch,
            train_path=Path(train_path),
            val_path=Path(val_path),
            seed=seed,
            **self.model_kwargs,
        )

    def get_model(self, *args, **kwargs):
        return self.apply_model_func(self.model_init.get_path_from_kwargs,
                                     *args, **kwargs)

    def init_model(self, *args, **kwargs):
        return self.apply_model_func(self.model_init.init_from_kwargs,
                                     *args, **kwargs)

    def init_all_test_dsets(self):
        test_paths = []
        for indexes in itertools.product(range(self.n_db), repeat=self.n_dim):
            kwargs = {
                dim: {dbs[i]}
                for (dim, dbs), i in zip(self.databases.items(), indexes)
            }
            test_path = self.init_test_dset(**kwargs)
            test_paths.append(test_path)
            self._dsets.add(test_path)
        return test_paths

    def get_all_test_dsets(self):
        idx_list = [list(range(self.n_db)) for _ in range(self.n_dim)]
        return self.get_test_dsets(idx_list)

    def get_test_dsets(self, idx_list):
        test_paths = []
        for indexes in itertools.product(*idx_list):
            kwargs = {
                dim: {dbs[i]}
                for (dim, dbs), i in zip(self.databases.items(), indexes)
            }
            test_path = self.get_test_dset(**kwargs)
            test_paths.append(test_path)
        return test_paths

    @property
    def n_db(self):
        n_db = len(next(iter(self.databases.values())))
        assert all(len(dbs) == n_db for dbs in self.databases.values())
        return n_db

    @property
    def n_dim(self):
        return len(self.databases)

    @property
    def n_metrics(self):
        return len(self.metrics)

    @property
    def n_archs(self):
        return len(self.model_archs)

    @property
    def n_mismatches(self):
        return 2**self.n_dim

    def _complementary_idx(self, idx_list):
        return [i for i in range(self.n_db) if i not in idx_list]

    def _train_db_idx(self, fold_idx, N):
        if N == 1:
            return [[fold_idx]]*self.n_dim
        elif N == self.n_db - 1:
            return [self._complementary_idx([fold_idx])]*self.n_dim
        else:
            raise ValueError(
                f'N must be 1 (low diversity training) or {self.n_db - 1} '
                f'(high diversity training), got {N}')

    def _test_db_idx(self, train_idx, dims):
        test_idx = [
            self._complementary_idx(train_idx[i]) for i in range(self.n_dim)
        ]
        for dim in dims:
            test_idx[dim] = train_idx[dim]
        return test_idx

    def _build_dset_kwargs(self, idx_list):
        kwargs = {}
        for (dim, dbs), indexes in zip(self.databases.items(), idx_list):
            kwargs[dim] = {dbs[i] for i in indexes}
        return kwargs

    def get_scores(self, model, test_paths):
        filename = os.path.join(model, 'scores.hdf5')
        h5f = h5py.File(filename)
        metric_idx = [
            list(h5f['metrics'].asstr()).index(m) for m in self.metrics
        ]
        scores = []
        for test_path in test_paths:
            h5path = f'last.ckpt/{os.path.basename(test_path)}'
            if h5path not in h5f.keys():
                raise ValueError(f'{model} not tested on {test_path}')
            scores.append(h5f[h5path][:, metric_idx, :])
        scores = np.concatenate(scores, axis=0)
        if self.delta_scores:
            scores = scores[:, :, 1] - scores[:, :, 0]
        else:
            scores = scores[:, :, 1]
        mean, std = scores.mean(axis=0), scores.std(axis=0)
        h5f.close()
        return mean, std

    def _init_fold(self, i_fold, matching_dims, N):
        # main model
        train_idx = self._train_db_idx(i_fold, N)
        train_kwargs = self._build_dset_kwargs(train_idx)
        train_path = self.init_train_dset(**train_kwargs)
        val_path = self.init_val_dset(**train_kwargs)
        # reference model
        train_idx_ref = self._test_db_idx(train_idx, matching_dims)
        train_kwargs_ref = self._build_dset_kwargs(train_idx_ref)
        train_path_ref = self.init_train_dset(**train_kwargs_ref)
        val_path_ref = self.init_val_dset(**train_kwargs_ref)
        # test dataset
        test_path = self.init_test_dset(**train_kwargs_ref)

        for kwargs in self.dict_product(
            arch=self.model_archs,
            seed=self.model_seeds,
        ):
            m = self.init_model(
                train_path=Path(train_path),
                val_path=Path(val_path),
                **kwargs,
            )
            m_ref = self.init_model(
                train_path=Path(train_path_ref),
                val_path=Path(val_path_ref),
                **kwargs,
            )
            if m not in self._evals:
                self._evals[m] = set()
            if m_ref not in self._evals:
                self._evals[m_ref] = set()
            self._evals[m].add(test_path)
            self._evals[m_ref].add(test_path)
            self._main_models[N].add(m)
        self._dsets.add(train_path)
        self._dsets.add(val_path)
        self._dsets.add(train_path_ref)
        self._dsets.add(val_path_ref)
        self._dsets.add(test_path)

    def init_experiment(self, eval_script):
        self._dsets = set()
        for n_match in reversed(range(self.n_dim)):  # number of MATCHING dims
            for dims in itertools.combinations(range(self.n_dim), n_match):
                for N in [1, self.n_db - 1]:  # low or high diversity training
                    for i_fold in range(self.n_db):
                        self._init_fold(i_fold, dims, N)

        # test_paths = self.init_all_test_dsets()
        self.write_eval_script(eval_script)
        self.check_deprecated_models()
        self.check_deprecated_dsets()
        self.print_main_models()

    def init_mini_experiment(self):
        to_print = {}
        for N in [1, self.n_db - 1]:  # low or high diversity training
            to_print[N] = []
            for i_fold in range(self.n_db):
                train_idx = self._train_db_idx(i_fold, N)
                train_kwargs = self._build_dset_kwargs(train_idx)
                train_path = self.init_train_dset(**train_kwargs)
                val_path = self.init_val_dset(**train_kwargs)
                test_idx = self._train_db_idx(i_fold, 1)
                test_kwargs = self._build_dset_kwargs(test_idx)
                test_path = self.init_test_dset(**test_kwargs)
                models = []
                for kwargs in self.dict_product(
                    arch=self.model_archs,
                    seed=self.model_seeds,
                ):
                    m = self.init_model(
                        train_path=Path(train_path),
                        val_path=Path(val_path),
                        **kwargs,
                    )
                    models.append(m)
                to_print[N].append([train_path, val_path, test_path] + models)
        for N, to_print in to_print.items():
            print(f'N={N}')
            for x in to_print:
                print(' '.join(x))

    def print_main_models(self):
        for N, models in self._main_models.items():
            print(f'N={N} main models:')
            for model in models:
                print(model)

    def write_eval_script(self, eval_script, cores=4, memory=4, batch_size=50):
        with open(eval_script, 'w') as f:
            for m, test_paths in self._evals.items():
                test_paths_str = ' '.join(test_paths)
                f.write(
                    f'./lsf/test_model.sh -i {m}  -t "{test_paths_str}" '
                    f'-c {cores} -m {memory} -b {batch_size}\n'
                )

    def check_deprecated_models(self):
        model_dir = self.model_init.dir_
        for model_id in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_id)
            if model_path not in self._evals:
                print('the following model was found in the system and was '
                      f'not attempted to be initialized: {model_path}')

    def check_deprecated_dsets(self):
        dset_dir = self.dset_init.dir_
        for kind in ['train', 'val', 'test']:
            subdir = os.path.join(dset_dir, kind)
            for dset_id in os.listdir(subdir):
                dset_path = os.path.join(subdir, dset_id).replace('\\', '/')
                if dset_path not in self._dsets:
                    print('the following dataset was found in the system and '
                          f'was not attempted to be initialized: {dset_path}')

    def _get_fold_scores(self, i_fold, matching_dims, N, seed):
        # main model
        train_idx = self._train_db_idx(i_fold, N)
        train_kwargs = self._build_dset_kwargs(train_idx)
        train_path = self.get_train_dset(**train_kwargs)
        val_path = self.get_val_dset(**train_kwargs)
        # reference model
        train_idx_ref = self._test_db_idx(train_idx, matching_dims)
        train_kwargs_ref = self._build_dset_kwargs(train_idx_ref)
        train_path_ref = self.get_train_dset(**train_kwargs_ref)
        val_path_ref = self.get_val_dset(**train_kwargs_ref)
        # test paths
        # test_paths = self.get_test_dsets(train_idx_ref)
        test_paths = [self.get_test_dset(**train_kwargs_ref)]
        for i_arch, arch in enumerate(self.model_archs):
            m = self.get_model(arch, train_path, val_path, seed)
            m_ref = self.get_model(arch, train_path_ref, val_path_ref, seed)
            mean, std = self.get_scores(m, test_paths)
            ref_mean, ref_std = self.get_scores(m_ref, test_paths)
            yield mean, std, ref_mean, ref_std

    def _get_matched_scores(self, i_fold, matching_dims, N, seed):
        train_idx = self._train_db_idx(i_fold, N)
        train_kwargs = self._build_dset_kwargs(train_idx)
        train_path = self.get_train_dset(**train_kwargs)
        val_path = self.get_val_dset(**train_kwargs)
        # test_paths = self.get_test_dsets(train_idx)
        test_paths = [self.get_test_dset(**train_kwargs)]
        for i_arch, arch in enumerate(self.model_archs):
            m = self.get_model(arch, train_path, val_path, seed)
            mean, std = self.get_scores(m, test_paths)
            yield mean, std

    def gather_all_scores(self, seed):
        shape = (2, self.n_mismatches, self.n_db, self.n_archs, self.n_metrics)
        mean = np.empty(shape)
        std = np.empty(shape)
        ref_mean = np.empty(shape)
        ref_std = np.empty(shape)

        for i_n, N in enumerate([1, self.n_db - 1]):

            i_mism = 0

            for ndim in range(self.n_dim):  # number of matching dimensions
                for dims in itertools.combinations(range(self.n_dim), ndim):
                    for i_fold in range(self.n_db):
                        for i_arch, data in enumerate(self._get_fold_scores(
                            i_fold, dims, N, seed
                        )):
                            mean[i_n, i_mism, i_fold, i_arch, :] = data[0]
                            std[i_n, i_mism, i_fold, i_arch, :] = data[1]
                            ref_mean[i_n, i_mism, i_fold, i_arch, :] = data[2]
                            ref_std[i_n, i_mism, i_fold, i_arch, :] = data[3]

                    i_mism += 1

        # last mismatch scenario: matched case
        for i_n, N in enumerate([1, self.n_db - 1]):
            for dims in [tuple(range(self.n_dim))]:
                for i_fold in range(self.n_db):
                    for i_arch, data in enumerate(self._get_matched_scores(
                        i_fold, dims, N, seed
                    )):
                        mean[i_n, -1, i_fold, i_arch, :] = data[0]
                        std[i_n, -1, i_fold, i_arch, :] = data[1]
                        ref_mean[i_n, -1, i_fold, i_arch, :] = data[0]
                        ref_std[i_n, -1, i_fold, i_arch, :] = data[1]

        return mean, std, ref_mean, ref_std

    @staticmethod
    def dict_product(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    def get_all_models(self):
        output = set()
        for n_match in reversed(range(self.n_dim)):  # number of MATCHING dims
            for dims in itertools.combinations(range(self.n_dim), n_match):
                for N in [1, self.n_db - 1]:  # low or high diversity training
                    for i_fold in range(self.n_db):
                        # main model
                        train_idx = self._train_db_idx(i_fold, N)
                        train_kw = self._build_dset_kwargs(train_idx)
                        train_path = self.get_train_dset(**train_kw)
                        val_path = self.get_val_dset(**train_kw)
                        # reference model
                        train_idx_ref = self._test_db_idx(train_idx, dims)
                        train_kw_ref = self._build_dset_kwargs(train_idx_ref)
                        train_path_ref = self.get_train_dset(**train_kw_ref)
                        val_path_ref = self.get_val_dset(**train_kw_ref)
                        for kwargs in self.dict_product(
                            arch=self.model_archs,
                            seed=self.model_seeds,
                        ):
                            output.add(self.get_model(
                                train_path=Path(train_path),
                                val_path=Path(val_path),
                                **kwargs,
                            ))
                            output.add(self.get_model(
                                train_path=Path(train_path_ref),
                                val_path=Path(val_path_ref),
                                **kwargs,
                            ))
        return list(output)

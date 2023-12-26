from typing import Callable

import torch
import torch.nn as nn

from ..criterion import CriterionRegistry
from ..registry import Registry

ModelRegistry = Registry('model')


class BreverBaseModel(nn.Module):
    """Base class for all models.

    Subclasses must implement the `train_step`, `val_step`, `optimizers` and
    `_enhance` methods. The `transform` and `pre_train` methods can be
    overwritten if relevant.

    The `__init__` method accepts a convenience `criterion` argument to assign
    a callable `criterion` attribute, though it is not mandatory to use it.

    Parameters
    ----------
    criterion : callable or str or None, optional
        Function usually used in the training step to calculate the loss. If
        `str`, a callable is loaded from the criterion registry. If `None`, the
        `criterion` attribute is not set. Default is `None`.
    """
    def __init__(
        self,
        criterion: Callable[..., torch.Tensor] | str | None = None,
    ):
        super().__init__()
        if criterion is not None:
            if isinstance(criterion, str):
                criterion = CriterionRegistry.get(criterion)
            self.criterion = criterion
        self._compiled_call_impl = None

    def optimizers(self):
        """Optimizer getter.

        Method called by the trainer to get initialized optimizers. In most
        cases this returns only one optimizer, but in some cases multiple
        optimizers are used, e.g. with GANs.

        Note the optimizers should be initialized in the model `__init__`
        method and not here.

        Returns
        -------
        optimizers : torch.optim.Optimizer or sequence of torch.optim.Optimizer
            Initialized optimizers. Can be a single optimizer or multiple
            optimizers.
        """
        raise NotImplementedError

    def transform(self, sources):
        """Model input pre-processing.

        Pre-processing that can be separated from inference in `forward`.
        Executed by workers when loading the data before making mini-batches,
        but also during validation to pre-process waveforms already moved to
        device. This means this should be able to run on CPU or GPU depending
        on the input device, even if the model was moved to GPU!

        Parameters
        ----------
        sources : torch.Tensor
            Input sources. Shape `(n_sources, n_channels, n_samples)`, where
            `n_sources = len(sources)`, `n_channels = 2` (left and right) and
            `n_samples` is the segment length.

        Returns
        -------
        model_inputs : torch.Tensor or tuple of torch.Tensor
            Pre-processed model inputs with arbitrary shapes. The last
            dimension of tensors should be homogeneous to time (e.g. STFT
            frames), such that it can be padded to form mini-batches if needed.
        """
        return sources

    def enhance(self, x, use_amp=False):
        """Noisy speech signal enhancement or source separation.

        Given a binaural noisy input mixture, estimates a single-channel clean
        speech signal if the model is a speech enhancement system, or
        single-channel separated sources if it's a source separation network.
        Supports batched inputs. Models should not overwrite this method and
        should overwrite `_enhance` instead.

        Parameters
        ----------
        x : torch.Tensor
            Binaural noisy mixture. Shape `(2, n_samples)` or `(batch_size, 2,
            n_samples)`.
        use_amp : bool, optional
            Whether to use automatic mixed precision. Default is `False`.

        Returns
        -------
        y : torch.Tensor
            Single-channel enhanced signal or separated sources. If unbatched
            inputs, shape `(n_samples,)` if enhanced signal or `(n_sources,
            n_samples)` if separated sources. If batched, shape `(batch_size,
            n_samples)` or `(batch_size, n_sources, n_samples)`.
        """
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        elif x.ndim != 3:
            raise ValueError(f'input must be 2 or 3 dimensional, got {x.ndim}')
        output = self._enhance(x, use_amp)
        if unbatched:
            output = output.squeeze(0)
        return output

    def _enhance(self, x, use_amp):
        """Batched noisy speech signal enhancement or source separation.

        Same as `enhance` but assumes batched inputs. Called inside `enhance`.
        All models must overwrite this method.

        Parameters
        ----------
        x : torch.Tensor
            Batched binaural noisy mixture. Shape `(batch_size, 2, n_samples)`.
        use_amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        y : torch.Tensor
            Batched single-channel enhanced signal or separated sources. Shape
            `(batch_size, n_samples)` or `(batch_size, n_sources, n_samples)`.
        """
        raise NotImplementedError

    def train_step(self, batch, lengths, use_amp, scaler):
        """Training step.

        Sets gradients to zero, calculates the loss and backpropagates it given
        batched observations from the dataloader. Called in the training loop.

        Parameters
        ----------
        batch : torch.Tensor or list of torch.Tensor
            Batched model inputs from the dataloader. `batch` is a tensor if
            the dataset's `__getitem__` returns a single output, or a list of
            tensors if it returns multiple outputs.
        lengths : torch.Tensor
            Original model input lengths along the last dimension. Can be
            important for post-processing, e.g. to ensure sample-wise losses
            are not aggregated over the zero-padded regions. Shape
            `(batch_size,)` if the dataset's `__getitem__` returns a single
            output, else shape `(batch_size, n_model_inputs)`.
        use_amp : bool
            Whether to use automatic mixed precision.
        scaler : torch.cuda.amp.GradScaler
            Gradient scaler for automatic mixed precision.

        Returns
        -------
        loss : torch.Tensor or dict
            Loss to backpropagate. Can be a dict if multiple losses are
            calculated like with GANs.
        """
        raise NotImplementedError

    def val_step(self, batch, lengths, use_amp):
        """Validation step.

        Calculates the loss to log given batched observations from the
        dataloader in a torch.no_grad() context manager. Called in the
        validation loop.

        Parameters
        ----------
        batch : torch.Tensor or list of torch.Tensor
            Batched model inputs from the dataloader. `batch` is a tensor if
            the dataset's `__getitem__` returns a single output, or a list of
            tensors if it returns multiple outputs.
        lengths : torch.Tensor
            Original model input lengths along the last dimension. Can be
            important for post-processing, e.g. to ensure sample-wise losses
            are not aggregated over the zero-padded regions. Shape
            `(batch_size,)` if the dataset's `__getitem__` returns a single
            output, else shape `(batch_size, n_model_inputs)`.
        use_amp : bool
            Whether to use automatic mixed precision.

        Returns
        -------
        loss : torch.Tensor or dict
            Loss to log. Can be a dict if multiple losses are calculated like
            with GANs.
        """
        raise NotImplementedError

    def pre_train(self, dataset, dataloader, epochs):
        """Pre-training instructions.

        Contains instructions that need to be run once before the training
        loop, e.g. calculate normalization statistics of input features.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The training dataset. Can be a `torch.utils.data.Subset` from a
            train/val split.
        dataloader : torch.utils.data.DataLoader
            The associated training dataloader.
        epochs : int
            Number of epochs. Useful for setting learning rate schedules.
        """
        pass

    def compile(self, *args, **kwargs):
        """In-place module compilation.

        Calls `torch.compile` on the model's `__call__`. Arguments are passed
        to `torch.compile`. See
        https://discuss.pytorch.org/t/how-should-i-use-torch-compile-properly/179021/6
        and the PR linked in the discussion for more details.

        In-place compilation might make it to torch core in the future, in
        which case this should be removed.
        """
        self._compiled_call_impl = torch.compile(self._call_impl, *args,
                                                 **kwargs)

    def __call__(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)
        else:
            return self._call_impl(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove _compiled_call_impl from the state
        state.pop('_compiled_call_impl', None)
        return state

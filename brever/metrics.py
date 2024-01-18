from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from batch_pystoi import stoi as batch_pystoi_stoi
from pesq import pesq as pesq_pesq
from pesq._pesq import (USAGE_BATCH, _check_fs_mode, _pesq_inner,
                        _processor_mapping)
from pesq.cypesq import PesqError
from pystoi import stoi as pystoi_stoi

from .criterion import CriterionRegistry
from .registry import Registry

MetricRegistry = Registry('metric')


def _stoi(x, y, fs, extended, batched, lengths):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.cpu().numpy()
    if batched:
        output = batch_pystoi_stoi(y, x, fs, extended=extended,
                                   lengths=lengths)
        if x.ndim == 1:
            output = output.item()
    else:
        if x.ndim == 1:
            if lengths is not None:
                raise ValueError('Non-batched stoi does not support lengths '
                                 'argument for 1D inputs.')
            output = pystoi_stoi(y, x, fs, extended=extended)
        else:
            if lengths is None:
                lengths = [x.shape[-1]]*x.shape[0]
            output = np.array([
                pystoi_stoi(yi[:length], xi[:length], fs, extended=extended)
                for xi, yi, length in zip(x, y, lengths)
            ])
    return output


def _pesq(x, y, fs, mode, normalized, batched, lengths):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.cpu().numpy()
    if batched:
        output = np.array(pesq_batch(fs, y, x, mode=mode, lengths=lengths))
        if x.ndim == 1:
            output = output.item()
    else:
        if x.ndim == 1:
            if lengths is not None:
                raise ValueError('Non-batched pesq does not support lengths '
                                 'argument for 1D inputs.')
            output = pesq_pesq(fs, y, x, mode=mode)
        else:
            if lengths is None:
                lengths = [x.shape[-1]]*x.shape[0]
            output = np.array([
                pesq_pesq(fs, yi[:length], xi[:length], mode=mode)
                for xi, yi, length in zip(x, y, lengths)
            ])
    if normalized:
        # see https://github.com/ludlows/PESQ/issues/13
        if mode == 'nb':
            # min = 1.016843313292765
            min = 1.0
            max = 4.548638319075995
        elif mode == 'wb':
            # min = 1.042694226789194
            min = 1.0
            max = 4.643888749336258
        else:
            raise ValueError(f"mode must be 'nb' or 'wb', got '{mode}'")
        output = (output - min) / (max - min)
        if (
            isinstance(output, np.ndarray)
            and (any(output < 0) or any(output > 1))
        ) or (
            isinstance(output, float)
            and (output < 0 or output > 1)
        ):
            raise RuntimeError('normalized PESQ score is out of bounds: '
                               f'{output}')
    return output


@MetricRegistry.register('pesq')
def pesq(x, y, fs=16000, mode='wb', normalized=False, batched=True,
         lengths=None):
    return _pesq(x, y, fs, mode, normalized, batched, lengths)


@MetricRegistry.register('stoi')
def stoi(x, y, fs=16000, batched=True, lengths=None):
    return _stoi(x, y, fs, False, batched, lengths)


@MetricRegistry.register('estoi')
def estoi(x, y, fs=16000, batched=True, lengths=None):
    return _stoi(x, y, fs, True, batched, lengths)


@MetricRegistry.register('snr')
def snr(x, y, lengths=None):
    x, y, lengths, unbatched = _check_input(x, y, lengths)
    output = - CriterionRegistry.get('snr')(x, y, lengths)
    return output.item() if unbatched else output


@MetricRegistry.register('sisnr')
def sisnr(x, y, lengths=None):
    x, y, lengths, unbatched = _check_input(x, y, lengths)
    output = - CriterionRegistry.get('sisnr')(x, y, lengths)
    return output.item() if unbatched else output


def _check_input(x, y, lengths):
    if x.shape != y.shape:
        raise ValueError('inputs must have same shape, got '
                         f'{x.shape} and {y.shape}')
    # add batch dimension
    unbatched = x.ndim == 1
    if unbatched:
        x, y = x.unsqueeze(0), y.unsqueeze(0)
    # add source dimension
    if x.ndim == 2:
        x, y = x.unsqueeze(1), y.unsqueeze(1)
    else:
        raise ValueError(f'input must be 1 or 2 dimensional, got {x.ndim}')
    # check lengths items are smaller than input length
    if lengths is None:
        lengths = torch.full((x.shape[0],), x.shape[-1], device=x.device)
    else:
        if len(lengths) != x.shape[0]:
            raise ValueError('lengths must have same length as batch size, '
                             f'got {len(lengths)} and {x.shape[0]}')
        if any(length > x.shape[-1] for length in lengths):
            raise ValueError('lengths items must be smaller than input '
                             f'length, got lengths={lengths} and '
                             f'input.shape={x.shape}')
    return x, y, lengths, unbatched


def pesq_batch(fs, ref, deg, mode, n_processor=cpu_count(),
               on_error=PesqError.RAISE_EXCEPTION, lengths=None):
    """Batched PESQ with lengths argument support.

    This is a copy/paste of https://github.com/ludlows/PESQ/pull/46 and
    should be removed if the PR ever gets approved.
    """
    _check_fs_mode(mode, fs, USAGE_BATCH)
    # check dimension
    if len(ref.shape) == 1:
        if lengths is not None:
            raise ValueError("cannot provide lengths if ref is 1D")
        if len(deg.shape) == 1 and ref.shape == deg.shape:
            return [_pesq_inner(ref, deg, fs, mode, PesqError.RETURN_VALUES)]
        elif len(deg.shape) == 2 and ref.shape[-1] == deg.shape[-1]:
            if n_processor <= 0:
                pesq_score = [np.nan for i in range(deg.shape[0])]
                for i in range(deg.shape[0]):
                    pesq_score[i] = _pesq_inner(
                        ref, deg[i, :], fs, mode, on_error
                    )
                return pesq_score
            else:
                with Pool(n_processor) as p:
                    return p.map(
                        partial(_pesq_inner, ref, fs=fs, mode=mode,
                                on_error=on_error),
                        [deg[i, :] for i in range(deg.shape[0])]
                    )
        else:
            raise ValueError("The shapes of `deg` is invalid!")
    elif len(ref.shape) == 2:
        if deg.shape == ref.shape:
            if lengths is None:
                lengths = [ref.shape[-1] for _ in range(ref.shape[0])]
            elif len(lengths) != ref.shape[0]:
                raise ValueError("len(lengths) does not match the batch size")
            if n_processor <= 0:
                pesq_score = [np.nan for i in range(deg.shape[0])]
                for i in range(deg.shape[0]):
                    pesq_score[i] = _pesq_inner(
                        ref[i, :lengths[i]], deg[i, :lengths[i]],
                        fs, mode, on_error
                    )
                return pesq_score
            else:
                return _processor_mapping(
                    _pesq_inner,
                    [
                        (
                            ref[i, :lengths[i]], deg[i, :lengths[i]],
                            fs, mode, on_error
                        )
                        for i in range(deg.shape[0])
                    ],
                    n_processor
                )
        else:
            raise ValueError("The shape of `deg` is invalid!")
    else:
        raise ValueError("The shape of `ref` should be either 1D or 2D!")

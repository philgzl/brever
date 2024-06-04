import argparse
import logging
import os
import pprint
import re
import sys

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from brever.batching import DistributedBatchSamplerWrapper, SortedBatchSampler
from brever.config import get_config
from brever.data import BreverDataLoader, BreverDataset
from brever.inspect import Path
from brever.logger import set_logger
from brever.metrics import MetricRegistry
from brever.models import ModelRegistry


def significant_figures(x, n=2):
    if isinstance(x, dict):
        return {k: significant_figures(v, n) for k, v in x.items()}
    elif x == 0:
        return x
    else:
        return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))


def main(i_input, input_):
    progress = f'[{i_input}/{len(args.inputs)}]'

    # check if model exists
    if not os.path.exists(input_):
        print(f'Model {input_} does not exist {progress}')
        return

    checkpoints_dir = os.path.join(input_, 'checkpoints')
    if input_.endswith('.ckpt'):
        model_dir = os.path.dirname(os.path.dirname(input_))
        checkpoint_path = input_
    else:
        model_dir = input_
        checkpoint_path = os.path.join(checkpoints_dir, 'last.ckpt')
    if args.best is not None:
        checkpoint_path = find_best_checkpoint(checkpoints_dir, args.best)

    # check if model is trained
    loss_path = os.path.join(model_dir, 'losses.npz')
    if not os.path.exists(loss_path) and not args.no_train_check:
        print(f'Model {input_} is not trained {progress}')
        return

    # load model config
    cfg_path = os.path.join(model_dir, 'config.yaml')
    cfg = get_config(cfg_path)

    # initialize ddp
    rank = 0
    device = 'cuda' if args.cuda else 'cpu'
    if args.ddp:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)

    # initialize logger
    log_file = os.path.join(model_dir, 'log_test.log')
    set_logger(log_file, args.ddp, rank)
    if rank == 0:
        logging.info(f'Testing {checkpoint_path} {progress}')
        logging.debug(f'Configuration: \n {pprint.pformat(cfg.to_dict())}')

    # initialize model
    model_cls = ModelRegistry.get(cfg.arch)
    model = model_cls(**cfg.model.to_dict())
    model = model.to(device)
    if args.ddp:
        model = DDP(model, device_ids=[device])

    # load checkpoint
    map_location = f'cuda:{device}' if isinstance(device, int) else device
    state = torch.load(checkpoint_path, map_location=map_location)
    get_model(model).load_state_dict(state['model'])
    if 'ema' in state.keys():
        ema = EMA(model.parameters(), decay=cfg.trainer.ema_decay)
        ema.load_state_dict(state['ema'])
        ema.copy_to()

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # test model
    for i, test_path in enumerate(args.tests):
        score_file = os.path.join(model_dir, 'scores.hdf5')
        test_model(i, model, cfg, test_path, score_file, checkpoint_path,
                   rank, device)
        if dist.is_initialized():
            dist.barrier()


def test_model(i_test, model, cfg, test_path, score_file, checkpoint_path,
               rank, device):
    # define score dataset path inside hdf5 score file
    checkpoint_name = os.path.basename(checkpoint_path)
    test_name = os.path.basename(os.path.normpath(test_path))
    h5path = f'{checkpoint_name}/{test_name}'

    # check if already tested
    progress = f'[{i_test}/{len(args.tests)}]'
    if os.path.exists(score_file):
        with h5py.File(score_file, 'r') as h5file:
            already_tested = h5path in h5file
        if already_tested and not args.force:
            if rank == 0:
                logging.info(f'Model already tested on {test_path} {progress}')
            return
    if rank == 0:
        logging.info(f'Evaluating on {test_path} {progress}')

    # initialize dataset
    dataset = BreverDataset(
        path=test_path,
        segment_length=0.0,
        fs=cfg.dataset.fs,
        sources=cfg.dataset.sources,
        transform=None,
    )

    # initialize sampler
    # use sorted batch sampler to minimize the amount of padding
    # TODO: the indexes in the hdf5 dataset do not match the mixture indexes!
    batch_sampler = SortedBatchSampler(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        dynamic=True,
        reverse=True,
    )
    if dist.is_initialized():
        batch_sampler = DistributedBatchSamplerWrapper(batch_sampler)

    # initialize dataloader
    dataloader = BreverDataLoader(
        dataset=dataset,
        num_workers=args.workers,
        batch_sampler=batch_sampler,
    )

    # initialize scores
    if dist.is_initialized():
        # calculate n_mix
        n_mix = 0
        for batch, lengths in dataloader:
            n_mix += batch.size(0)
    else:
        n_mix = len(dataset)
    dset_scores = np.empty((n_mix, len(args.metrics), 2))
    avg_delta_scores = {metric: 0 for metric in args.metrics}

    # main loop
    i_mix = 0
    with tqdm(dataloader, file=sys.stdout, position=rank) as t:
        for batch, lengths in t:
            batch, lengths = batch.to(device), lengths.to(device)
            input_, target = batch[:, 0], batch[:, 1:]
            output = get_model(model).enhance(
                input_, use_amp=cfg.trainer.use_amp
            )
            # TODO: if source separation network, find clean speech estimate
            # among the separated sources; take first source for now
            if output.ndim == 3:
                output = output[:, 0]
            # TODO: find clean speech among the target sources based on the
            # BreverDataset sources attribute; take first source for now
            target = target[:, 0]
            # average left and right channels
            input_ = input_.mean(-2)
            target = target.mean(-2)
            # calculate metrics
            batch_size = batch.shape[0]
            j_mix = i_mix + batch_size
            for i_metric, metric in enumerate(args.metrics):
                metric_func = MetricRegistry.get(metric)
                input_score = metric_func(input_, target, lengths=lengths)
                output_score = metric_func(output, target, lengths=lengths)
                dset_scores[i_mix:j_mix, i_metric, 0] = input_score.tolist()
                dset_scores[i_mix:j_mix, i_metric, 1] = output_score.tolist()
                delta = output_score - input_score
                avg_delta_scores[metric] = (
                    (delta.sum().item() + avg_delta_scores[metric]*i_mix)
                    / (i_mix + batch_size)
                )

            if args.output_dir is not None:
                for x, name in [(input_, 'input'), (output, 'output')]:
                    for i in range(batch_size):
                        filename = f'{i_mix + i:05d}_{name}.flac'
                        path = os.path.join(args.output_dir, filename)
                        torchaudio.save(
                            path, x[i].unsqueeze(0).cpu().float(),
                            cfg.dataset.fs
                        )

            i_mix = j_mix

            t.set_postfix(significant_figures(avg_delta_scores))

    if dist.is_initialized():
        # gather scores
        score_gather_list = [object() for _ in range(dist.get_world_size())]
        dist.gather_object(
            dset_scores,
            object_gather_list=score_gather_list if rank == 0 else None,
        )
        # gather average delta scores
        avg_gather_list = [object() for _ in range(dist.get_world_size())]
        dist.gather_object(
            avg_delta_scores,
            object_gather_list=avg_gather_list if rank == 0 else None,
        )

    if rank == 0:
        # concatenate scores
        if dist.is_initialized():
            dset_scores = np.concatenate(score_gather_list, axis=0)
            avg_delta_scores = {
                metric: sum(
                    avg[metric] for avg in avg_gather_list
                ) / dist.get_world_size()
                for metric in args.metrics
            }

        # log scores
        logging.info('Average delta scores:')
        for metric, delta in avg_delta_scores.items():
            logging.info(f'{metric}: {delta:.2e}')

        # update scores file
        if os.path.exists(score_file):
            h5file = h5py.File(score_file, 'a')
            if h5path in h5file.keys():
                h5dset = h5file[h5path]
                h5dset[...] = dset_scores
            else:
                h5dset = h5file.create_dataset(h5path, data=dset_scores)
        else:
            h5file = h5py.File(score_file, 'w')
            h5file['metrics'] = [metric for metric in args.metrics]
            h5file['which'] = ['input', 'output']
            h5dset = h5file.create_dataset(h5path, data=dset_scores)
        h5dset.dims[0].label = 'mixture'
        h5dset.dims[1].label = 'metric'
        h5dset.dims[2].label = 'which'
        h5dset.dims[1].attach_scale(h5file['metrics'])
        h5dset.dims[2].attach_scale(h5file['which'])
        h5file.close()


def find_best_checkpoint(model_dir, metric):
    regex = rf'^.*?_{metric}=(\d+\.\d+(?:e(?:\+|-)\d+)?).*?\.ckpt$'
    matches = [re.match(regex, filename) for filename in os.listdir(model_dir)]
    ckpts_and_scores = [
        (
            os.path.join(model_dir, match.group(0)),
            float(match.group(1)),
        )
        for match in matches if match
    ]
    best_ckpt = max(ckpts_and_scores, key=lambda x: x[1])[0]
    return best_ckpt


def get_model(model):
    if isinstance(model, DDP):
        model = model.module
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test a model')
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='model directories or checkpoints')
    parser.add_argument('-t', '--tests', type=Path, nargs='+', required=True,
                        help='test dataset paths')
    parser.add_argument('-f', '--force', action='store_true',
                        help='test even if already tested')
    parser.add_argument('--output_dir',
                        help='where to write signals')
    parser.add_argument('--cuda', action='store_true',
                        help='run on GPU')
    parser.add_argument('--metrics', nargs='+',
                        default=['pesq', 'stoi', 'estoi', 'snr', 'sisnr'],
                        help='metrics to evaluate with')
    parser.add_argument('--no_train_check', action='store_true',
                        help='test even if model is not trained')
    parser.add_argument('--best',
                        help='metric to use for checkpoint selection')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--ddp', action='store_true',
                        help='use DDP')
    args = parser.parse_args()

    if args.output_dir is not None and args.ddp:
        raise ValueError('cannot use DDP with output_dir')

    for i, input_ in enumerate(args.inputs):
        main(i, input_)

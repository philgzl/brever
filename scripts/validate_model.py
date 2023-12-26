import argparse
import logging
import os
import pprint
import sys

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from brever.batching import SortedBatchSampler
from brever.config import get_config
from brever.data import BreverDataLoader, BreverDataset
from brever.logger import set_logger
from brever.models import ModelRegistry


def main(input):
    # check if model exists
    if not os.path.exists(input):
        print(f'Model {input} does not exist')
        return

    checkpoints_dir = os.path.join(input, 'checkpoints')
    if input.endswith('.ckpt'):
        model_dir = os.path.dirname(os.path.dirname(input))
        checkpoint_path = input
    else:
        model_dir = input
        checkpoint_path = os.path.join(checkpoints_dir, 'last.ckpt')

    # check if model is trained
    loss_path = os.path.join(model_dir, 'losses.npz')
    if not os.path.exists(loss_path) and not args.no_train_check:
        print(f'Model {input} is not trained')
        return

    # check if already validated
    val_file = os.path.join(model_dir, 'val.npz')
    npz_key = os.path.basename(checkpoint_path)
    if os.path.exists(val_file):
        npz_obj = np.load(val_file)
        already_validated = npz_key in npz_obj.keys()
        if already_validated and not args.force:
            logging.info('Model already validated')
            return

    # load model config
    cfg_path = os.path.join(model_dir, 'config.yaml')
    cfg = get_config(cfg_path)

    # initialize logger
    log_file = os.path.join(model_dir, 'log_val.log')
    set_logger(log_file)
    logging.info(f'Validating {checkpoint_path}')
    logging.debug(f'Configuration: \n {pprint.pformat(cfg.to_dict())}')

    # initialize model
    model_cls = ModelRegistry.get(cfg.arch)
    model = model_cls(**cfg.model.to_dict())
    model = model.to('cuda')

    # load checkpoint
    state = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(state['model'])
    if 'ema' in state.keys():
        ema = EMA(model.parameters(), decay=cfg.trainer.ema_decay)
        ema.load_state_dict(state['ema'])
        ema.copy_to()

    # disable gradients and set model to eval
    torch.set_grad_enabled(False)
    model.eval()

    # initialize dataset
    dataset = BreverDataset(
        path=cfg.val_path,
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

    # initialize dataloader
    dataloader = BreverDataLoader(
        dataset=dataset,
        num_workers=args.workers,
        batch_sampler=batch_sampler,
    )

    # initialize scores
    n_mix = len(dataset)
    scores = np.empty(n_mix)

    # main loop
    i_mix = 0
    with tqdm(dataloader, file=sys.stdout) as t:
        for batch, lengths in t:
            batch, lengths = batch.to('cuda'), lengths.to('cuda')

            # the validation dataset yields raw waveforms
            # manually apply the model transform and recreate the batch to
            # calculate the validation loss
            transformed, trans_lengths = BreverDataLoader._collate_fn([
                model.transform(x[..., :l]) for x, l in zip(batch, lengths)
            ])
            loss = model.val_step(
                transformed, trans_lengths, cfg.trainer.use_amp
            )

            batch_size = batch.shape[0]
            j_mix = i_mix + batch_size
            scores[i_mix:j_mix] = loss.tolist()
            i_mix = j_mix

    avg_score = scores.mean()

    # update scores file
    if os.path.exists(val_file):
        npz_obj = dict(np.load(val_file))
    else:
        npz_obj = {}
    npz_obj[npz_key] = avg_score
    np.savez(val_file, **npz_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validate a model')
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='model directories or checkpoints')
    parser.add_argument('-f', '--force', action='store_true',
                        help='validate even if already validated')
    parser.add_argument('--output_dir',
                        help='where to write signals')
    parser.add_argument('--no_train_check', action='store_true',
                        help='validate even if model is not trained')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers')
    args = parser.parse_args()

    for input_ in args.inputs:
        main(input_)

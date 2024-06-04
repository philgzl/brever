import argparse
import logging
import os
import pprint
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb
from dotenv import load_dotenv

from brever.args import ModelArgParser
from brever.config import get_config
from brever.data import BreverDataset
from brever.logger import set_logger
from brever.models import ModelRegistry
from brever.training import BreverTrainer


def main():
    # check if already trained
    loss_path = os.path.join(args.input, 'losses.npz')
    if os.path.exists(loss_path) and not args.force:
        if args.force:
            os.remove(loss_path)
        else:
            raise FileExistsError(f'training already done: {loss_path}')

    # load model config
    cfg_path = os.path.join(args.input, 'config.yaml')
    cfg = get_config(cfg_path)

    # supersede config options provided by user
    cfg.update_from_args(args, ModelArgParser.trainer_arg_map())

    # init distributed group
    trainer_kwargs = cfg.trainer.to_dict()
    rank = trainer_kwargs.pop('rank')
    device = trainer_kwargs.pop('device')
    if cfg.trainer.ddp:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()

    # initialize logger
    log_file = os.path.join(args.input, 'log_train.log')
    set_logger(log_file, cfg.trainer.ddp, rank)
    if rank == 0:
        logging.info(f'Training {args.input}')
        logging.info(f'Configuration: \n {pprint.pformat(cfg.to_dict())}')

    # initialize wandb
    if cfg.trainer.use_wandb and rank == 0:
        load_dotenv()
        missing_vars = [var for var in ['WANDB_ENTITY', 'WANDB_PROJECT']
                        if var not in os.environ]
        if missing_vars:
            logging.warning(f'{" and ".join(missing_vars)} environment '
                            f'variable{"s" if len(missing_vars) > 1 else ""} '
                            'not set. Using wandb defaults.')
        configured = wandb.login(timeout=0)
        if not configured:
            raise ValueError('Could not login to wandb. This can be solved by '
                             'calling `wandb login` or by setting the '
                             'WANDB_API_KEY environment variable in a .env '
                             'file in the project root directory.')
        wandb.init(
            config=cfg.to_json(),
            name=os.path.basename(os.path.normpath(args.input)),
            dir=args.input,
            id=args.wandb_run_id,
            resume=args.wandb_run_id is not None,
        )

    # seed for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # initialize model
    logging.debug('Initializing model')
    model_cls = ModelRegistry.get(cfg.arch)
    model = model_cls(**cfg.model.to_dict())

    # initialize datasets
    logging.debug('Initializing datasets')
    if rank == 0:
        check_datasets(cfg.train_path, cfg.val_path)
    max_segment_length = cfg.dataset.max_segment_length
    if cfg.trainer.dynamic_batch_size and max_segment_length == 0:
        logging.debug('Setting max_segment_length to dynamic_batch_size to '
                      'ensure a batch can fit at least one segment.')
        max_segment_length = float(cfg.trainer.batch_size)
    train_dataset = BreverDataset(
        path=cfg.train_path,
        segment_length=cfg.dataset.segment_length,
        overlap_length=cfg.dataset.overlap_length,
        fs=cfg.dataset.fs,
        sources=cfg.dataset.sources,
        segment_strategy=cfg.dataset.segment_strategy,
        max_segment_length=max_segment_length,
        tar=cfg.dataset.tar,
        transform=model.transform,
        dynamic_mixing=cfg.dataset.dynamic_mixing,
        dynamic_mixtures_per_epoch=cfg.dataset.dynamic_mixtures_per_epoch,
    )
    val_dataset = BreverDataset(
        path=cfg.val_path,
        segment_length=0.0,
        overlap_length=0.0,
        fs=cfg.dataset.fs,
        sources=cfg.dataset.sources,
        segment_strategy='pass',
        max_segment_length=max_segment_length,
        tar=cfg.dataset.tar,
        transform=None,
        dynamic_mixing=False,
    )

    # initialize trainer
    logging.debug('Initializing trainer')
    ignore_checkpoint = trainer_kwargs.pop('ignore_checkpoint')
    trainer = BreverTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_dirpath=args.input,
        device=device,
        rank=rank,
        ignore_checkpoint=ignore_checkpoint or args.force,
        **trainer_kwargs,
    )

    # run
    logging.debug('Running trainer')
    trainer.run()


def check_datasets(train_path, val_path):
    train_cfg_path = os.path.join(train_path, 'config.yaml')
    val_cfg_path = os.path.join(val_path, 'config.yaml')
    if not os.path.exists(train_cfg_path) or not os.path.exists(val_cfg_path):
        logging.warning(f'Could not find {train_cfg_path} or {val_cfg_path}. '
                        'Skipping dataset check.')
        return
    train_cfg = get_config(train_cfg_path)
    val_cfg = get_config(val_cfg_path)
    if (
        train_cfg.rmm.seed == val_cfg.rmm.seed
        and train_cfg.rmm.speakers == val_cfg.rmm.speakers
        and train_cfg.rmm.noises == val_cfg.rmm.noises
        and train_cfg.rmm.rooms == val_cfg.rmm.rooms
        and train_cfg.rmm.speech_files == val_cfg.rmm.speech_files
        and train_cfg.rmm.noise_files == val_cfg.rmm.noise_files
        and train_cfg.rmm.room_files == val_cfg.rmm.room_files
    ):
        logging.warning(
            'Training and validation datasets have the same seed and the same '
            'same speech, noise and room files. They might be the same or too '
            'similar for the validation to be meaningful.'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model',
                                     conflict_handler='resolve')
    parser.add_argument('input', help='model directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='train even if already trained')
    parser.add_argument('--wandb_run_id',
                        help='id of wandb run to resume')

    group = parser.add_argument_group('the following options supersede the '
                                      'config file')

    ModelArgParser.add_dataset_args(group, new_group=False)
    ModelArgParser.add_trainer_args(group, new_group=False)
    ModelArgParser.add_extra_args(group, new_group=False)
    args = parser.parse_args()
    main()

import itertools
import logging
import operator
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from .batching import BatchSamplerRegistry, DistributedBatchSamplerWrapper
from .data import BreverDataLoader, BreverDataset
from .inspect import NoParse, Parse
from .metrics import MetricRegistry
from .models import count_params
from .models.base import BreverBaseModel


class BreverTrainer:
    def __init__(
        self,
        model: NoParse[BreverBaseModel],
        train_dataset: NoParse[BreverDataset],
        val_dataset: NoParse[BreverDataset],
        model_dirpath: NoParse[str],
        workers: int = 0,
        epochs: int = 100,
        device: int | Parse[str] = 'cuda',
        batch_sampler: str = 'bucket',
        batch_size: int = 32,
        num_buckets: int = 10,
        dynamic_batch_size: bool = True,
        fs: int = 16000,
        ema: bool = False,
        ema_decay: float = 0.999,
        ignore_checkpoint: bool = False,
        preload: bool = False,
        ddp: bool = False,
        rank: int = 0,
        use_wandb: bool = False,
        profile: bool = False,
        val_metrics: set[str] = {'pesq', 'estoi', 'snr'},
        val_period: int = 10,
        use_amp: bool = False,
        compile: bool = False,
        save_on_epochs: list[int] = [],
    ):
        # set workers to 0 if preloading
        if preload and workers > 0:
            logging.warning('Cannot use workers > 0 with preload=True. '
                            'Forcing workers=0.')
            workers = 0

        # move model to device, cast to DDP and compile
        model = model.to(device)
        if ddp:
            model = DDP(model, device_ids=[device])
        if compile:
            torch._logging.set_logs(dynamo=logging.INFO)
            model.compile(dynamic=True)

        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_dirpath = model_dirpath
        self.epochs = epochs
        self.device = device
        self.ignore_checkpoint = ignore_checkpoint
        self.preload = preload
        self.rank = rank
        self.use_wandb = use_wandb
        self.profile = profile
        self.val_metrics = val_metrics
        self.val_period = val_period
        self.save_on_epochs = save_on_epochs

        self.checkpoints_dir = os.path.join(model_dirpath, 'checkpoints')
        self.last_ckpt_path = os.path.join(self.checkpoints_dir, 'last.ckpt')
        self.epochs_ran = 0
        self.max_memory_allocated = 0
        self._profiler = None

        # batch samplers
        train_batch_sampler_cls = BatchSamplerRegistry.get(batch_sampler)
        train_batch_sampler_kwargs = dict(
            batch_size=batch_size,
            dynamic=dynamic_batch_size,
            fs=fs,
        )
        if train_batch_sampler_cls == 'bucket':
            train_batch_sampler_kwargs['num_buckets'] = num_buckets
        self.train_batch_sampler = train_batch_sampler_cls(
            dataset=train_dataset,
            **train_batch_sampler_kwargs,
        )
        # for the validation batch sampler use the sorted batch sampler to
        # minimize padding and dynamic=True to prevent memory overflow
        val_batch_sampler_cls = BatchSamplerRegistry.get('sorted')
        # infer validation dynamic batch size
        if dynamic_batch_size:
            val_batch_size = batch_size
        else:
            max_segment_length = train_dataset.get_max_segment_length()
            val_batch_size = batch_size*max_segment_length/fs
        self.val_batch_sampler = val_batch_sampler_cls(
            dataset=val_dataset,
            batch_size=val_batch_size,
            dynamic=True,
            fs=fs
        )

        # distributed samplers
        if dist.is_initialized():
            self.train_batch_sampler = DistributedBatchSamplerWrapper(
                self.train_batch_sampler
            )
            self.val_batch_sampler = DistributedBatchSamplerWrapper(
                self.val_batch_sampler
            )

        # dataloaders
        self.train_dataloader = BreverDataLoader(
            dataset=train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=workers,
        )
        self.val_dataloader = BreverDataLoader(
            dataset=val_dataset,
            batch_sampler=self.val_batch_sampler,
            num_workers=workers,
        )

        # exponential moving average
        if ema:
            self.ema = ExponentialMovingAverage(
                model.parameters(),
                decay=ema_decay
            )
        else:
            self.ema = None

        # loss logger
        self.loss_logger = LossLogger(model_dirpath)

        # checkpoint saver
        self.checkpoint_saver = CheckpointSaver(
            dirpath=self.checkpoints_dir,
            save_func=self.save_checkpoint,
        )

        # timer
        self.timer = TrainingTimer(epochs, val_period)

        # automatic mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def optimizers(self):
        optimizers = self.get_model().optimizers()
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizers = [optimizers]
        elif not isinstance(optimizers, (tuple, list)):
            raise ValueError(f'the model `optimizers` method must return a '
                             f'{torch.optim.Optimizer.__name__} '
                             f'or a sequence, got '
                             f'{optimizers.__class__.__name__}')
        return optimizers

    def run(self):

        def _rank_zero_log(msg):
            if self.rank == 0:
                logging.info(msg)

        # print number of parameters and dataset duration
        num_params = f'{round(count_params(self.model))/1e6:.2f} M'
        _rank_zero_log(f'Number of parameters: {num_params}')
        for dset, dset_name in [
            (self.train_dataset, 'Training dataset'),
            (self.val_dataset, 'Validation dataset'),
        ]:
            for duration, duration_name in [
                (dset._duration, 'duration'),
                (dset._effective_duration, 'effective duration'),
            ]:
                if duration == float('inf'):
                    fmt_time = 'inf'
                else:
                    h, m = divmod(int(duration), 3600)
                    m, s = divmod(m, 60)
                    fmt_time = f'{h} h {m} m {s} s'
                _rank_zero_log(f'{dset_name} {duration_name}: {fmt_time}')

        # check for a checkpoint
        checkpoint_loaded = False
        if not self.ignore_checkpoint and os.path.exists(self.last_ckpt_path):
            _rank_zero_log('Checkpoint found')
            self.load_checkpoint()
            # if training was interrupted then resume training
            if self.epochs_ran < self.epochs:
                _rank_zero_log(f'Resuming training at epoch {self.epochs_ran}')
            else:
                _rank_zero_log('Model is already trained')
                return
            checkpoint_loaded = True

        # preload data
        if self.preload:
            _rank_zero_log('Preloading data')
            self.train_dataset.preload(self.device, tqdm_desc='train')
            self.val_dataset.preload(self.device, tqdm_desc='  val')

        # pre-training instructions
        if not checkpoint_loaded:
            _rank_zero_log('Pre-training model instructions')
            self.get_model().pre_train(
                self.train_dataset, self.train_dataloader, self.epochs
            )

        # start profiler
        if self.profile:

            def trace_handler(profiler):
                _rank_zero_log('\n' + profiler.key_averages().table(
                    sort_by='self_cuda_time_total', row_limit=-1)
                )

            _rank_zero_log('Starting profiler')
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1,
                ),
                on_trace_ready=trace_handler
            )
            self._profiler.start()

        # start training loop
        try:
            _rank_zero_log('Starting training loop')
            self.training_loop()
        except Exception:
            raise
        finally:
            if self._profiler is not None:
                self._profiler.stop()

    def training_loop(self):
        if self.rank == 0:
            self.timer.start()
        for epoch in range(self.epochs_ran, self.epochs):
            self.train_dataloader.set_epoch(epoch)
            self.val_dataloader.set_epoch(epoch)
            # train
            train_loss = self.routine(epoch, train=True)
            # evaluate
            validate = self.val_period != 0 and epoch % self.val_period == 0
            if validate:
                with torch.no_grad():
                    val_loss, val_metrics = self.routine(epoch, train=False)
                self.get_model().on_validate(
                    val_loss if len(val_loss) > 1
                    else next(iter(val_loss.values()))
                )
            else:
                val_loss, val_metrics = {}, {}
            # ddp reduce
            if dist.is_initialized():
                self.reduce(train_loss, val_loss, val_metrics)
            # log and save best model
            self.epochs_ran += 1
            if self.rank == 0:
                self.loss_logger.add(train_loss, val_loss, val_metrics)
                self.loss_logger.log(epoch)
                if self.use_wandb:
                    self.wandb_log(train_loss, val_loss, val_metrics)
                # save best model
                self.checkpoint_saver(epoch, val_loss, val_metrics)
                # save last model
                self.save_checkpoint()
                # save additional checkpoints
                if epoch in self.save_on_epochs:
                    self.save_checkpoint(os.path.join(
                        self.checkpoints_dir, f'epoch={epoch}.ckpt'
                    ))
            # ddp sync
            if dist.is_initialized():
                dist.barrier()
        # plot and save losses
        if self.rank == 0:
            self.timer.final_log()
            self.loss_logger.plot_and_save()

    def routine(self, epoch, train=True):
        if train:
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.val_dataloader
            if self.ema is not None:
                self.ema.store()
                self.ema.copy_to()
            avg_metrics = MathDict()
        avg_loss = MathDict()
        tqdm_desc = f'epoch {epoch}, ' + ('train' if train else '  val') \
            + (f', rank {self.rank}' if dist.is_initialized() else '')
        for batch, lengths in tqdm(dataloader, file=sys.stdout,
                                   desc=tqdm_desc, position=self.rank):
            if isinstance(batch, list):
                batch = [x.to(self.device) for x in batch]
            else:
                batch = batch.to(self.device)
            lengths = lengths.to(self.device)
            model = self.get_model()
            use_amp = self.scaler.is_enabled()
            if train:
                loss = model.train_step(batch, lengths, use_amp, self.scaler)
                if self.ema is not None:
                    self.ema.update()
            else:
                # the validation dataset yields raw waveforms
                # manually apply the model transform and recreate the batch to
                # calculate the validation loss
                transformed, trans_lengths = BreverDataLoader._collate_fn([
                    model.transform(x[..., :l]) for x, l in zip(batch, lengths)
                ])
                loss = model.val_step(transformed, trans_lengths, use_amp)
                # finally compute metrics
                metrics = self.compute_metrics(batch, lengths, use_amp)
                avg_metrics += metrics
            if isinstance(loss, torch.Tensor):
                loss = {'loss': loss}
            elif not isinstance(loss, dict):
                raise ValueError(f'train_step and val_step must return a '
                                 f'tensor or a dict, got '
                                 f'{loss.__class__.__name__}')
            loss = {k: v.detach() for k, v in loss.items()}
            avg_loss += loss
            if self._profiler is not None:
                self._profiler.step()
        avg_loss /= len(dataloader)
        if train:
            output = avg_loss
        else:
            if self.ema is not None:
                self.ema.restore()
            avg_metrics /= len(dataloader)
            output = avg_loss, avg_metrics
        if dist.is_initialized():
            dist.barrier()
        # update time spent
        if self.rank == 0:
            self.timer.step(is_validation_step=not train)
            self.timer.log()
        return output

    def reduce(self, *tensor_dicts):
        for tensor_dict in tensor_dicts:
            for key, tensor in tensor_dict.items():
                dist.reduce(tensor, 0)
                tensor /= dist.get_world_size()

    def compute_metrics(self, batch, lengths, use_amp):
        # batch has shape (batch_size, sources, channels, samples)
        if not self.val_metrics:
            return {}
        input_, target = batch[:, 0], batch[:, 1:]
        output = self.get_model().enhance(input_, use_amp=use_amp)
        # TODO: if source separation network, find clean speech estimate
        # among the separated sources; take first source for now
        if output.ndim == 3:
            output = output[:, 0]
        # TODO: find clean speech among the target sources based on the
        # BreverDataset sources attribute; take first source for now
        target = target[:, 0]
        # average left and right channels
        target = target.mean(-2)
        # calculate metrics
        metrics = {}
        for metric_name in self.val_metrics:
            metric = MetricRegistry.get(metric_name)
            metric_values = metric(output, target, lengths=lengths)
            metrics[metric_name] = metric_values.mean()
        return metrics

    def wandb_log(self, train_loss, val_loss, val_metrics):
        wandb.log({
            'train': train_loss,
            'val': {
                **val_loss,
                'metrics': val_metrics,
            }
        })

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.last_ckpt_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.rank == 0:
            state = {
                'epochs': self.epochs_ran,
                'model': self.get_model().state_dict(),
                'optimizers': [optimizer.state_dict()
                               for optimizer in self.optimizers()],
                'scaler': self.scaler.state_dict(),
                'losses': {
                    'train': self.loss_logger.train_loss,
                    'val': self.loss_logger.val_loss,
                },
                'max_memory_allocated': max(
                    torch.cuda.max_memory_allocated(),
                    self.max_memory_allocated,
                ),
                'timer': self.timer.state_dict(),
                'best_ckpts': self.checkpoint_saver.best,
            }
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
            torch.save(state, path)

    def load_checkpoint(self):
        if isinstance(self.device, int):
            map_location = f'cuda:{self.device}'
        else:
            map_location = self.device
        state = torch.load(self.last_ckpt_path, map_location=map_location)
        self.get_model().load_state_dict(state['model'])
        for optimizer, substate in zip(self.optimizers(), state['optimizers']):
            optimizer.load_state_dict(substate)
        self.scaler.load_state_dict(state['scaler'])
        self.loss_logger.train_loss = state['losses']['train']
        self.loss_logger.val_loss = state['losses']['val']
        self.epochs_ran = state['epochs']
        self.max_memory_allocated = state['max_memory_allocated']
        self.timer.load_state_dict(state['timer'])
        self.checkpoint_saver.best = state['best_ckpts']
        if self.ema is not None:
            if 'ema' in state.keys():
                self.ema.load_state_dict(state['ema'])
            else:
                raise ValueError('exponential moving average state not found '
                                 'in state dict')

    def get_model(self):
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        return model


class TrainingTimer:
    def __init__(self, epochs, val_period):
        self.epochs = epochs
        self.val_period = val_period
        self.train_steps_taken = 0
        self.val_steps_taken = 0
        self.train_steps_measured = 0
        self.val_steps_measured = 0
        self.avg_train_duration = None
        self.avg_val_duration = None
        self.start_time = None
        self.step_start_time = None
        self.resume_offset = 0
        self.first_session_step = True

        if val_period == 0:
            self.avg_val_duration = 0

    def load_state_dict(self, state):
        self.train_steps_taken = state['train_steps_taken']
        self.val_steps_taken = state['val_steps_taken']
        self.train_steps_measured = state['train_steps_measured']
        self.val_steps_measured = state['val_steps_measured']
        self.avg_train_duration = state['avg_train_duration']
        self.avg_val_duration = state['avg_val_duration']
        self.resume_offset = state['resume_offset']

    def state_dict(self):
        return dict(
            train_steps_taken=self.train_steps_taken,
            val_steps_taken=self.val_steps_taken,
            train_steps_measured=self.train_steps_measured,
            val_steps_measured=self.val_steps_measured,
            avg_train_duration=self.avg_train_duration,
            avg_val_duration=self.avg_val_duration,
            resume_offset=self.total_elapsed_time,
        )

    def start(self):
        start_time = time.time()
        self.start_time = start_time
        self.step_start_time = start_time

    def step(self, is_validation_step=False):
        step_end_time = time.time()
        step_duration = step_end_time - self.step_start_time
        if is_validation_step:
            if not self.first_session_step:
                self.update_avg_val_duration(step_duration)
            self.val_steps_taken += 1
        else:
            if not self.first_session_step:
                self.update_avg_train_duration(step_duration)
            self.train_steps_taken += 1
        self.first_session_step = False
        self.step_start_time = step_end_time

    def update_avg_train_duration(self, duration):
        if self.avg_train_duration is None:
            self.avg_train_duration = duration
        else:
            self.avg_train_duration = (
                self.avg_train_duration*self.train_steps_measured + duration
            ) / (self.train_steps_measured + 1)
        self.train_steps_measured += 1

    def update_avg_val_duration(self, duration):
        if self.avg_val_duration is None:
            self.avg_val_duration = duration
        else:
            self.avg_val_duration = (
                self.avg_val_duration*self.val_steps_measured + duration
            ) / (self.val_steps_measured + 1)
        self.val_steps_measured += 1

    @staticmethod
    def fmt_time(time):
        if time is None:
            return '--'
        h, m, s = int(time//3600), int((time % 3600)//60), int(time % 60)
        output = f'{s} s'
        if time >= 60:
            output = f'{m} m {output}'
        if time >= 3600:
            output = f'{h} h {output}'
        return output

    def log(self):
        logging.info(', '.join([
            f'Avg train time: {self.fmt_time(self.avg_train_duration)}',
            f'Avg val time: {self.fmt_time(self.avg_val_duration)}',
            f'ETA: {self.fmt_time(self.estimated_time_left)}',
        ]))

    def final_log(self):
        total_time = self.total_elapsed_time
        logging.info(
            f'Time spent: {int(total_time/3600)} h '
            f'{int(total_time % 3600 / 60)} m {int(total_time % 60)} s'
        )

    @property
    def total_elapsed_time(self):
        return self.session_elapsed_time + self.resume_offset

    @property
    def session_elapsed_time(self):
        return time.time() - self.start_time

    @property
    def train_steps(self):
        return self.epochs

    @property
    def val_steps(self):
        return 0 if self.val_period == 0 else self.epochs//self.val_period

    @property
    def train_steps_left(self):
        return self.train_steps - self.train_steps_taken

    @property
    def val_steps_left(self):
        return self.val_steps - self.val_steps_taken

    @property
    def estimated_time_left(self):
        if self.avg_train_duration is None or self.avg_val_duration is None:
            return None
        else:
            return self.avg_train_duration*self.train_steps_left \
                + self.avg_val_duration*self.val_steps_left


class LossLogger:
    def __init__(self, dirpath):
        self.train_loss = []
        self.val_loss = []
        self.val_metrics = []
        self.dirpath = dirpath

    def add(self, train_loss, val_loss, val_metrics):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.val_metrics.append(val_metrics)

    def log(self, epoch):
        last_train_loss = self.train_loss[-1]
        last_val_loss = self.val_loss[-1]
        last_val_metrics = self.val_metrics[-1]
        logging.info(
            f'Epoch {epoch}: ' + ', '.join(itertools.chain(
                (f'train_{k}: {v:.2e}' for k, v in last_train_loss.items()),
                (f'val_{k}: {v:.2e}' for k, v in last_val_loss.items()),
                (f'val_{k}: {v:.2e}' for k, v in last_val_metrics.items()),
            ))
        )

    def plot_and_save(self):
        losses = self._to_numpy_dict()
        self.plot(losses)
        self.save(losses)

    def _to_numpy_dict(self):
        # this is called only at the end of the training loop so calling
        # .item() should be fine here
        output = dict()
        for loss, tag in [
            (self.train_loss, 'train'),
            (self.val_loss, 'val'),
            (self.val_metrics, 'metrics')
        ]:
            for epoch, d in enumerate(loss):
                for k in d.keys():
                    out_key = f'{tag}_{k}'
                    if out_key not in output:
                        output[out_key] = []
                    output[out_key].append((epoch, d[k].item()))
        # convert to numpy array
        for k, v in output.items():
            output[k] = np.array(v)
        return output

    def plot(self, losses):
        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True)
        plt.rc('grid', color='w', linestyle='solid')
        fig, ax = plt.subplots()
        for k, v in losses.items():
            if not k.startswith('metrics'):
                ax.plot(v[:, 0], v[:, 1], label=k)
        ax.legend()
        ax.set_xlabel('epoch')
        ax.set_ylabel('error')
        ax.grid(True)
        plot_output_path = os.path.join(self.dirpath, 'training_curve.png')
        fig.tight_layout()
        fig.savefig(plot_output_path)
        plt.close(fig)

    def save(self, losses):
        loss_path = os.path.join(self.dirpath, 'losses.npz')
        np.savez(loss_path, **losses)


class CheckpointSaver:
    def __init__(self, dirpath, save_func):
        self.dirpath = dirpath
        self.save_func = save_func
        self.best = {}

    def __call__(self, epoch, loss, metrics):
        for d, op in [
            (loss, operator.lt),
            (metrics, operator.gt),
        ]:
            for name, val in d.items():
                first_time = name not in self.best
                if first_time or op(val, self.best[name]['val']):
                    filename = \
                        f'epoch={epoch}_{name}={self._fmt_float(val)}.ckpt'
                    filepath = os.path.join(self.dirpath, filename)
                    self.save_func(filepath)
                    logging.info(f'New best {name}, saving {filepath}')
                    if not first_time:
                        if os.path.exists(self.best[name]['filepath']):
                            os.remove(self.best[name]['filepath'])
                        else:
                            logging.warning(f'Previous best {name} checkpoint '
                                            f'{self.best[name]["filepath"]} '
                                            f'does not exist. Skipping '
                                            'removal.')
                    self.best[name] = {'val': val, 'filepath': filepath}

    @staticmethod
    def _fmt_float(x):
        return f'{x:.2e}' if abs(x) < 0.1 or abs(x) >= 100 else f'{x:.2f}'


class MathDict(dict):
    @staticmethod
    def __apply_op(input_, other, op, default):
        if isinstance(other, dict):
            for key, value in other.items():
                input_[key] = op(input_.get(key, default), value)
        elif isinstance(other, (int, float)):
            for key in input_.keys():
                input_[key] = op(input_[key], other)
        return input_

    def __add__(self, other):
        return self.__apply_op(MathDict(self), other, lambda x, y: x + y, 0)

    def __sub__(self, other):
        return self.__apply_op(MathDict(self), other, lambda x, y: x - y, 0)

    def __mul__(self, other):
        return self.__apply_op(MathDict(self), other, lambda x, y: x * y, 1)

    def __truediv__(self, other):
        return self.__apply_op(MathDict(self), other, lambda x, y: x / y, 1)

    def __iadd__(self, other):
        return self.__apply_op(self, other, lambda x, y: x + y, 0)

    def __isub__(self, other):
        return self.__apply_op(self, other, lambda x, y: x - y, 0)

    def __imul__(self, other):
        return self.__apply_op(self, other, lambda x, y: x * y, 1)

    def __itruediv__(self, other):
        return self.__apply_op(self, other, lambda x, y: x / y, 1)


class EarlyStopping:
    """Early stopping.

    This is deprecated and probably broken! Usage is not recommended! If ever
    fixed, should be able to track the validation metrics just like
    `CheckpointSaver`.
    """

    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.min_loss = np.inf

    def __call__(self, loss):
        score = -loss
        save, stop = False, False
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of '
                             f'{self.patience}')
            if self.counter >= self.patience:
                stop = True
        else:
            self.best_score = score
            if self.verbose:
                logging.info(f'Minimum validation loss decreased from '
                             f'{self.min_loss:.6f} to {loss:.6f}. '
                             f'Saving model.')
            self.min_loss = loss
            self.counter = 0
            save = True
        return save, stop

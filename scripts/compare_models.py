import argparse
import os

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

from brever.args import ModelArgParser
from brever.config import get_config
from brever.utils import pretty_table

plt.rcParams['patch.linewidth'] = .5
plt.rcParams['patch.force_edgecolor'] = True
plt.rcParams['grid.linestyle'] = ':'

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def fmt_time(time_):
    h, m, s = int(time_//3600), int((time_ % 3600)//60), int(time_ % 60)
    return f'{h:>2} h {m:>2} m {s:>2} s'


def fmt_memory(memory):
    memory = round(memory/1e9, 2)
    return f'{memory:>5} GB'


def fmt_score(score):
    return f"{score:.2e}"


def get_color(i):
    if args.colors is None:
        return COLORS[i % len(COLORS)]
    else:
        color = args.colors[i % len(args.colors)]
        try:
            return COLORS[int(color) % len(COLORS)]
        except ValueError:
            return color


def make_header(test_name, metric, i_test):
    if args.test_aliases is None:
        header = f'{test_name}-{metric}'
    else:
        header = f'{args.test_aliases[i_test]}-{metric}'
    if args.delta_scores:
        header = f'{header}_i'
    return header


def main():
    summary = {}
    labels = []
    _, ax_curve = plt.subplots()
    n_metrics = len(args.metrics)

    checkpoints = [
        x if x.endswith('.ckpt')
        else os.path.join(x, 'checkpoints', 'last.ckpt')
        for x in args.inputs
    ]

    if args.tests is not None:
        n_tests = len(args.tests)
        args.tests = [
            os.path.basename(os.path.normpath(x))
            for x in args.tests
        ]

        fig_test = plt.figure(figsize=args.figsize)

        if args.outer_gridspec is None:
            outer_gs = gridspec.GridSpec(1, n_tests, figure=fig_test)
        else:
            outer_gs = gridspec.GridSpec(*args.outer_gridspec, figure=fig_test)

        if args.outer_gridspec_slices is None:
            args.outer_gridspec_slices = list(range(n_tests))
        else:
            args.outer_gridspec_slices = [
                slice(*[int(y) for y in x.split(',')])
                for x in args.outer_gridspec_slices
            ]

        axes_test = np.empty((n_tests, n_metrics), dtype=object)
        ghost_axes = np.empty(n_tests, dtype=object)
        for i_test, test_name in enumerate(args.tests):
            slice_ = args.outer_gridspec_slices[i_test]
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, n_metrics, subplot_spec=outer_gs[slice_],
                height_ratios=[0.05, 0.95], hspace=0,
            )
            for i_metric, metric in enumerate(args.metrics):
                ax = fig_test.add_subplot(inner_gs[1, i_metric])
                if args.delta_scores:
                    ax.set_title(rf'$\Delta${metric}')
                else:
                    ax.set_title(metric)
                ax.set_xticks([])
                ax.grid()
                ax.set_axisbelow(True)
                axes_test[i_test, i_metric] = ax
            # add ghost axes for titles
            ghost_ax = fig_test.add_subplot(inner_gs[:])
            ghost_ax.axis('off')
            if args.test_aliases is None:
                ghost_ax.set_title(test_name)
            else:
                ghost_ax.set_title(args.test_aliases[i_test])
            ghost_axes[i_test] = ghost_ax

    for i, checkpoint in enumerate(checkpoints):
        model_dir = os.path.dirname(os.path.dirname(checkpoint))

        color = get_color(i)

        if args.legend is not None:
            label = args.legend[i]
            row_header = label
            summary[row_header] = {}
        elif args.legend_params is not None:
            config = get_config(os.path.join(model_dir, 'config.yaml'))
            arg_map = ModelArgParser.arg_map(config.arch)
            label = {}
            row_header = checkpoint
            summary[row_header] = {}
            for param_name in args.legend_params:
                param_val = config.get_field(arg_map[param_name])
                label[param_name] = param_val
                summary[row_header][param_name] = param_val
            label = [f'{key}: {val}' for key, val in label.items()]
            label = ', '.join(label)
        else:
            label = checkpoint
            label = label.removeprefix(get_config('config/paths.yaml').MODELS)
            label = label.removesuffix('.ckpt')
            label = label.replace('/checkpoints/', ':')
            row_header = label
            summary[row_header] = {}
        labels.append(label)

        loss_path = os.path.join(model_dir, 'losses.npz')
        if os.path.exists(loss_path):
            loss_data = np.load(loss_path)
        else:
            print(f"WARNING: {loss_path} not found, skipping")
            loss_data = None

        if loss_data is not None:
            train_loss = loss_data['train_loss']
            ax_curve.plot(train_loss[:, 0], train_loss[:, 1], label=label,
                          color=color)
            if 'val_loss' in loss_data:
                val_loss = loss_data['val_loss']
                ax_curve.plot(val_loss[:, 0], val_loss[:, 1], '--',
                              color=color)

        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location='cpu')
            summary[row_header]['training_time'] = fmt_time(
                state['timer']['resume_offset']
            )
            summary[row_header]['gpu'] = fmt_memory(
                state['max_memory_allocated']
            )
        else:
            print(f"WARNING: {checkpoint} not found, skipping")
            summary[row_header]['training_time'] = ''
            summary[row_header]['gpu'] = ''
        if loss_data is not None:
            if 'val_loss' in loss_data:
                summary[row_header]['min_val_loss'] = \
                    fmt_score(min(loss_data['val_loss'][:, 1]))
            else:
                print(f"WARNING: no val_loss in {loss_path}, skipping")
                summary[row_header]['min_val_loss'] = ''
        else:
            summary[row_header]['min_val_loss'] = ''

        if args.tests is not None:
            score_file = os.path.join(model_dir, 'scores.hdf5')
            if not os.path.exists(score_file):
                print(f"WARNING: {score_file} not found, skipping")
            else:
                h5f = h5py.File(score_file)
                for i_test, test_name in enumerate(args.tests):
                    h5path = f'{os.path.basename(checkpoint)}/{test_name}'
                    if h5path not in h5f:
                        print(f"WARNING: {h5path} not found in "
                              f"{score_file}, skipping")
                    else:
                        for i_metric, metric in enumerate(args.metrics):
                            k = list(h5f['metrics'].asstr()).index(metric)
                            scores = h5f[h5path][:, k, 1]
                            if args.delta_scores:
                                scores -= h5f[h5path][:, k, 0]
                            mean = np.mean(scores)
                            if args.yerr == 'std':
                                yerr = np.std(scores)
                            elif args.yerr == 'sem':
                                yerr = np.std(scores)/np.sqrt(len(scores))
                            else:
                                raise ValueError('yerr must be std or sem, '
                                                 f'got {args.yerr}')
                            col_header = make_header(test_name, metric, i_test)
                            summary[row_header][col_header] = fmt_score(mean)
                            ax = axes_test[i_test, i_metric]
                            ax.bar([i], [mean], width=1, label=label,
                                   yerr=[yerr], color=color)
                h5f.close()

            for test_name in args.tests:
                for metric in args.metrics:
                    col_header = make_header(test_name, metric, i_test)
                    if col_header not in summary[row_header]:
                        summary[row_header][col_header] = ''

    if not args.no_legend:
        lines = [
            Line2D([], [], color='k', linestyle='-'),
            Line2D([], [], color='k', linestyle='--'),
        ]
        lh = ax_curve.legend(loc=1)
        ax_curve.legend(lines, ['train', 'val'], loc=2, ncol=args.legend_ncol)
        ax_curve.add_artist(lh)

    ax_curve.set_ylim(args.ymin, args.ymax)
    ax_curve.grid()

    pretty_table(summary, order_by=args.order_by, reverse=True)

    if args.tests is not None:
        label_handle_map = {}
        for i_test in range(n_tests):
            handles = list(filter(
                lambda x: isinstance(x, BarContainer),
                axes_test[i_test, 0].containers
            ))
            for handle in handles:
                label = handle.get_label()
                if label not in label_handle_map:
                    label_handle_map[label] = handle
        label_handle_map = {
            label: label_handle_map[label] for label in labels
            if label in label_handle_map
        }
        labels, handles = zip(*label_handle_map.items())
        if args.legend_loc is None:
            loc = args.legend_loc
        elif len(args.legend_loc) == 1:
            loc, = args.legend_loc
        else:
            loc = [float(x) for x in args.legend_loc]
        fig_test.legend(handles, labels, loc=loc, ncol=args.legend_ncol,
                        bbox_to_anchor=args.legend_bbox_to_anchor)
        fig_test.tight_layout(rect=args.rect)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compare models')
    parser.add_argument('-i', '--inputs', nargs='+', required=True,
                        help='paths to model directories or checkpoints')
    parser.add_argument('-t', '--tests', nargs='+',
                        help='test dataset paths or basenames')
    parser.add_argument('--legend_params', nargs='+',
                        help='hyperparameters to label models with')
    parser.add_argument('--legend', nargs='+',
                        help='hard set model labels in legend')
    parser.add_argument('--no_legend', action='store_true',
                        help='disable legend')
    parser.add_argument('--legend_ncol', type=int, default=1,
                        help='number of legend columns')
    parser.add_argument('--legend_bbox_to_anchor', type=float, nargs='+',
                        help='legend bbox_to_anchor option')
    parser.add_argument('--legend_loc', nargs='+',
                        help='legend location')
    parser.add_argument('--metrics', nargs='+',
                        default=['pesq', 'estoi', 'snr', 'sisnr'],
                        help='objective metrics to plot')
    parser.add_argument('--delta_scores', action='store_true',
                        help='plot objective metric improvements')
    parser.add_argument('--order_by',
                        help='column to order by in pretty table')
    parser.add_argument('--yerr', choices=['std', 'sem'], default='sem',
                        help='error bars as std or sem across files')
    parser.add_argument('--test_aliases', nargs='+',
                        help='test dataset labels')
    parser.add_argument('--rect', type=float, nargs=4,
                        help='tight_layout rect option')
    parser.add_argument('--colors', nargs='+',
                        help='model colors')
    parser.add_argument('--outer_gridspec', type=int, nargs=2,
                        help='outer gridspec rows and columns')
    parser.add_argument('--outer_gridspec_slices', nargs='+',
                        help='outer gridspec slices')
    parser.add_argument('--figsize', type=int, nargs=2,
                        help='figure size')
    parser.add_argument('--ymin', type=float,
                        help='training curve plot y-axis min value')
    parser.add_argument('--ymax', type=float,
                        help='training curve plot y-axis max value')
    parser.add_argument('--no_show', action='store_true',
                        help='do not show plots')
    args = parser.parse_args()
    main()

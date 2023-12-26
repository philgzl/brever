import argparse
import time

from tqdm import tqdm

from brever.batching import BatchSamplerRegistry
from brever.data import BreverDataLoader, BreverDataset


def main():
    print('Initializing dataset')
    kwargs = {}
    if args.sources is not None:
        kwargs['sources'] = args.sources
    if args.dynamic:
        kwargs['max_segment_length'] = args.batch_size
    dataset = BreverDataset(
        path=args.input,
        segment_length=args.segment_length,
        fs=args.fs,
        **kwargs,
    )

    print('Initializing batch sampler')
    batch_sampler_cls = BatchSamplerRegistry.get(args.sampler)
    batch_sampler_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        dynamic=args.dynamic,
        fs=args.fs,
    )
    if args.sampler == 'bucket':
        batch_sampler_kwargs['num_buckets'] = args.num_buckets
    batch_sampler = batch_sampler_cls(**batch_sampler_kwargs)

    print('Initializing data loader')
    dataloader = BreverDataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
    )

    print('Starting benchmark')
    elapsed_time = 0
    start_time = time.time()
    for i in range(args.epochs):
        dataloader.set_epoch(i)
        for sources, lengths in tqdm(dataloader):
            pass
        dt = time.time() - start_time - elapsed_time
        elapsed_time = time.time() - start_time
        print(f'Time on epoch {i}: {dt:.2f}')
    print(f'Total time: {elapsed_time:.2f}')
    print(f'Averate time per epoch: {elapsed_time/args.epochs:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark a dataset')
    parser.add_argument('input', help='dataset directory')
    parser.add_argument('--segment_length', type=float, default=0.0)
    parser.add_argument('--sampler', type=str, default='bucket')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--batch_size', type=float, default=1)
    parser.add_argument('--buckets', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--fs', type=int, default=16e3)
    parser.add_argument('--sources', type=str, nargs='+')
    parser.add_argument('--num_buckets', type=int, default=10)
    args = parser.parse_args()
    main()

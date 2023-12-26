import argparse

import matplotlib.pyplot as plt

from brever.data import BreverDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    dset = BreverDataset(args.path)
    print(f'Dataset duration: {dset._duration/3600:.2f} h')
    print('Dataset effective duration after segmentation: '
          f'{dset._effective_duration/3600:.2f} h')
    print(f'Number of segments: {len(dset)}')

    lengths = [dset.get_segment_length(i) / dset.fs for i in range(len(dset))]

    print('Segment length statistics:')
    print(
        f'    {"mean":<6} {sum(lengths) / len(lengths):>6.2f} s\n'
        f'    {"median":<6} {sorted(lengths)[len(lengths) // 2]:>6.2f} s\n'
        f'    {"min":<6} {min(lengths):>6.2f} s\n'
        f'    {"max":<6} {max(lengths):>6.2f} s\n'
    )

    plt.hist(lengths, bins=100)
    plt.xlabel('Mixture length (s)')
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    main()

import argparse

import numpy as np
from tqdm import tqdm

from brever.io import AudioFileLoader
from brever.mixture import split_brir
from brever.utils import pretty_table


def main():
    loader = AudioFileLoader()
    summary = {}
    rooms = set()
    for regexp in args.regexes:
        rooms = rooms.union(loader.get_rooms(regexp))
    if args.plot:
        iterable = sorted(rooms)
    else:
        iterable = tqdm(sorted(rooms))
    total_brirs = 0
    avg_drr = 0
    avg_rt60 = 0
    for room in iterable:
        summary[room] = {}
        brirs, _ = loader.load_brirs(room)
        drr = np.mean([estimate_drr(brir) for brir in brirs[:1]])
        rt60 = np.mean([estimate_rt60(brir, room) for brir in brirs[:1]])
        summary[room]['brirs'] = len(brirs)
        summary[room]['drr [dB]'] = f'{drr:.2f}'
        summary[room]['rt60 [s]'] = f'{rt60:.2f}'
        total_brirs += len(brirs)
        avg_drr += drr * len(brirs)
        avg_rt60 += rt60 * len(brirs)
    summary['.*'] = {
        'brirs': total_brirs,
        'drr [dB]': f'{avg_drr/total_brirs:.2f}',
        'rt60 [s]': f'{avg_rt60/total_brirs:.2f}',
    }
    pretty_table(summary)


def estimate_drr(brir):
    brir_early, brir_late = split_brir(brir)
    return 10*np.log10(np.sum(brir_early**2)/np.sum(brir_late**2))


def estimate_rt60(brir, room, fs=16000):
    # Calculate the energy decay curve (EDC)
    edc = np.sum(brir**2, axis=1)
    edc = np.cumsum(edc[::-1])[::-1]/np.sum(edc)
    edc = 10*np.log10(edc + 1e-10)

    # Find the -20 dB and -40 dB points
    idx1 = np.argmax(edc <= args.edc_db_1)
    idx2 = np.argmax(edc <= args.edc_db_2)

    # Calculate the RT60
    rt60 = (idx2 - idx1) / fs * 60 / (args.edc_db_1 - args.edc_db_2)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(edc))/fs, edc)
        plt.plot([idx1/fs], [edc[idx1]], 'ro')
        plt.plot([idx2/fs], [edc[idx2]], 'ro')
        slope = (args.edc_db_2 - args.edc_db_1) / (idx2 - idx1)
        offset = args.edc_db_1 - slope * idx1
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        plt.plot([0, len(edc)/fs], [offset, offset + slope*len(edc)], 'r--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        print(f'{room} RT60: {rt60:.2f} [s]')
        plt.show()

    return rt60


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BRIR specifications')
    parser.add_argument('regexes', nargs='*', default=['.*'],
                        help='room regular expressions')
    parser.add_argument('--plot', action='store_true',
                        help='plot EDC fit for RT60 estimation')
    parser.add_argument('--edc_db_1', default=-20, type=float,
                        help='first dB value on EDC for RT60 estimation')
    parser.add_argument('--edc_db_2', default=-30, type=float,
                        help='second dB value on EDC for RT60 estimation')
    args = parser.parse_args()
    main()

import argparse
import json
import logging
import os
import pprint
import shutil
import sys
import tarfile
import tempfile

import soundfile as sf
from tqdm import tqdm

from brever.config import get_config
from brever.logger import set_logger
from brever.mixture import RandomMixtureMaker


def main():
    # check if already created
    mix_info_path = os.path.join(args.input, 'mixture_info.json')
    if os.path.exists(mix_info_path) and not args.force:
        raise FileExistsError(f'dataset already created: {mix_info_path}')

    # load config file
    cfg_path = os.path.join(args.input, 'config.yaml')
    cfg = get_config(cfg_path)

    # init logger
    log_file = os.path.join(args.input, 'log.log')
    set_logger(log_file)
    logging.info(f'Creating {args.input}')
    logging.info(f'Configuration: \n {pprint.pformat(cfg.to_dict())}')

    # output directory or archive
    mix_dirname = 'audio'
    if args.no_tar:
        mix_dirpath = os.path.join(args.input, mix_dirname)
        if os.path.exists(mix_dirpath):
            shutil.rmtree(mix_dirpath)
        os.mkdir(mix_dirpath)
    else:
        archive_path = os.path.join(args.input, f'{mix_dirname}.tar')
        archive = tarfile.open(archive_path, 'w')

    # mixture maker
    rand_mix_maker = RandomMixtureMaker(**cfg.rmm.to_dict())

    # main loop
    metadatas = []
    duration = 0
    i = 0
    with tqdm(total=cfg.duration, unit_scale=1, file=sys.stdout) as pbar:
        while duration < cfg.duration:

            # make mixture and save
            mix_obj, metadata = rand_mix_maker()
            for name in cfg.sources:
                filename = f'{i:05d}_{name}.flac'
                if args.no_tar:
                    filepath = os.path.join(mix_dirpath, filename)
                    sf.write(filepath, getattr(mix_obj, name), cfg.rmm.fs)
                else:
                    temp = tempfile.NamedTemporaryFile(
                        prefix='brever_',
                        suffix='.flac',
                        delete=False,
                    )
                    sf.write(temp, getattr(mix_obj, name), cfg.rmm.fs)
                    temp.close()
                    arcname = os.path.join(mix_dirname, filename)
                    archive.add(temp.name, arcname=arcname)
                    os.remove(temp.name)
            metadatas.append(metadata)

            # update duration and progress bar
            step = len(mix_obj)/cfg.rmm.fs
            pbar.update(min(step, cfg.duration - duration))
            duration += step
            i += 1

    # close archive
    if not args.no_tar:
        archive.close()

    # save mixtures metadata
    with open(mix_info_path, 'w') as f:
        json.dump(metadatas, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create a dataset')
    parser.add_argument('input',
                        help='dataset directory')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite if already exists')
    parser.add_argument('--no_tar', action='store_true',
                        help='do not save mixtures in tar archive')
    args = parser.parse_args()
    main()

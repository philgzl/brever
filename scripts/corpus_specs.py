import soundfile as sf

from brever.io import AudioFileLoader
from brever.logger import set_logger
from brever.utils import pretty_table


def speech_corpora_details(loader):
    dict_ = {}
    for db in [
        'timit',
        'libri',
        'wsj0',
        'clarity',
        'vctk',
    ]:
        spks = loader.get_speakers(db)
        dict_[db] = {}
        dict_[db]['speakers'] = len(spks)
        utts = sum(len(x) for x in spks.values())
        utts_per_spk = [len(x) for x in spks.values()]
        dict_[db]['utterances'] = utts
        dict_[db]['avg_utt/spk'] = round(sum(utts_per_spk)/len(utts_per_spk))
        dict_[db]['min_utt/spk'] = min(utts_per_spk)
        dict_[db]['max_utt/spk'] = max(utts_per_spk)
        duration = loader.get_duration(f'{db}_.*', reduce_=False)[0]
        total = sum(duration)
        dict_[db]['duration'] = f'{total/3600:.1f}h'
        dict_[db]['avg_utt_len'] = f'{total/utts:.1f}s'
        dict_[db]['min_utt_len'] = f'{min(duration):.1f}s'
        dict_[db]['max_utt_len'] = f'{max(duration):.1f}s'

    pretty_table(dict_, 'corpus')


def noise_database_details(loader):
    dict_ = {}
    for db in [
        'dcase',
        'noisex',
        'icra',
        'demand',
        'arte',
    ]:
        files = loader.get_noise_files(f'{db}_.*')
        dict_[db] = {}
        dict_[db]['duration'] = \
            f'{sum(sf.info(file).duration for file in files)/3600:.1f}h'
        dict_[db]['files'] = len(files)

    pretty_table(dict_, 'database')


def brir_database_details(loader):
    dict_ = {}
    for db in [
        'surrey',
        'ash',
        'bras',
        'catt',
        'avil',
    ]:
        dict_[db] = {}
        rooms = loader.get_rooms(f'{db}_.*')
        dict_[db]['rooms'] = len(rooms)
        dict_[db]['files'] = 0
        for room in rooms:
            brirs, _ = loader.load_brirs(room)
            dict_[db]['files'] += len(brirs)

    pretty_table(dict_, 'database')


if __name__ == '__main__':
    set_logger()
    loader = AudioFileLoader()
    print('Speech corpora...')
    speech_corpora_details(loader)
    print('Noise databases...')
    noise_database_details(loader)
    print('BRIR databases...')
    brir_database_details(loader)

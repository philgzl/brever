import re

import soundfile as sf

from .io import is_long_recording
from .random import (AngleRandGen, ChoiceRandGen, DistRandGen,
                     MultiChoiceRandGen, MultiDistRandGen, NoiseFileRandGen,
                     Seeder, TargetFileRandGen)


class BaseMetadata:
    def __init__(self, name=None, toggle=True):
        self._rand_gens = []
        self._constants = []
        self._metadatas = []
        self.name = name
        self.toggle = toggle

    def add_rand_gen(self, rand_gen, name=None):
        self._rand_gens.append((rand_gen, name))
        return rand_gen

    def add_metadata(self, metadata):
        self._metadatas.append(metadata)
        return metadata

    def add_constant(self, value, name):
        self._constants.append((value, name))
        return value

    def roll(self):
        for rand_gen, _ in self._rand_gens:
            rand_gen.roll()
        for metadata in self._metadatas:
            metadata.roll()

    def get(self, toggle=None):
        output = {
            **{
                name: rand_gen.get()
                for rand_gen, name in self._rand_gens
                if name is not None
            },
            **{name: value for value, name in self._constants},
            **{
                key: value
                for metadata in self._metadatas
                for key, value in metadata.get().items()
            },
        }
        if self.name is not None:
            output = {self.name: output}
        if toggle is None:
            if not self.toggle:
                output = {}
        elif not toggle:
            output = {}
        return output


class Metadata(BaseMetadata):
    def __init__(self, constants={}, dists={}, name=None, toggle=True,
                 seeder=None):
        super().__init__(name=name, toggle=toggle)
        for name, value in constants.items():
            self.add_constant(value, name)
        for name, dist in dists.items():
            self.add_rand_gen(
                DistRandGen(
                    dist_name=dist['name'],
                    dist_args=dist['args'],
                    seed=None if seeder is None else seeder(),
                ),
                name=name,
            )


class RoomMetadata(BaseMetadata):
    def __init__(self, loader, rooms, seeder):
        super().__init__()
        self.room_regexps = self.add_rand_gen(
            ChoiceRandGen(
                pool=rooms,
                seed=seeder(),
            )
        )
        self.rooms = self.add_rand_gen(
            MultiChoiceRandGen(
                pool_dict=loader._room_regexps,
                seed=seeder(),
            )
        )

    def get(self):
        room_regexp = self.room_regexps.get()
        return {'room': self.rooms.get(room_regexp)}


class TargetMetadata(BaseMetadata):
    def __init__(self, loader, speakers, weight_by_avg_length, file_lims,
                 angle_lims, angle_parity, seeder):
        super().__init__()
        # For the speaker random generator, the probability distribution must
        # be weighted according to the average duration of the sentences,
        # otherwise the speech material in the dataset will be unbalanced.
        # Example: TIMIT sentences are 3 seconds long on average, while
        # LibriSpeech sentences are 12 seconds long on average, so making a
        # dataset using 50 TIMIT sentences and 50 LibriSpeech sentences will
        # result in much more LibriSpeech material.
        if weight_by_avg_length:
            weights = loader.calc_weights(speakers)
        else:
            weights = None
        self.speakers = self.add_rand_gen(
            ChoiceRandGen(
                pool=speakers,
                weights=weights,
                seed=seeder(),
            )
        )
        self.speaker_ids = self.add_rand_gen(
            MultiChoiceRandGen(
                pool_dict={
                    regexp: sorted(filter(
                        re.compile(regexp).match,
                        loader._speech_files.keys(),
                    )) for regexp in speakers},
                seed=seeder(),
            )
        )
        self.files = self.add_rand_gen(
            TargetFileRandGen(
                pool_dict=loader._speech_files,
                lims=file_lims,
                seed=seeder(),
            )
        )
        self.angles = self.add_rand_gen(
            AngleRandGen(
                pool_dict=loader._room_angles,
                lims=angle_lims,
                parity=angle_parity,
                seed=seeder(),
            )
        )

    def get(self, room):
        speaker = self.speakers.get()
        speaker_id = self.speaker_ids.get(speaker)
        return {
            'target': {
                'file': self.files.get(speaker_id),
                'angle': self.angles.get(room),
            }
        }


class NoiseMetadata(BaseMetadata):
    def __init__(self, loader, noises, num, file_lims, angle_lims,
                 angle_parity, seeder):
        super().__init__()
        self.noises = self.add_rand_gen(
            ChoiceRandGen(
                pool=noises,
                size=num[1],
                seed=seeder(),
                squeeze=False,
            )
        )
        self.nums = self.add_rand_gen(
            DistRandGen(
                dist_name='randint',
                dist_args=[num[0], num[1]+1],
                seed=seeder(),
            )
        )
        self.files = self.add_rand_gen(
            NoiseFileRandGen(
                pool_dict=loader._noise_files,
                lims=file_lims,
                size=num[1],
                replace=False,
                seed=seeder(),
                squeeze=False,
            )
        )
        self.angles = self.add_rand_gen(
            AngleRandGen(
                pool_dict=loader._room_angles,
                lims=angle_lims,
                size=num[1],
                parity=angle_parity,
                seed=seeder(),
                squeeze=False,
            )
        )
        self.indexes = self.add_rand_gen(
            MultiDistRandGen(
                dist_name='randint',
                dist_args=[0, 16000*3600],  # should cover any file
                size=num[1],
                seed=seeder(),
            )
        )

        self.loader = loader
        self.file_lims = file_lims

    def get(self, room, target_frames):
        number = self.nums.get()
        noises = self.noises.get()[:number]
        angles = self.angles.get(room)[:number]
        idxs = self.indexes.get()[:number]
        if number == 0:
            return {}
        output = {'noises': []}
        for i, (noise, angle, i_start) in enumerate(zip(noises, angles, idxs)):
            file, i_min, i_max = self.get_file_and_idx_lims(
                i, noise, target_frames, i_start
            )
            output['noises'].append({
                'type': noise,
                'angle': angle,
                'file': file,
                'i_start': i_start,
                'i_min': i_min,
                'i_max': i_max,
            })
        return output

    def get_file_and_idx_lims(self, i, noise, target_frames, i_start):
        if noise.startswith('colored_') or noise == 'ssn':
            file, i_min, i_max = None, None, None
        else:
            file = self.files.get(noise, i)
            noise_frames = sf.info(file).frames
            if is_long_recording(noise):
                i_min = round(self.file_lims[0]*noise_frames)
                i_max = round(self.file_lims[1]*noise_frames)
            else:
                i_min, i_max = 0, noise_frames
        return file, i_min, i_max


class DecayMetadata(BaseMetadata):
    def __init__(self, toggle, color, rt60_dist_name, rt60_dist_args,
                 drr_dist_name, drr_dist_args, delay_dist_name,
                 delay_dist_args, seeder):
        super().__init__(name='decay', toggle=toggle)
        self.color = self.add_constant(color, 'color')
        self.rt60s = self.add_rand_gen(
            DistRandGen(
                dist_name=rt60_dist_name,
                dist_args=rt60_dist_args,
                seed=seeder(),
            ),
            name='rt60',
        )
        self.drrs = self.add_rand_gen(
            DistRandGen(
                dist_name=drr_dist_name,
                dist_args=drr_dist_args,
                seed=seeder(),
            ),
            name='drr',
        )
        self.delays = self.add_rand_gen(
            DistRandGen(
                dist_name=delay_dist_name,
                dist_args=delay_dist_args,
                seed=seeder(),
            ),
            name='delay',
        )
        self.seeds = self.add_rand_gen(
            DistRandGen(
                dist_name='randint',
                dist_args=[0, 2**16],
                seed=seeder(),
            ),
            name='seed',
        )


class MixtureMetadata(BaseMetadata):
    def __init__(
        self,
        loader,
        fs: int = 16000,
        seed: int = 0,
        padding: float = 0.0,
        uniform_tmr: bool = False,
        reflection_boundary: float = 0.05,
        speakers: set[str] = {'libri_.*'},
        noises: set[str] = {'dcase_.*'},
        rooms: set[str] = {'surrey_.*'},
        target_snr_dist_name: str = 'uniform',
        target_snr_dist_args: tuple[float, float] = (-5.0, 10.0),
        target_angle: tuple[float, float] = (-90.0, 90.0),
        noise_num: tuple[int, int] = (1, 3),
        noise_angle: tuple[float, float] = (-90.0, 90.0),
        noise_ndr_dist_name: str = 'uniform',
        noise_ndr_dist_args: tuple[float, float] = (0.0, 30.0),
        diffuse: bool = False,
        diffuse_color: str = 'white',
        diffuse_ltas_eq: bool = False,
        decay: bool = False,
        decay_color: str = 'white',
        decay_rt60_dist_name: str = 'uniform',
        decay_rt60_dist_args: tuple[float, float] = (0.1, 5.0),
        decay_drr_dist_name: str = 'uniform',
        decay_drr_dist_args: tuple[float, float] = (5.0, 35.0),
        decay_delay_dist_name: str = 'uniform',
        decay_delay_dist_args: tuple[float, float] = (0.075, 0.100),
        rms_jitter_dist_name: str = 'uniform',
        rms_jitter_dist_args: tuple[float, float] = (0.0, 0.0),
        speech_files: tuple[float, float] = (0.0, 1.0),
        noise_files: tuple[float, float] = (0.0, 1.0),
        room_files: str = 'all',
        weight_by_avg_length: bool = False,
    ):
        super().__init__()

        seeder = Seeder(seed)

        self.room_meta = self.add_metadata(
            RoomMetadata(
                loader=loader,
                rooms=rooms,
                seeder=seeder,
            )
        )
        self.target_meta = self.add_metadata(
            TargetMetadata(
                loader=loader,
                speakers=speakers,
                weight_by_avg_length=weight_by_avg_length,
                file_lims=speech_files,
                angle_lims=target_angle,
                angle_parity=room_files,
                seeder=seeder,
            )
        )
        self.noise_meta = self.add_metadata(
            NoiseMetadata(
                loader=loader,
                noises=noises,
                num=noise_num,
                file_lims=noise_files,
                angle_lims=noise_angle,
                angle_parity=room_files,
                seeder=seeder,
            )
        )
        self.decay_meta = self.add_metadata(
            DecayMetadata(
                toggle=decay,
                color=decay_color,
                rt60_dist_name=decay_rt60_dist_name,
                rt60_dist_args=decay_rt60_dist_args,
                drr_dist_name=decay_drr_dist_name,
                drr_dist_args=decay_drr_dist_args,
                delay_dist_name=decay_delay_dist_name,
                delay_dist_args=decay_delay_dist_args,
                seeder=seeder,
            )
        )
        self.diffuse_meta = self.add_metadata(
            Metadata(
                constants={
                    'color': diffuse_color,
                    'ltas_eq': diffuse_ltas_eq,
                },
                name='diffuse',
                toggle=diffuse,
                seeder=seeder,
            )
        )
        self.ndr_meta = self.add_metadata(
            Metadata(
                dists={
                    'ndr': {
                        'name': noise_ndr_dist_name,
                        'args': noise_ndr_dist_args,
                    },
                },
                seeder=seeder,
            )
        )
        self.snr_meta = self.add_metadata(
            Metadata(
                dists={
                    'snr': {
                        'name': target_snr_dist_name,
                        'args': target_snr_dist_args,
                    },
                },
                seeder=seeder,
            )
        )
        self.rms_jitter_meta = self.add_metadata(
            Metadata(
                dists={
                    'rms_jitter': {
                        'name': rms_jitter_dist_name,
                        'args': rms_jitter_dist_args,
                    },
                },
                seeder=seeder,
            )
        )
        self.tmr_meta = self.add_metadata(
            Metadata(
                dists={
                    'tmr': {
                        'name': 'uniform',
                        'args': (0.0, 1.0),
                    },
                },
                toggle=uniform_tmr,
                seeder=seeder,
            )
        )

    def get(self):
        room_meta = self.room_meta.get()
        target_meta = self.target_meta.get(room_meta['room'])
        frames = sf.info(target_meta['target']['file']).frames
        noise_meta = self.noise_meta.get(room_meta['room'], frames)
        decay_meta = self.decay_meta.get()
        diffuse_meta = self.diffuse_meta.get()
        ndr_meta = self.ndr_meta.get(toggle=diffuse_meta and noise_meta)
        snr_meta = self.snr_meta.get(toggle=diffuse_meta or noise_meta)
        rms_jitter_meta = self.rms_jitter_meta.get()
        tmr_meta = self.tmr_meta.get()
        return {
            **room_meta,
            **target_meta,
            **noise_meta,
            **decay_meta,
            **diffuse_meta,
            **ndr_meta,
            **snr_meta,
            **rms_jitter_meta,
            **tmr_meta,
            'frames': frames,
        }

import re

from .io import AudioFileLoader
from .metadata import MixtureMetadata
from .mixture import BRIRDecay, Mixture, colored_noise, match_ltas


class RandomMixtureMaker:
    """Main random mixture maker class."""

    def __init__(
        self,
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
        # init audio file loader
        self.loader = AudioFileLoader(fs)
        self.loader.scan_material(speakers, noises, rooms)

        # set attributes needed later
        self.fs = fs
        self.padding = padding
        self.reflection_boundary = reflection_boundary

        # calculate ltas now for efficiency
        if ((diffuse and diffuse_ltas_eq)
                or ('ssn' in noises and noise_num[1] > 0)):
            self.ltas = self.loader.calc_ltas(speakers)
        else:
            self.ltas = None

        self.metadata = MixtureMetadata(
            self.loader,
            fs=fs,
            seed=seed,
            padding=padding,
            uniform_tmr=uniform_tmr,
            reflection_boundary=reflection_boundary,
            speakers=speakers,
            noises=noises,
            rooms=rooms,
            target_snr_dist_name=target_snr_dist_name,
            target_snr_dist_args=target_snr_dist_args,
            target_angle=target_angle,
            noise_num=noise_num,
            noise_angle=noise_angle,
            noise_ndr_dist_name=noise_ndr_dist_name,
            noise_ndr_dist_args=noise_ndr_dist_args,
            diffuse=diffuse,
            diffuse_color=diffuse_color,
            diffuse_ltas_eq=diffuse_ltas_eq,
            decay=decay,
            decay_color=decay_color,
            decay_rt60_dist_name=decay_rt60_dist_name,
            decay_rt60_dist_args=decay_rt60_dist_args,
            decay_drr_dist_name=decay_drr_dist_name,
            decay_drr_dist_args=decay_drr_dist_args,
            decay_delay_dist_name=decay_delay_dist_name,
            decay_delay_dist_args=decay_delay_dist_args,
            rms_jitter_dist_name=rms_jitter_dist_name,
            rms_jitter_dist_args=rms_jitter_dist_args,
            speech_files=speech_files,
            noise_files=noise_files,
            room_files=room_files,
            weight_by_avg_length=weight_by_avg_length,
        )

    def __call__(self):
        self.metadata.roll()
        metadata = self.metadata.get()
        mix = self.make_from_metadata(metadata)
        return mix, metadata

    def make_from_metadata(self, metadata):
        mix = Mixture()
        decay = self.init_decay(metadata)
        self.add_target(mix, metadata, decay)
        self.add_noises(mix, metadata, decay)
        self.add_diffuse_noise(mix, metadata)
        if 'ndr' in metadata:
            mix.set_ndr(metadata['ndr'])
        if 'snr' in metadata:
            mix.set_snr(metadata['snr'])
        if 'tmr' in metadata:
            mix.set_tmr(metadata['tmr'])
        mix.set_rms(mix.get_rms() + metadata['rms_jitter'])
        return mix

    def init_decay(self, metadata):
        if 'decay' in metadata:
            return BRIRDecay(
                rt60=metadata['decay']['rt60'],
                drr=metadata['decay']['drr'],
                delay=metadata['decay']['delay'],
                color=metadata['decay']['color'],
                fs=self.fs,
            )
        else:
            return None

    def add_target(self, mix, metadata, decay):
        x = self.loader.load_file(metadata['target']['file'])
        brir, _ = self.loader.load_brirs(metadata['room'],
                                         metadata['target']['angle'])
        if decay is not None:
            brir = decay(brir, seed=metadata['decay']['seed'])
        mix.add_speech(
            x=x,
            brir=brir,
            reflection_boundary=self.reflection_boundary,
            padding=self.padding,
            fs=self.fs,
        )

    def add_noises(self, mix, metadata, decay):
        if 'noises' in metadata:
            xs = self.make_noises(metadata)
            angles = [noise['angle'] for noise in metadata['noises']]
            brirs, _ = self.loader.load_brirs(metadata['room'], angles)
            if decay is not None:
                brirs = [decay(brir) for brir in brirs]
            mix.add_noises(xs, brirs)

    def add_diffuse_noise(self, mix, metadata):
        if 'diffuse' in metadata:
            brirs, _ = self.loader.load_brirs(metadata['room'])
            mix.add_diffuse_noise(
                brirs=brirs,
                color=metadata['diffuse']['color'],
                ltas=self.ltas if metadata['diffuse']['ltas_eq'] else None,
            )

    def make_noises(self, metadata):
        xs = []
        for i, noise in enumerate(metadata['noises']):
            if noise['type'].startswith('colored_'):
                color = re.match('^colored_(.*)$', noise['type']).group(1)
                x = colored_noise(color, metadata['frames'])
            elif noise['type'] == 'ssn':
                x = colored_noise('white', metadata['frames'])
                x = match_ltas(x, self.ltas)
            else:
                x = self.loader.load_noise(
                    file=noise['file'],
                    n_samples=metadata['frames'],
                    i_start=noise['i_start'],
                    i_min=noise['i_min'],
                    i_max=noise['i_max'],
                )
            xs.append(x)
        return xs

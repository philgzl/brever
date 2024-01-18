# brever
Speech enhancement in noisy and reverberant environments using deep neural networks.

* Generate datasets of noisy and reverberant mixtures using multiple databases of clean speech utterances, noise recordings and binaural room impulse responses (BRIRs).
* Train different learning-based speech enhancement systems. Currently implemented models are:
  * A feed-forward neural network (FFNN)-based system
  * Conv-TasNet ([Y. Luo and N. Mesgarani](https://doi.org/10.1109/TASLP.2019.2915167))
  * DCCRN ([Y. Hu et al.](https://doi.org/10.21437/Interspeech.2020-2537))
  * SGMSE+ ([J. Richter et al.](https://doi.org/10.1109/TASLP.2023.3285241)), SGMSE+M ([J.-M. Lemercier et al.](https://doi.org/10.1109/ICASSP49357.2023.10095258))
  * MANNER ([H. J. Park et al.](https://doi.org/10.1109/ICASSP43922.2022.9747120))
  * MetricGAN+ ([S. Fu et al.](https://doi.org/10.21437/Interspeech.2021-599)), MetricGAN-OKD ([W. Shin et al.](https://proceedings.mlr.press/v202/shin23b.html)) [experimental; does not seem to work]
* Evaluate models in terms of different metrics: PESQ, STOI, ESTOI, SNR, SI-SNR.

"brever" reads "reverb" backwards.

# Table of Contents
- [Installation](#installation)
- [External databases](#external-databases)
- [How to use](#how-to-use)
  * [Creating datasets](#creating-datasets)
    + [Using custom datasets](#using-custom-datasets)
    + [Using VoiceBank+DEMAND](#using-voicebank-demand)
  * [Training models](#training-models)
  * [Testing models](#testing-models)
- [Contributing](#contributing)
  * [Implementing new models](#implementing-new-models)
  * [TODO](#todo)
- [Docker](#docker)
- [Related publications](#related-publications)

# Installation

The code was tested with Python 3.10 and 3.11.

1. Clone the repo:
```
git clone https://github.com/philgzl/brever.git
cd brever
```

2. Create a virtual environment (optional):
```
python -m venv venv
source venv/bin/activate
```

3. Install requirements:
```
pip install -r requirements.txt
```

# External databases

External databases of clean speech utterances, noise recordings and binaural room impulse responses (BRIRs) are required to generate datasets of noisy and reverberant mixtures. The following databases are used:

- Speech databases:
  - [TIMIT](https://doi.org/10.35111/17gk-bn40) (not open)
  - [LibriSpeech](http://www.openslr.org/12) (open)
  - [WSJ0](https://doi.org/10.35111/ewkm-cg47) (not open)
  - [VCTK](https://doi.org/10.7488/ds/2645) (open)
  - [Clarity](https://doi.org/10.17866/rd.salford.16918180.v3) (open)
- Noise databases:
  - [TAU](https://doi.org/10.5281/zenodo.2589280) (open, referred to as `dcase` in the code)
  - [NOISEX](https://doi.org/10.1016/0167-6393(93)90095-3) (not open)
  - [ICRA](https://icra-audiology.org/Repository/icra-noise) (not open)
  - [DEMAND](https://doi.org/10.5281/zenodo.1227121) (open)
  - [ARTE](https://doi.org/10.5281/zenodo.3386569) (open)
- BRIR databases:
  - [Surrey](https://github.com/IoSR-Surrey/RealRoomBRIRs) (open)
  - [ASH](https://github.com/ShanonPearce/ASH-IR-Dataset) (open)
  - [BRAS](https://doi.org/10.14279/depositonce-6726.3) (open)
  - [CATT](http://iosr.surrey.ac.uk/software/index.php#CATT_RIRs) (open)
  - [AVIL](https://doi.org/10.17743/jaes.2020.0026) (not open)

The path to each database in the file system is specified in `config/paths.yaml`.

The open databases can be downloaded using the `download_databases.sh` script (tested on Linux only). The script also resamples the VCTK, Clarity and TAU files from 48 kHz to 16 kHz and converts them to mono FLAC files using `ffmpeg`. Once the script is finished, the integrity of the files can be checked using `pytest tests/test_loader.py` (the tests for the non-open databases are skipped).

The WSJ0 files should be reorganized by speaker using [this script](https://github.com/philgzl/wsj0-convert).

# How to use

## Creating datasets

You can initialize a dataset using `scripts/init_dataset.py`. The script takes as mandatory argument the type of dataset (`train`, `val` or `test`) and as optional arguments the parameters for the dataset. The script creates a new directory under `data/datasets/train/`, `data/datasets/val/` or `data/datasets/test/` which contains a `config.yaml` file with all the dataset parameters. The directory is named by default after a unique ID generated from the `config.yaml` file.
```
usage: init_dataset.py [-h] [--fs FS] [--seed SEED] [--padding PADDING]
                       [--uniform_tmr UNIFORM_TMR]
                       [--reflection_boundary REFLECTION_BOUNDARY]
                       [--speakers SPEAKERS] [--noises NOISES] [--rooms ROOMS]
                       [--target_snr_dist_name TARGET_SNR_DIST_NAME]
                       [--target_snr_dist_args TARGET_SNR_DIST_ARGS]
                       [--target_angle TARGET_ANGLE] [--noise_num NOISE_NUM]
                       [--noise_angle NOISE_ANGLE]
                       [--noise_ndr_dist_name NOISE_NDR_DIST_NAME]
                       [--noise_ndr_dist_args NOISE_NDR_DIST_ARGS]
                       [--diffuse DIFFUSE] [--diffuse_color DIFFUSE_COLOR]
                       [--diffuse_ltas_eq DIFFUSE_LTAS_EQ] [--decay DECAY]
                       [--decay_color DECAY_COLOR]
                       [--decay_rt60_dist_name DECAY_RT60_DIST_NAME]
                       [--decay_rt60_dist_args DECAY_RT60_DIST_ARGS]
                       [--decay_drr_dist_name DECAY_DRR_DIST_NAME]
                       [--decay_drr_dist_args DECAY_DRR_DIST_ARGS]
                       [--decay_delay_dist_name DECAY_DELAY_DIST_NAME]
                       [--decay_delay_dist_args DECAY_DELAY_DIST_ARGS]
                       [--rms_jitter_dist_name RMS_JITTER_DIST_NAME]
                       [--rms_jitter_dist_args RMS_JITTER_DIST_ARGS]
                       [--speech_files SPEECH_FILES]
                       [--noise_files NOISE_FILES] [--room_files ROOM_FILES]
                       [--weight_by_avg_length WEIGHT_BY_AVG_LENGTH]
                       [--duration DURATION] [--sources SOURCES] [-f]
                       [-n NAME] [--all_databases]
                       {train,val,test}

initialize a dataset

positional arguments:
  {train,val,test}      dump in train or test subdir

options:
  -h, --help            show this help message and exit
  -f, --force           overwrite config file if already exists
  -n NAME, --name NAME  dataset name
  --all_databases       use all databases

random mixture maker options:
  --fs FS
  --seed SEED
  --padding PADDING
  --uniform_tmr UNIFORM_TMR
  --reflection_boundary REFLECTION_BOUNDARY
  --speakers SPEAKERS
  --noises NOISES
  --rooms ROOMS
  --target_snr_dist_name TARGET_SNR_DIST_NAME
  --target_snr_dist_args TARGET_SNR_DIST_ARGS
  --target_angle TARGET_ANGLE
  --noise_num NOISE_NUM
  --noise_angle NOISE_ANGLE
  --noise_ndr_dist_name NOISE_NDR_DIST_NAME
  --noise_ndr_dist_args NOISE_NDR_DIST_ARGS
  --diffuse DIFFUSE
  --diffuse_color DIFFUSE_COLOR
  --diffuse_ltas_eq DIFFUSE_LTAS_EQ
  --decay DECAY
  --decay_color DECAY_COLOR
  --decay_rt60_dist_name DECAY_RT60_DIST_NAME
  --decay_rt60_dist_args DECAY_RT60_DIST_ARGS
  --decay_drr_dist_name DECAY_DRR_DIST_NAME
  --decay_drr_dist_args DECAY_DRR_DIST_ARGS
  --decay_delay_dist_name DECAY_DELAY_DIST_NAME
  --decay_delay_dist_args DECAY_DELAY_DIST_ARGS
  --rms_jitter_dist_name RMS_JITTER_DIST_NAME
  --rms_jitter_dist_args RMS_JITTER_DIST_ARGS
  --speech_files SPEECH_FILES
  --noise_files NOISE_FILES
  --room_files ROOM_FILES
  --weight_by_avg_length WEIGHT_BY_AVG_LENGTH

extra options:
  --duration DURATION
  --sources SOURCES
```

The dataset is then created using the `scripts/create_dataset.py` script.
```
usage: create_dataset.py [-h] [-f] [--no_tar] input

create a dataset

positional arguments:
  input        dataset directory

options:
  -h, --help   show this help message and exit
  -f, --force  overwrite if already exists
  --no_tar     do not save mixtures in tar archive
```

Example:
```
$ python scripts/init_dataset.py train --duration 3600 --speakers libri_.* --noises demand --rooms surrey_.*
Initialized data/datasets/train/<dataset_id>/config.yaml
$ python scripts/create_dataset.py data/datasets/train/<dataset_id>/
```

The following files are then created next to the `config.yaml` file:
- `audio.tar`: an archive containing noisy and clean speech files in FLAC format
- `log.log`: a log file
- `mixture_info.json`: metadata about each mixture

### Using custom datasets

You can use your own datasets of noisy speech mixtures, but the files should be organized as follows:
- Each of your training, validation and test directories should contain a sub-directory called `audio` containing all the audio files.
- The audio files should be named `f'{i:05d}_mixture.flac'` for the noisy speech files and `f'{i:05d}_foreground.flac'` for the clean speech files where `i` is the acoustic scene index starting from `0`.
- Finally, either
  * Bundle each `audio` sub-directory into a tar archive named `audio.tar` using e.g. `tar -cvf audio.tar audio/`
  * Or make sure to pass `--tar 0` to `scripts/init_model.py` or `scripts/train_model.py` so the training script does not look for a tar archive.

Example:
```
$ tree <my_custom_train_dataset>
<my_custom_train_dataset>
└── audio
    ├── 00000_foreground.flac
    ├── 00000_mixture.flac
    ├── 00001_foreground.flac
    ├── 00001_mixture.flac
    ├── 00002_foreground.flac
    ├── 00002_mixture.flac
    ...
$ tree <my_custom_val_dataset>
<my_custom_val_dataset>
└── audio
    ├── 00000_foreground.flac
    ├── 00000_mixture.flac
    ├── 00001_foreground.flac
    ├── 00001_mixture.flac
    ├── 00002_foreground.flac
    ├── 00002_mixture.flac
    ...
```

Or:
```
$ tree <my_custom_train_dataset>
<my_custom_train_dataset>
└── audio.tar
    ├── audio/00000_foreground.flac
    ├── audio/00000_mixture.flac
    ├── audio/00001_foreground.flac
    ├── audio/00001_mixture.flac
    ├── audio/00002_foreground.flac
    ├── audio/00002_mixture.flac
    ...
$ tree <my_custom_val_dataset>
<my_custom_val_dataset>
└── audio.tar
    ├── audio/00000_foreground.flac
    ├── audio/00000_mixture.flac
    ├── audio/00001_foreground.flac
    ├── audio/00001_mixture.flac
    ├── audio/00002_foreground.flac
    ├── audio/00002_mixture.flac
    ...
```

### Using VoiceBank+DEMAND

You can also download the [VoiceBank+DEMAND](https://doi.org/10.7488/ds/2117) dataset using `scripts/vbdemand_to_brever.py` to quickly get started. The script resamples the files from 48 kHz to 16 kHz and converts them to FLAC. The `--val_speakers` option can be used to select the set of speakers used for validation (`p226` and `p287` by default). The training, validation and test sets are then placed in `data/datasets/train/vbdemand/`, `data/datasets/val/vbdemand/` and `data/datasets/test/vbdemand/` respectively.

## Training models

You can initialize a model using `scripts/init_model.py`. The script takes as optional arguments the training parameters, and as a mandatory sub-command the model architecture. The sub-command then takes as optional arguments the parameters for the model. The script creates a new directory under `models/` which contains a `config.yaml` file with all the model parameters. The directory is named by default after a unique ID generated from the `config.yaml` file. The `--train_path` and `--val_path` arguments are mandatory.
```
usage: init_model.py [-h] [--segment_length SEGMENT_LENGTH]
                     [--overlap_length OVERLAP_LENGTH] [--sources SOURCES]
                     [--segment_strategy SEGMENT_STRATEGY]
                     [--max_segment_length MAX_SEGMENT_LENGTH] [--tar TAR]
                     [--workers WORKERS] [--epochs EPOCHS] [--device DEVICE]
                     [--batch_sampler BATCH_SAMPLER] [--batch_size BATCH_SIZE]
                     [--num_buckets NUM_BUCKETS]
                     [--dynamic_batch_size DYNAMIC_BATCH_SIZE] [--fs FS]
                     [--ema EMA] [--ema_decay EMA_DECAY]
                     [--ignore_checkpoint IGNORE_CHECKPOINT]
                     [--preload PRELOAD] [--ddp DDP] [--rank RANK]
                     [--use_wandb USE_WANDB] [--profile PROFILE]
                     [--val_metrics VAL_METRICS] [--val_period VAL_PERIOD]
                     [--use_amp USE_AMP] [--compile COMPILE]
                     [--save_on_epochs SAVE_ON_EPOCHS] [--seed SEED]
                     --train_path TRAIN_PATH --val_path VAL_PATH [-f]
                     [-n NAME]
                     {convtasnet,dccrn,ffnn,manner,metricganokd,metricganp,sgmsep,sgmsepm,idmse,idmselarge}
                     ...

initialize a model

positional arguments:
  {convtasnet,dccrn,ffnn,manner,metricganokd,metricganp,sgmsep,sgmsepm,idmse,idmselarge}
                        model architecture

options:
  -h, --help            show this help message and exit
  -f, --force           overwrite config file if already exists
  -n NAME, --name NAME  model name

dataset options:
  --segment_length SEGMENT_LENGTH
  --overlap_length OVERLAP_LENGTH
  --sources SOURCES
  --segment_strategy SEGMENT_STRATEGY
  --max_segment_length MAX_SEGMENT_LENGTH
  --tar TAR

trainer options:
  --workers WORKERS
  --epochs EPOCHS
  --device DEVICE
  --batch_sampler BATCH_SAMPLER
  --batch_size BATCH_SIZE
  --num_buckets NUM_BUCKETS
  --dynamic_batch_size DYNAMIC_BATCH_SIZE
  --fs FS
  --ema EMA
  --ema_decay EMA_DECAY
  --ignore_checkpoint IGNORE_CHECKPOINT
  --preload PRELOAD
  --ddp DDP
  --rank RANK
  --use_wandb USE_WANDB
  --profile PROFILE
  --val_metrics VAL_METRICS
  --val_period VAL_PERIOD
  --use_amp USE_AMP
  --compile COMPILE
  --save_on_epochs SAVE_ON_EPOCHS

extra options:
  --seed SEED
  --train_path TRAIN_PATH
  --val_path VAL_PATH
```

The model is then trained using the `scripts/train_model.py` script. Training options can be provided, which will override the parameters in the `config.yaml` file.
```
usage: train_model.py [-h] [-f] [--wandb_run_id WANDB_RUN_ID]
                      [--segment_length SEGMENT_LENGTH]
                      [--overlap_length OVERLAP_LENGTH] [--sources SOURCES]
                      [--segment_strategy SEGMENT_STRATEGY]
                      [--max_segment_length MAX_SEGMENT_LENGTH] [--tar TAR]
                      [--workers WORKERS] [--epochs EPOCHS] [--device DEVICE]
                      [--batch_sampler BATCH_SAMPLER]
                      [--batch_size BATCH_SIZE] [--num_buckets NUM_BUCKETS]
                      [--dynamic_batch_size DYNAMIC_BATCH_SIZE] [--fs FS]
                      [--ema EMA] [--ema_decay EMA_DECAY]
                      [--ignore_checkpoint IGNORE_CHECKPOINT]
                      [--preload PRELOAD] [--ddp DDP] [--rank RANK]
                      [--use_wandb USE_WANDB] [--profile PROFILE]
                      [--val_metrics VAL_METRICS] [--val_period VAL_PERIOD]
                      [--use_amp USE_AMP] [--compile COMPILE]
                      [--save_on_epochs SAVE_ON_EPOCHS] [--seed SEED]
                      [--train_path TRAIN_PATH] [--val_path VAL_PATH]
                      input

train a model

positional arguments:
  input                 model directory

options:
  -h, --help            show this help message and exit
  -f, --force           train even if already trained
  --wandb_run_id WANDB_RUN_ID
                        id of wandb run to resume

the following options supersede the config file:
  --segment_length SEGMENT_LENGTH
  --overlap_length OVERLAP_LENGTH
  --sources SOURCES
  --segment_strategy SEGMENT_STRATEGY
  --max_segment_length MAX_SEGMENT_LENGTH
  --tar TAR
  --workers WORKERS
  --epochs EPOCHS
  --device DEVICE
  --batch_sampler BATCH_SAMPLER
  --batch_size BATCH_SIZE
  --num_buckets NUM_BUCKETS
  --dynamic_batch_size DYNAMIC_BATCH_SIZE
  --fs FS
  --ema EMA
  --ema_decay EMA_DECAY
  --ignore_checkpoint IGNORE_CHECKPOINT
  --preload PRELOAD
  --ddp DDP
  --rank RANK
  --use_wandb USE_WANDB
  --profile PROFILE
  --val_metrics VAL_METRICS
  --val_period VAL_PERIOD
  --use_amp USE_AMP
  --compile COMPILE
  --save_on_epochs SAVE_ON_EPOCHS
  --seed SEED
  --train_path TRAIN_PATH
  --val_path VAL_PATH
```

Example:
```
$ python scripts/init_model.py --train_path data/datasets/train/<dataset_id>/ convtasnet
Initialized models/<model_id>/config.yaml
$ python scripts/train_model.py models/<model_id>/
```

The following files are then created next to the `config.yaml` file:
- `checkpoints/`: a sub-directory containing the model checkpoints
- `log_train.log`: a log file
- `losses.npz`: training and validation curves in NumPy format
- `training_curve.png`: a plot of the training and validation curves

## Testing models

You can evaluate a trained model using the `scripts/test_model.py` script.
```
usage: test_model.py [-h] -i INPUTS [INPUTS ...] -t TESTS [TESTS ...] [-f]
                     [--output_dir OUTPUT_DIR] [--cuda]
                     [--metrics METRICS [METRICS ...]] [--no_train_check]
                     [--best BEST] [--batch_size BATCH_SIZE]
                     [--workers WORKERS] [--ddp]

test a model

options:
  -h, --help            show this help message and exit
  -i INPUTS [INPUTS ...], --inputs INPUTS [INPUTS ...]
                        model directories or checkpoints
  -t TESTS [TESTS ...], --tests TESTS [TESTS ...]
                        test dataset paths
  -f, --force           test even if already tested
  --output_dir OUTPUT_DIR
                        where to write signals
  --cuda                run on GPU
  --metrics METRICS [METRICS ...]
                        metrics to evaluate with
  --no_train_check      test even if model is not trained
  --best BEST           metric to use for checkpoint selection
  --batch_size BATCH_SIZE
                        batch size
  --workers WORKERS     number of workers
  --ddp                 use DDP
```

Example:
```
python scripts/test_model.py -i models/<model_id>/ -t data/datasets/test/<dataset_id>/
```
This creates a `scores.hdf5` file in the model directory containing the objective metrics of the enhanced output mixtures and the unprocessed input mixtures.

To visualize the scores, you can use the `scripts/compare_models.py` script.

Example:
```
python scripts/compare_models.py -i models/<model_id_1>/ models/<model_id_2>/ -t data/datasets/test/<dataset_id>/
```

# Contributing

## Implementing new models

- Inherit from `brever.models.base.BreverBaseModel` and implement the `loss` and `_enhance` methods. Eventually override other methods if needed. See the docstrings in `brever/models/base.py` for details. See the implementation of the other models in `brever/models/` for examples.
- Add the model to `brever.models.base.ModelRegistry` using the `register` decorator. This will expose the model to the command line.
- Add default arguments and type hints to the `__init__` method of the model. This will allow for type checks and default values when initializing models from the command line.
- Create a default config file for the model in `config/models/`. You can do this automatically using `scripts/write_default_config.py`. Eventually replace the default options for the trainer and the dataset with values that make sense for your model.
- Implement tests in `tests/test_models.py` and `tests/test_training.py`.

## TODO

- Fix MetricGAN+
- Add CSIG, CBAK, COVL metrics
- Dynamically check for `audio/` or `audio.tar` in dataset directories
- Support `.wav` files. Eventually support arbitrary file extensions.
- Convert BRAS files from SOFA to WAV
- Add noise type selection for DEMAND and ARTE
- Make `download_databases.sh` cross-platform

# Docker

To build the image:
```
docker build -t brever:latest .
```

To start the container:
```
docker run -it --rm -v ./models:/brever/models -v ./data:/brever/data brever:latest
```

# Related publications

```bibtex
@article{gonzalez2023assessing,
    title={Assessing the Generalization Gap of Learning-Based Speech Enhancement Systems in Noisy and Reverberant Environments},
    author={Philippe Gonzalez and Tommy Sonne Alstrøm and Tobias May},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    volume={31},
    pages={3390-3403},
    year={2023},
    doi={10.1109/TASLP.2023.3318965},
}

@inproceedings{gonzalez2023batching,
    title={On Batching Variable Size Inputs for Training End-to-End Speech Enhancement Systems},
    author={Philippe Gonzalez and Tommy Sonne Alstrøm and Tobias May},
    booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing},
    pages={1--5},
    year={2023},
    doi={10.1109/ICASSP49357.2023.10097075},
}

@misc{gonzalez2023diffusion,
    title={Diffusion-Based Speech Enhancement in Matched and Mismatched Conditions Using a Heun-Based Sampler},
    author={Philippe Gonzalez and Zheng-Hua Tan and Jan Østergaard and Jesper Jensen and Tommy Sonne Alstrøm and Tobias May},
    year={2023},
    eprint={2312.02683},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
}

@misc{gonzalez2023investigating,
    title={Investigating the Design Space of Diffusion Models for Speech Enhancement},
    author={Philippe Gonzalez and Zheng-Hua Tan and Jan Østergaard and Jesper Jensen and Tommy Sonne Alstrøm and Tobias May},
    year={2023},
    eprint={2312.04370},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
}
```

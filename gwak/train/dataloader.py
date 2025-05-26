import h5py
import yaml
import logging
import argparse
import numpy as np
from math import ceil
from pathlib import Path
from typing import Callable, List, Optional, Union

import wandb
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_observed_strain, get_ifo_geometry, compute_network_snr

from torch.distributions.uniform import Uniform
from ml4gw.distributions import Cosine

from gwak import data
from abc import ABC
import copy
import sys

class CleanGlitchPairedDataset(torch.utils.data.IterableDataset):
    def __init__(self, clean_ds, glitch_ds):
        self.clean_ds = clean_ds
        self.glitch_ds = glitch_ds

    def __len__(self):
        return len(self.clean_ds)
    
    def __iter__(self):
        for clean_x, glitch_x in zip(iter(self.clean_ds), iter(self.glitch_ds)):
            yield clean_x, glitch_x


class TimeSlidesDataloader(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: Path,
        sample_rate: int,
        kernel_length: float, # how many data points
        psd_length: int, # for whitening
        fduration: int,
        fftlength: int,
        batch_size: int,
        batches_per_epoch: int,
        num_workers: int,
        glitch_root: Path,
        data_saving_file: Path = None,
        ifos: str = 'HL'
    ):
        super().__init__()
        self.train_fnames, self.val_fnames, self.test_fnames = self.train_val_test_split(data_dir)
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.fduration = fduration
        self.fftlength = fftlength
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers
        self.glitch_root = glitch_root
        self.data_saving_file = data_saving_file
        if type(ifos) == list:
            self.ifos = ifos
        else:
            self.ifos = [f'{ifo}1' for ifo in ifos]
            if self.ifos not in [['H1', 'L1'], ['H1', 'V1'], ['L1', 'V1'], ['H1', 'L1', 'V1']]:
                print(f"Unrecognized ifo configuration {self.ifos}, please specify a valid one")
                sys.exit(1)
        if self.data_saving_file is not None:
            Path(self.data_saving_file.parents[0]).mkdir(parents=True, exist_ok=True)
            self.data_group = h5py.File(self.data_saving_file, "w")

        self._logger = self.get_logger()

    def train_val_test_split(self, data_dir, val_split=0.1, test_split=0.1):
        all_files = list(Path(data_dir).glob('*.h5')) + list(Path(data_dir).glob('*.hdf5'))
        all_files = sorted(all_files)
        n_all_files = len(all_files)
        # adding handling for the case where data_dir has subdirs for train/test/val
        if n_all_files == 0:
            subdirs = list(Path(data_dir).glob('*'))
            train, test, val = None, None, None
            for subdir in subdirs:
                subdir_files = list(subdir.glob('*.h5')) + list(subdir.glob('*.hdf5'))
                if subdir.stem == 'train':
                    train = subdir_files
                elif subdir.stem == 'test':
                    test = subdir_files
                elif subdir.stem == 'val':
                    val = subdir_files
            if train is None:
                print("You passed a directory with no h5 files and no train/ subdir!")
                sys.exit(1)
            elif test is None and val is None:
                print("You passed a directory with no test/ or val/ subdir!")
                sys.exit(1)
            elif test is None and val is not None:
                print("You passed a directory with a val/ directory but no test/, setting test set = val set")
                test = val
            elif test is not None and val is None:
                print("You passed a directory with a test/ directory but no val/, setting val set = test set")
                val = test
            return train, val, test
        else:
            n_train_files = int(n_all_files * (1 - val_split - test_split))
            n_val_files = int(n_all_files * val_split)
            return all_files[:n_train_files], all_files[n_train_files:n_train_files+n_val_files], all_files[n_train_files+n_val_files:]

    def make_dataset(self, fnames, coincident, mode):
        return Hdf5TimeSeriesDataset(
            fnames,
            channels=self.ifos,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            psd_length=self.psd_length,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=coincident,
            mode=mode,
            glitch_root=self.glitch_root,
            ifos=self.ifos,
        )

    def setup(self, stage=None):
        if stage in ("fit", None):

            train_clean_dataset = self. make_dataset(
                self.train_fnames, 
                coincident=False, 
                mode="clean"
            )

            val_clean_dataset = self. make_dataset(
                self.val_fnames, 
                coincident=False, 
                mode="clean"
            )

            train_glitch_dataset = self. make_dataset(
                self.train_fnames, 
                coincident=True, 
                mode="glitch"
            )

            val_glitch_dataset = self. make_dataset(
                self.val_fnames, 
                coincident=True, 
                mode="glitch"
            )

            # Wrap both datasets in the paired dataset
            self.train_paired_dataset = CleanGlitchPairedDataset(
                train_clean_dataset, 
                train_glitch_dataset
            )

            self.val_paired_dataset = CleanGlitchPairedDataset(
                val_clean_dataset, 
                val_glitch_dataset
            )

        if stage == "test":
            test_clean_dataset = self. make_dataset(
                self.test_fnames, 
                coincident=False, 
                mode="clean"
            )

            test_glitch_dataset = self. make_dataset(
                self.test_fnames, 
                coincident=True, 
                mode="glitch"
            )

            self.test_paired_dataset = CleanGlitchPairedDataset(
                test_clean_dataset, 
                test_glitch_dataset
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_paired_dataset, batch_size=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_paired_dataset, batch_size=None)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_paired_dataset, batch_size=None)

    def get_logger(self):
        logger_name = 'GwakBaseDataloader'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger

    def whiten(self, batch):

        # split batch into psd data and data to be whitened
        split_size = int((self.kernel_length + self.fduration) * self.sample_rate)
        splits = [batch.size(-1) - split_size, split_size]
        psd_data, batch = torch.split(batch, splits, dim=-1)

        # psd estimator
        # takes tensor of shape (batch_size, num_ifos, psd_length)
        spectral_density = SpectralDensity(
            self.sample_rate,
            self.fftlength,
            average = 'median'
        )
        spectral_density = spectral_density.to('cuda') if torch.cuda.is_available() else spectral_density

        # calculate psds
        psds = spectral_density(psd_data.double())

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        )
        whitener = whitener.to('cuda') if torch.cuda.is_available() else whitener

        whitened = whitener(batch.double(), psds.double())

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            clean_batch, glitch_batch = batch
            # Mix the clean and glitch data
            clean_batch[int(self.batch_size/2):] = glitch_batch[:int(self.batch_size/2)]
            # whiten
            batch = self.whiten(clean_batch)

            if self.trainer.training and (self.data_saving_file is not None):

                step_name = f"Training/Step_{self.trainer.global_step:06d}_BK"
                self.data_group.create_dataset(step_name, data = batch.cpu())

            if self.trainer.validating and (self.data_saving_file is not None):

                step_name = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                self.data_group.create_dataset(step_name, data = batch.cpu())

            return batch

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):
        pass

    def inject(self, batch, waveforms):
        pass


class GwakBaseDataloader(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: Path,
        sample_rate: int,
        kernel_length: float, # how many data points
        psd_length: int, # for whitening
        fduration: int,
        fftlength: int,
        batch_size: int,
        batches_per_epoch: int,
        num_workers: int,
        glitch_root: Path = None,
        data_saving_file: Path = None,
        ifos: str = 'HL'
    ):
        super().__init__()
        self.train_fnames, self.val_fnames, self.test_fnames = self.train_val_test_split(data_dir)
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.psd_length = psd_length
        self.fduration = fduration
        self.fftlength = fftlength
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_workers = num_workers
        self.glitch_root = glitch_root
        self.data_saving_file = data_saving_file
        if type(ifos) == list:
            self.ifos = ifos
        else:
            self.ifos = [f'{ifo}1' for ifo in ifos]
            if self.ifos not in [['H1', 'L1'], ['H1', 'V1'], ['L1', 'V1'], ['H1', 'L1', 'V1']]:
                print(f"Unrecognized ifo configuration {self.ifos}, please specify a valid one")
                sys.exit(1)
        print("ifos are", self.ifos)
        print("data dir is", data_dir)

        if self.data_saving_file is not None:
            Path(self.data_saving_file.parents[0]).mkdir(parents=True, exist_ok=True)
            self.data_group = h5py.File(self.data_saving_file, "w")

        self._logger = self.get_logger()

        # define a config dictionary that we can manipulate downstream for signal-specific kwargs
        # to be used for e.g. additional kwargs for each signal type needed for waveform generation
        self.config = {
            "sample_rate": sample_rate,
            "kernel_length": kernel_length,
            "psd_length": psd_length,
            "fduration": fduration,
            "fftlength": fftlength,
        }



    def train_val_test_split(self, data_dir, val_split=0.1, test_split=0.1):
        all_files = list(Path(data_dir).glob('*.h5')) + list(Path(data_dir).glob('*.hdf5'))
        all_files = sorted(all_files)
        n_all_files = len(all_files)
        # adding handling for the case where data_dir has subdirs for train/test/val
        if n_all_files == 0:
            subdirs = list(Path(data_dir).glob('*'))
            train, test, val = None, None, None
            for subdir in subdirs:
                subdir_files = list(subdir.glob('*.h5')) + list(subdir.glob('*.hdf5'))
                if subdir.stem == 'train':
                    train = subdir_files
                elif subdir.stem == 'test':
                    test = subdir_files
                elif subdir.stem == 'val':
                    val = subdir_files
            if train is None:
                print("You passed a directory with no h5 files and no train/ subdir!")
                sys.exit(1)
            elif test is None and val is None:
                print("You passed a directory with no test/ or val/ subdir!")
                sys.exit(1)
            elif test is None and val is not None:
                print("You passed a directory with a val/ directory but no test/, setting test set = val set")
                test = val
            elif test is not None and val is None:
                print("You passed a directory with a test/ directory but no val/, setting val set = test set")
                val = test
            return train, val, test
        else:
            n_train_files = int(n_all_files * (1 - val_split - test_split))
            n_val_files = int(n_all_files * val_split)
            return all_files[:n_train_files], all_files[n_train_files:n_train_files+n_val_files], all_files[n_train_files+n_val_files:]


    def train_dataloader(self):

        dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=self.ifos,
                kernel_length=self.kernel_length,
                fduration=self.fduration,
                psd_length=self.psd_length,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                batches_per_epoch=self.batches_per_epoch,
                coincident=False,
                mode='clean',
                glitch_root=self.glitch_root,
                ifos=self.ifos
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader

    def val_dataloader(self):

        dataset = Hdf5TimeSeriesDataset(
            self.val_fnames,
            channels=self.ifos,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            psd_length=self.psd_length,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            mode='clean',
            glitch_root=self.glitch_root,
            ifos=self.ifos
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )

        return dataloader

    def test_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.test_fnames,
            channels=self.ifos,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            psd_length=self.psd_length,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            mode='clean',
            glitch_root=self.glitch_root,
            ifos=self.ifos
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader

    def get_logger(self):
        logger_name = 'GwakBaseDataloader'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        return logger

    def whiten(self, batch):

        # split batch into psd data and data to be whitened
        split_size = int((self.kernel_length + self.fduration) * self.sample_rate)
        splits = [batch.size(-1) - split_size, split_size]
        psd_data, batch = torch.split(batch, splits, dim=-1)

        # psd estimator
        # takes tensor of shape (batch_size, num_ifos, psd_length)
        spectral_density = SpectralDensity(
            self.sample_rate,
            self.fftlength,
            average = 'median'
        )
        spectral_density = spectral_density.to('cuda') if torch.cuda.is_available() else spectral_density

        # calculate psds
        psds = spectral_density(psd_data.double())

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        )
        whitener = whitener.to('cuda') if torch.cuda.is_available() else whitener

        whitened = whitener(batch.double(), psds.double())

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch
            # inject waveforms; maybe also whiten data preprocess etc..
            batch = self.whiten(batch)

            if self.trainer.training and (self.data_saving_file is not None):

                step_name = f"Training/Step_{self.trainer.global_step:06d}_BK"
                self.data_group.create_dataset(step_name, data = batch.cpu())

            if self.trainer.validating and (self.data_saving_file is not None):

                step_name = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                self.data_group.create_dataset(step_name, data = batch.cpu())

            return batch

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):
        pass

    def inject(self, batch, waveforms):
        pass


class SignalDataloader(GwakBaseDataloader):
    def __init__(
        self,
        signal_classes, # string names of signal class(es) desired
        priors, # priors for each class
        waveforms, # waveforms for each class
        extra_kwargs, # any additional kwargs a particular signal needs to generate waveforms (e.g. ringdown duration)
        cache_dir: Optional[str] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_classes = signal_classes if type(signal_classes) == list else [signal_classes]
        self.num_classes = len(signal_classes)
        self.waveforms = waveforms if type(waveforms) == list else [waveforms]
        self.priors = priors if type(priors) == list else [priors]
        self.extra_kwargs = extra_kwargs if type(extra_kwargs) == list else [extra_kwargs]
        self._is_glitch = (self.signal_classes == ["Glitch"]
                           or (self.num_classes == 1
                               and self.signal_classes[0] == "Glitch"))
        self.cache_dir = cache_dir

        self.signal_configs = []
        for i in range(len(signal_classes)):
            signal_config = copy.deepcopy(self.config)
            if extra_kwargs[i] is not None:
                signal_config.update(extra_kwargs[i])
            self.signal_configs.append(signal_config)

        # Projection parameters
        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        self.phic_prior = Uniform(0, 2 * torch.pi)

        # CCSN second-derivitive waveform data
        file_path = Path(__file__).resolve()
        self.ccsn_dict = load_h5_as_dict(
            chosen_signals=file_path.parents[1] / "data/configs/ccsn.yaml",
            source_file=Path("/home/hongyin.chen/Data/3DCCSN_PREMIERE/Resampled")
        )

        # compute number of events to generate per class per batch
        self.num_per_class = self.num_classes * [self.batch_size//self.num_classes]
        for i in range(self.batch_size % self.num_classes):
            self.num_per_class[i] += 1

        # save correspondence between numerical labels and signal names
        # convention is label 1 = first signal, label 2 = second signal, etc.
        class_labels = [i+1 for i in range(self.num_classes)]
        if self.data_saving_file is not None:
            self.data_group.create_dataset("class_label_numbers",data=np.array(class_labels))
            self.data_group["class_label_names"] = self.signal_classes

        self.generate_waveforms_ccsn = CCSN_Injector(
            ifos=self.ifos,
            signals_dict=self.ccsn_dict,
            sample_rate=self.sample_rate,
            sample_duration=0.5,
            buffer_duration=2.5
        )

    def make_dataset(self, fnames, coincident, mode):
        return Hdf5TimeSeriesDataset(
            fnames,
            channels=self.ifos,
            kernel_length=self.kernel_length,
            fduration=self.fduration,
            psd_length=self.psd_length,
            sample_rate=self.sample_rate,
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=coincident,
            mode=mode,
            glitch_root=self.glitch_root,
            ifos=self.ifos,
            cache_dir=self.cache_dir,
        )

    def setup(self, stage=None):
        if stage in ("fit", None):

            train_clean_dataset = self. make_dataset(
                self.train_fnames,
                coincident=False,
                mode="clean"
            )

            val_clean_dataset = self. make_dataset(
                self.val_fnames,
                coincident=False,
                mode="clean"
            )

            train_glitch_dataset = self. make_dataset(
                self.train_fnames,
                coincident=True,
                mode="glitch"
            )

            val_glitch_dataset = self. make_dataset(
                self.val_fnames,
                coincident=True,
                mode="glitch"
            )

            # Wrap both datasets in the paired dataset
            self.train_paired_dataset = CleanGlitchPairedDataset(
                train_clean_dataset,
                train_glitch_dataset
            )

            self.val_paired_dataset = CleanGlitchPairedDataset(
                val_clean_dataset,
                val_glitch_dataset
            )

        if stage == "test":
            if self.glitch_root is not None:
                test_glitch_dataset = self.make_dataset(
                    self.test_fnames, 
                    coincident=True, 
                    mode="glitch"
                )
                self.test_paired_dataset = CleanGlitchPairedDataset(
                    test_clean_dataset, 
                    test_glitch_dataset
                )
            else:
                self.test_paired_dataset = self.make_dataset(
                self.test_fnames, 
                coincident=False, 
                mode="raw"
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_paired_dataset, batch_size=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_paired_dataset, batch_size=None)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_paired_dataset, batch_size=None)

    def generate_waveforms(self, batch_size, parameters=None, ras=None, decs=None):
        all_responses = []
        output_params = [] if parameters is None else parameters
        output_ras = [] if ras is None else ras
        output_decs = [] if decs is None else decs
        output_phics = []
        for i, signal_class in enumerate(self.signal_classes):
            if signal_class == 'BBH':
                responses, params, ra, dec, phic = generate_waveforms_bbh(
                    self.num_per_class[i],
                    self.priors[i],
                    self.waveforms[i],
                    self,
                    self.signal_configs[i],
                    self.ifos,
                    parameters=parameters[i] if parameters is not None else None,
                    ra=ras[i] if ras is not None else None,
                    dec=decs[i] if decs is not None else None
                )
            if signal_class == "CCSN":
                responses, dec, phic = self.generate_waveforms_ccsn(
                    total_counts=self.num_per_class[i]
                )
                params, ra = None, None

            elif signal_class in ["Background", "Glitch"]:
                responses, params, ra, dec, phic = None, None, None, None, None

            else:
                responses, params, ra, dec, phic = generate_waveforms_standard(
                    self.num_per_class[i],
                    self.priors[i],
                    self.waveforms[i],
                    self,
                    self.signal_configs[i],
                    self.ifos,
                    parameters=parameters[i] if parameters is not None else None,
                    ra=ras[i] if ras is not None else None,
                    dec=decs[i] if decs is not None else None
                )

            all_responses.append(responses)
            if parameters is None:
                output_params.append(params)
            if ras is None:
                output_ras.append(ra)
            if decs is None:
                output_decs.append(dec)
            output_phics.append(phic)

        return all_responses, output_params, output_ras, output_decs, output_phics

    def inject(self, batch, waveforms, output_snrs = False):

        # split batch into psd data and data to be whitened
        split_size = int((self.kernel_length + self.fduration) * self.sample_rate)
        splits = [batch.size(-1) - split_size, split_size]
        psd_data, batch = torch.split(batch, splits, dim=-1)

        # psd estimator
        # takes tensor of shape (batch_size, num_ifos, psd_length)
        spectral_density = SpectralDensity(
            self.sample_rate,
            self.fftlength,
            average = 'median'
        )
        spectral_density = spectral_density.to('cuda') if torch.cuda.is_available() else spectral_density

        # calculate psds
        psds = spectral_density(psd_data.double())
        if torch.any(torch.isnan(psds)):
            self._logger.info('psds fucked')

        # Waveform padding
        if waveforms is not None:
            inj_len = waveforms.shape[-1]
            window_len = splits[1]

            # new implementation: center the window of length window_len
            # about the max amplitude point in the signal waveform
            max_channel = torch.max(torch.abs(waveforms),dim=1)[0] # get largest of [H1,L1] for every timestep
            imax = torch.argmax(max_channel,dim=1).cpu().numpy() # get index of largest time value
            waveforms = list(torch.unbind(waveforms,dim=0))
            left = window_len//2
            right = window_len - left
            for i in range(len(waveforms)):
                left_idx = imax[i] - left
                right_idx = imax[i] + right - 1
                left_pad = max(0,-left_idx)
                right_pad = max(0, -((inj_len-1)-right_idx))
                new_left_idx = max(0,left_idx)
                new_right_idx = left_pad + right_idx
                waveforms[i] = F.pad(waveforms[i], (left_pad, right_pad), mode='constant', value=0)[..., new_left_idx:new_right_idx+1].unsqueeze(0)
            waveforms = torch.cat(waveforms,dim=0)

            if torch.any(torch.isnan(waveforms)):
                self._logger.info('centered waveforms fucked')

            injected = batch + waveforms
        else:
            injected = batch

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        )
        whitener = whitener.to('cuda') if torch.cuda.is_available() else whitener

        whitened = whitener(injected.double(), psds.double())
        if torch.any(torch.isnan(whitened)):
            self._logger.info('whitened fucked before dividing by std')

        psd_resample_size = 1+injected.shape[-1]//2 if injected.shape[-1] % 2 == 0 else (injected.shape[-1]+1)//2
        psds_resampled = F.interpolate(psds.double(), size=psd_resample_size, mode='linear', align_corners=False)

        snrs = torch.zeros(len(whitened)).to('cuda')
        if waveforms is not None:
            snrs = compute_network_snr(waveforms, psds_resampled, self.sample_rate)

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        if torch.any(torch.isnan(stds)):
            self._logger.info('stds fucked (nan)')
        if torch.any(stds == 0):
            self._logger.info('stds fucked (zero)')
        whitened = whitened / stds

        if torch.any(torch.isnan(whitened)):
            self._logger.info('whitened fucked')

        if output_snrs:
            return whitened, snrs
        else:
            return whitened

    def multiInject(self,waveforms,batch):
        sub_batches = []
        idx_lo = 0
        for i in range(self.num_classes):
            sub_batches.append(
                self.inject(batch[idx_lo:idx_lo+self.num_per_class[i]], waveforms[i]))
            idx_lo += self.num_per_class[i]
            if torch.any(torch.isnan(sub_batches[-1])):
                self._logger.info(f'had a nan batch with class {self.signal_classes[i]}')
        batch = torch.cat(sub_batches)
        return batch

    def multiInject_SNR(self,waveforms,batch):
        sub_batches = []
        sub_batches_snr = []
        idx_lo = 0
        for i in range(self.num_classes):
            whitened, snrs = self.inject(batch[idx_lo:idx_lo+self.num_per_class[i]], waveforms[i], output_snrs=True)
            sub_batches.append(whitened)
            sub_batches_snr.append(snrs)
            #sub_batches.append(self.inject(batch[idx_lo:idx_lo+self.num_per_class[i]], waveforms[i]))
            idx_lo += self.num_per_class[i]
        batch = torch.cat(sub_batches)
        snrs = torch.cat(sub_batches_snr)
        return batch, snrs

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            # raw_batch, clean_batch, glitch_batch = batch
            clean_batch, glitch_batch = batch

            # generate waveforms (method also returns the params used to generate the waveforms; these are not used in vanilla loader but useful for augmentation loader)
            waveforms, params, ras, decs, phics = self.generate_waveforms(clean_batch.shape[0])

            for wf in waveforms:
                if wf is None:
                    continue
                if torch.any(torch.isnan(wf)):
                    self._logger.info('waveforms fucked')
            if torch.any(torch.isnan(clean_batch)):
                self._logger.info('batch fucked')
            # inject waveforms; maybe also whiten data preprocess etc..
            _clean_batch = self.multiInject(waveforms, clean_batch)
            glitch_batch = self.multiInject(waveforms, glitch_batch)
            while torch.any(torch.isnan(_clean_batch)):
                self._logger.info('batch fucked after inject, regenerating')
                waveforms, params, ras, decs, phics = self.generate_waveforms(_clean_batch.shape[0])
                _clean_batch = self.multiInject(waveforms, clean_batch)
                
            clean_batch = _clean_batch
            labels = torch.cat([(i+1)*torch.ones(self.num_per_class[i]) for i in range(self.num_classes)]).to('cuda')
            labels = labels.to('cuda') if torch.cuda.is_available() else labels

            glitch_mask = (torch.where(labels == self.signal_classes.index("Glitch")+1))[0]
            glitch_chunk = glitch_batch[:glitch_mask.shape[0]]
            clean_batch[glitch_mask] = glitch_chunk
            perm = torch.randperm(clean_batch.size(0)).to('cuda') if torch.cuda.is_available() else perm
            batch = clean_batch[perm]
            labels = labels[perm]

            if self.trainer.training and (self.data_saving_file is not None):
                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                for i,cls in enumerate(self.signal_classes):
                    bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK{cls}"
                    inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ{cls}"
                    label_step = f"Training/Step_{self.trainer.global_step:06d}_LABEL{cls}"
                    prem_mask = torch.where(labels == self.signal_classes.index(cls) + 1)

                    self.data_group.create_dataset(bk_step, data = batch[prem_mask].cpu())
                    self.data_group.create_dataset(label_step, data = labels[prem_mask].cpu())
                    if waveforms[i] is not None:
                        self.data_group.create_dataset(inj_step, data = waveforms[i].cpu())

            if self.trainer.validating and (self.data_saving_file is not None):
                for i,cls in enumerate(self.signal_classes):
                    bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK{cls}"
                    inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ{cls}"
                    label_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_LAB{cls}"
                    prem_mask = torch.where(labels == self.signal_classes.index(cls) + 1)

                    self.data_group.create_dataset(bk_step, data = batch[prem_mask].cpu())
                    self.data_group.create_dataset(label_step, data = labels[prem_mask].cpu())
                    if waveforms[i] is not None:
                        self.data_group.create_dataset(inj_step, data = waveforms[i].cpu())

            return batch, labels


class AugmentationSignalDataloader(SignalDataloader):
    def __init__(
            self,
            ra_prior=None,
            dec_prior=None,
            sky_location_augmentation=True,
            distance_augmentation=False,
            tc_augmentation=False,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        # override default priors if provided
        if ra_prior is not None:
            self.ra_prior = ra_prior
        if dec_prior is not None:
            self.dec_prior = dec_prior

        self.sky_location_augmentation = sky_location_augmentation
        self.distance_augmentation = distance_augmentation
        self.tc_augmentation = tc_augmentation

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch

            # generate waveforms with one set of parameters
            waveforms_aug0, params, ras, decs, phics = self.generate_waveforms(batch.shape[0])

            # resample various parameters depending on augmentation settings
            if self.sky_location_augmentation:
                for i in range(self.num_classes):
                    ras[i] = self.ra_prior.sample((self.num_per_class[i],))
                    decs[i] = self.dec_prior.sample((self.num_per_class[i],))
            if self.distance_augmentation:
                for i in range(self.num_classes):
                    params[i]['distance'] = self.priors[i].sample(self.num_per_class[i])['distance']
            if self.tc_augmentation:
                pass # not sure what this one is meant to be, but it wasn't implemented yet

            # generate another set of waveforms with augmented parameters
            waveforms_aug1, _, _, _, _ = self.generate_waveforms(batch.shape[0], parameters=params, ras=ras, decs=decs)

            # inject waveforms; maybe also whiten data preprocess etc..

            batch_aug0 = self.multiInject(batch, waveforms_aug0)
            batch_aug1 = self.multiInject(batch, waveforms_aug1)
            batch = torch.stack([batch_aug0, batch_aug1])
            labels = torch.cat([(i+1)*torch.ones(self.num_per_class[i]) for i in range(self.num_classes)])
            labels = labels.to('cuda') if torch.cuda.is_available() else labels

            if self.trainer.training and (self.data_saving_file is not None):

                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"
                label_step = f"Training/Step_{self.trainer.global_step:06d}_LABEL"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())
                self.data_group.create_dataset(label_step, data = labels.cpu())

            if self.trainer.validating and (self.data_saving_file is not None):

                bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ"
                label_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_LAB"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())
                self.data_group.create_dataset(label_step, data = labels.cpu())

            return batch, labels



def generate_waveforms_standard(
    batch_size,
    prior,
    waveform,
    loader,
    config,
    ifos,
    parameters=None,
    ra=None,
    dec=None
):
    # get detector orientations
    tensors, vertices = get_ifo_geometry(*ifos)

    # sample from prior and generate waveforms
    if parameters is None:
        parameters = prior.sample(batch_size) # dict[str, torch.tensor]
    if ra is None:
        ra = loader.ra_prior.sample((batch_size,))
    if dec is None:
        dec = loader.dec_prior.sample((batch_size,))
    phic = loader.phic_prior.sample((batch_size,))

    cross, plus = waveform(**parameters) 

    if loader.signal_classes == ['BBH']:
        cross, plus = torch.fft.irfft(cross)* config['sample_rate'], torch.fft.irfft(plus) * config['sample_rate']       
    # compute detector responses
    responses = compute_observed_strain(
        dec,
        phic,
        ra,
        tensors,
        vertices,
        config['sample_rate'],
        cross=cross.float(),
        plus=plus.float()
    )
    responses = responses.to('cuda') if torch.cuda.is_available() else responses

    return responses, parameters, ra, dec, phic

def generate_waveforms_bbh(
    batch_size,
    prior,
    waveform,
    loader,
    config,
    ifos,
    parameters=None,
    ra=None,
    dec=None
):
    # get detector orientations
    tensors, vertices = get_ifo_geometry(*ifos)

    if parameters is None:
        # sample from prior and generate waveforms
        parameters = prior.sample(batch_size) # dict[str, torch.tensor]
    if ra is None:
        ra = loader.ra_prior.sample((batch_size,))
    if dec is None:
        dec = loader.dec_prior.sample((batch_size,))

    cross, plus = waveform(**parameters)
    cross, plus = torch.fft.irfft(cross), torch.fft.irfft(plus)
    # Normalization
    cross *= config['sample_rate']
    plus *= config['sample_rate']

    # roll the waveforms to join
    # the coalescence and ringdown
    ringdown_size = int(config['ringdown_duration'] * config['sample_rate'])
    cross = torch.roll(cross, -ringdown_size, dims=-1)
    plus = torch.roll(plus, -ringdown_size, dims=-1)


    # compute detector responses
    responses = compute_observed_strain(
        # parameters['dec'],
        dec,
        parameters['phic'], # psi
        # parameters['ra'],
        ra,
        tensors,
        vertices,
        config['sample_rate'],
        cross=cross.float(),
        plus=plus.float()
    )
    responses = responses.to('cuda') if torch.cuda.is_available() else responses

    return responses, parameters, ra, dec, parameters['phic']


def load_h5_as_dict(
    chosen_signals: Path,
    source_file: Path
)-> dict:
    """Open up a buffer to load in different CCSN wavefroms.

    Args:
        chosen_signals (Path): A file with names of each wavefrom.
        source_file (Path): The path that contains reasmpled raw waveform.

    Returns:
        dict: Time and resampled SQDM of Each waveform
    """
    with open(chosen_signals) as f:
        selected_ccsn = yaml.load(f, Loader=yaml.SafeLoader)

    source_file = Path(source_file)

    grand_dict = {}
    ccsn_list = []

    for key in selected_ccsn.keys():

        for name in selected_ccsn[key]:
            ccsn_list.append(f"{key}/{name}")

    for name in ccsn_list:

        with h5py.File(source_file/ f'{name}.h5', 'r', locking=False) as h:

            time = np.array(h['time'][:])
            quad_moment = h['quad_moment'][:]

        grand_dict[name] =  [time, quad_moment]

    return grand_dict

def get_hp_hc_from_q2ij(
    q2ij,
    theta: np.ndarray,
    phi: np.ndarray
):

    '''
    The orientation of GW emition is given by theta, phi
    '''

    hp =\
        q2ij[:,0,0]*(np.cos(theta)**2*np.cos(phi)**2 - np.sin(phi)**2).reshape(-1, 1)\
        + q2ij[:,1,1]*(np.cos(theta)**2*np.sin(phi)**2 - np.cos(phi)**2).reshape(-1, 1)\
        + q2ij[:,2,2]*(np.sin(theta)**2).reshape(-1, 1)\
        + q2ij[:,0,1]*(np.cos(theta)**2*np.sin(2*phi) - np.sin(2*phi)).reshape(-1, 1)\
        - q2ij[:,1,2]*(np.sin(2*theta)*np.sin(phi)).reshape(-1, 1)\
        - q2ij[:,2,0]*(np.sin(2*theta)*np.cos(phi)).reshape(-1, 1)

    hc = 2*(
        - q2ij[:,0,0]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,1,1]*(np.cos(theta)*np.sin(phi)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,0,1]*(np.cos(theta)*np.cos(2*phi)).reshape(-1, 1)
        - q2ij[:,1,2]*(np.sin(theta)*np.cos(phi)).reshape(-1, 1)
        + q2ij[:,2,0]*(np.sin(theta)*np.sin(phi)).reshape(-1, 1)
    )

    return hp, hc

def padding(
    time,
    hp,
    hc,
    distance,
    sample_kernel = 3,
    sample_rate = 4096,
    time_shift = -0.15, # shift zero to distination time
):

    # Two polarization
    signal = np.zeros([hp.shape[0], 2, int(sample_kernel * sample_rate)])

    half_kernel_idx = int(sample_kernel * sample_rate/2)
    time_shift_idx = int(time_shift * sample_rate)
    t0_idx = int(time[0] * sample_rate)

    start = half_kernel_idx + t0_idx + time_shift_idx
    end = half_kernel_idx + t0_idx + time.shape[0] + time_shift_idx

    signal[:, 0, start:end] = hp / distance.reshape(-1, 1)
    signal[:, 1, start:end] = hc / distance.reshape(-1, 1)

    return signal

from ml4gw import gw
from ml4gw.distributions import PowerLaw
from ml4gw.transforms import SnrRescaler
class CCSN_Injector:

    def __init__(
        self,
        ifos,
        signals_dict,
        sample_rate,
        sample_duration,
        buffer_duration = 4,
        off_set = 0.15,
        time_shift = -0.2
    ):

        self.tensors, self.vertices = gw.get_ifo_geometry(*ifos)

        self.signals = signals_dict
        self.ccsn_list = list(self.signals.keys())

        self.sample_rate = sample_rate
        self.sample_duration = sample_duration
        self.buffer_duration = buffer_duration
        self.off_set = off_set
        self.time_shift = time_shift
        self.kernel_length = int(sample_duration * sample_rate)
        self.buffer_length = int(buffer_duration * sample_rate)
        self.max_center_offset = int((buffer_duration/2 - sample_duration - off_set) * sample_rate)

        if off_set <= -time_shift:

            logging.info(f"Core bounce siganl may leak out of sample kernel by {-time_shift - off_set}")

    def __call__(
        self,
        total_counts # batch_size
    ):

        ccsn_num = len(self.ccsn_list)
        ccsn_sample = np.random.choice(ccsn_num, total_counts)
        ccsn_counts = np.eye(ccsn_num)[ccsn_sample].sum(0).astype("int")
        X = torch.empty((total_counts, 2, self.buffer_length))
        ccsne_agg_count = 0

        for name, count in zip(self.ccsn_list, ccsn_counts):

            time = self.signals[name][0]
            quad_moment = torch.Tensor(self.signals[name][1]) * 10

            theta = torch.Tensor(np.random.uniform(0, np.pi, count))
            phi = torch.Tensor(np.random.uniform(0, 2*np.pi, count))

            hp, hc = get_hp_hc_from_q2ij(
                quad_moment,
                theta=theta,
                phi=phi
            )

            hp_hc = padding(
                time,
                hp,
                hc,
                np.random.uniform(1, 10, count),
                sample_kernel = self.buffer_duration,
                sample_rate = self.sample_rate,
                time_shift = self.time_shift, # Core-bounce will be at here
            )

            X[ccsne_agg_count:ccsne_agg_count+count, :, :] = torch.Tensor(hp_hc)


            ccsne_agg_count += count
        X = X[:, :, 2048:-2048]
        dec_distro = Cosine()
        psi_distro = Uniform(0, np.pi)
        phi_distro = Uniform(0, 2 * np.pi)

        dec = dec_distro.rsample(total_counts)
        psi = psi_distro.sample((total_counts,))
        phi = phi_distro.sample((total_counts,))

        ht = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=X[:,0,:],
            cross=X[:,1,:]
        )

        ht = ht.to('cuda') if torch.cuda.is_available() else ht


        return ht, dec, phi
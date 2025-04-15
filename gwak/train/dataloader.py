import h5py
import logging
import argparse
import numpy as np
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
from ml4gw.gw import compute_observed_strain, get_ifo_geometry, compute_ifo_snr

from torch.distributions.uniform import Uniform
from ml4gw.distributions import Cosine

from gwak import data
from abc import ABC
import copy


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
        data_saving_file: Path = None
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
        self.data_saving_file = data_saving_file
        if self.data_saving_file is not None:

            Path(self.data_saving_file.parents[0]).mkdir(parents=True, exist_ok=True)
            self.data_group = h5py.File(self.data_saving_file, "w")

        self._logger = self.get_logger()

    def train_val_test_split(self, data_dir, val_split=0.1, test_split=0.1):

        all_files = list(Path(data_dir).glob('*.hdf5'))
        n_all_files = len(all_files)
        n_train_files = int(n_all_files * (1 - val_split - test_split))
        n_val_files = int(n_all_files * val_split)

        return all_files[:n_train_files], all_files[n_train_files:n_train_files+n_val_files], all_files[n_train_files+n_val_files:]

    def train_dataloader(self):

        dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=['H1', 'L1'],
                kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate),#int(self.sample_rate * self.sample_length),
                batch_size=self.batch_size,
                batches_per_epoch=self.batches_per_epoch,
                coincident=False,
            )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader

    def val_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.val_fnames,
            channels=['H1', 'L1'],
            kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate), # int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.test_fnames,
            channels=['H1', 'L1'],
            kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate), # int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
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

            # Time-slide L1 relative to H1 before whitening
            max_shift = batch.shape[-1] // 10  # 10% of signal length
            shifts = torch.randint(-max_shift, max_shift + 1, (batch.shape[0],), device=batch.device)
            for i, shift in enumerate(shifts):
                batch[i, 1] = torch.roll(batch[i, 1], shifts=shift.item(), dims=0)  # roll L1

            # whiten
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
        data_saving_file: Path = None
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
        self.data_saving_file = data_saving_file
        
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

        all_files = list(Path(data_dir).glob('*.hdf5'))
        n_all_files = len(all_files)
        n_train_files = int(n_all_files * (1 - val_split - test_split))
        n_val_files = int(n_all_files * val_split)

        return all_files[:n_train_files], all_files[n_train_files:n_train_files+n_val_files], all_files[n_train_files+n_val_files:]

    def train_dataloader(self):

        dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=['H1', 'L1'],
                kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate),#int(self.sample_rate * self.sample_length),
                batch_size=self.batch_size,
                batches_per_epoch=self.batches_per_epoch,
                coincident=False,
            )

        #pin_memory = isinstance(
        #    self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        #)
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader

    def val_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.val_fnames,
            channels=['H1', 'L1'],
            kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate), # int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
        )

        #pin_memory = isinstance(
        #    self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        #)
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=False
        )
        return dataloader
    
    def test_dataloader(self):
        dataset = Hdf5TimeSeriesDataset(
            self.test_fnames,
            channels=['H1', 'L1'],
            kernel_size=int((self.psd_length + self.fduration + self.kernel_length) * self.sample_rate), # int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
        )

        #pin_memory = isinstance(
        #    self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        #)
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
        signal_classes: list[str] | str, # string names of signal class(es) desired
        priors: list[Optional[data.BasePrior]] | Optional[data.BasePrior], # priors for each class
        waveforms: list[Optional[torch.nn.Module]] | Optional[torch.nn.Module], # waveforms for each class
        extra_kwargs: list[Optional[dict]] | Optional[dict], # any additional kwargs a particular signal needs to generate waveforms (e.g. ringdown duration)
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_classes = signal_classes if type(signal_classes) == list else [signal_classes]
        self.num_classes = len(signal_classes)
        self.waveforms = waveforms if type(waveforms) == list else [waveforms]
        self.priors = priors if type(priors) == list else [priors]
        self.extra_kwargs = extra_kwargs if type(extra_kwargs) == list else [extra_kwargs]
        self.signal_configs = []
        for i in range(len(signal_classes)):
            signal_config = copy.deepcopy(self.config)
            print(extra_kwargs[i],extra_kwargs[i] is None)
            if extra_kwargs[i] is not None:
                signal_config.update(extra_kwargs[i])
            self.signal_configs.append(signal_config)

        # Projection parameters
        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        self.phic_prior = Uniform(0, 2 * torch.pi)

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
                    parameters=parameters[i] if parameters is not None else None,
                    ra=ras[i] if ras is not None else None,
                    dec=decs[i] if decs is not None else None
                )
            elif signal_class == "Background":
                responses, params, ra, dec, phic = None, None, None, None, None
            else:
                responses, params, ra, dec, phic = generate_waveforms_standard(
                    self.num_per_class[i],
                    self.priors[i],
                    self.waveforms[i],
                    self,
                    self.signal_configs[i],
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

    def inject(self, batch, waveforms):

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

        # Waveform padding
        if waveforms is not None:
            inj_len = waveforms.shape[-1]
            window_len = splits[1]
            half = int((window_len - inj_len)/2)
            first_half, second_half = half, window_len - half - inj_len

            # old implementation - pad with zeros if inj_len < window_len
            # otherwise take the center chunk of waveform with length window_len
            #waveforms = F.pad(
            #    input=waveforms,
            #    pad=(first_half, second_half),
            #    mode='constant',
            #    value=0
            #)

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

        psds_resampled = F.interpolate(psds.double(), size=1537, mode='linear', align_corners=False)
        snrs = compute_ifo_snr(injected.double(), psds_resampled, 2048)

        # compute network SNR 
        snrs = snrs**2
        snrs = torch.sum(snrs, dim=-1)  ** 0.5

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened
    
    def multiInject(self,waveforms,batch):
        sub_batches = []
        idx_lo = 0
        for i in range(self.num_classes):
            sub_batches.append(self.inject(batch[idx_lo:idx_lo+self.num_per_class[i]], waveforms[i]))
            idx_lo += self.num_per_class[i]
        batch = torch.cat(sub_batches)
        return batch
    
    def multiInject_SNR(self,waveforms,batch):
        sub_batches = []
        sub_batches_snr = []
        idx_lo = 0
        for i in range(self.num_classes):
            print(i)
            whitened, snrs = self.inject(batch[idx_lo:idx_lo+self.num_per_class[i]], waveforms[i])
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
            [batch] = batch

            # generate waveforms (method also returns the params used to generate the waveforms; these are not used in vanilla loader but useful for augmentation loader)
            waveforms, params, ras, decs, phics = self.generate_waveforms(batch.shape[0])
            
            # inject waveforms; maybe also whiten data preprocess etc..    
            batch = self.multiInject(waveforms, batch)
            labels = torch.cat([(i+1)*torch.ones(self.num_per_class[i]) for i in range(self.num_classes)]).to('cuda')
            labels = labels.to('cuda') if torch.cuda.is_available() else labels

            perm = torch.randperm(batch.size(0)).to('cuda') if torch.cuda.is_available() else perm
            batch = batch[perm]
            labels = labels[perm]

            if self.trainer.training and (self.data_saving_file is not None):
                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                idx_curr = 0
                for i,cls in enumerate(self.signal_classes):
                    bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK{cls}"
                    inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ{cls}"
                    label_step = f"Training/Step_{self.trainer.global_step:06d}_LABEL{cls}"
                    data_range = slice(idx_curr,idx_curr+self.num_per_class[i])
                    idx_curr += self.num_per_class[i]

                    self.data_group.create_dataset(bk_step, data = batch[data_range].cpu())
                    self.data_group.create_dataset(inj_step, data = waveforms[i].cpu())
                    self.data_group.create_dataset(label_step, data = labels[data_range].cpu())

            if self.trainer.validating and (self.data_saving_file is not None):
                idx_curr = 0
                for i,cls in enumerate(self.signal_classes):
                    bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK{cls}"
                    inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ{cls}"
                    label_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_LAB{cls}"
                    data_range = slice(idx_curr,idx_curr+self.num_per_class[i])
                    idx_curr += self.num_per_class[i]

                    self.data_group.create_dataset(bk_step, data = batch[data_range].cpu())
                    self.data_group.create_dataset(inj_step, data = waveforms[i].cpu())
                    self.data_group.create_dataset(label_step, data = labels[data_range].cpu())

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
        
def generate_waveforms_standard(batch_size, prior, waveform, loader, config, parameters=None, ra=None, dec=None):
        # get detector orientations
        ifos = ['H1', 'L1']
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

def generate_waveforms_bbh(batch_size, prior, waveform, loader, config, parameters=None, ra=None, dec=None):
        # get detector orientations
        ifos = ['H1', 'L1']
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
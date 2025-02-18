import h5py
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

import wandb
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

import ml4gw
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.gw import compute_observed_strain, get_ifo_geometry
from bilby.gw.conversion import bilby_to_lalsimulation_spins

from torch.distributions.uniform import Uniform
from ml4gw.distributions import Cosine

from gwak import data
from abc import ABC

class GwakFileDataloader(pl.LightningDataModule):

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
        self.train_fnames, self.val_fnames = self.train_val_split(data_dir)
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

    def train_val_split(self, data_dir, val_split=0.2):

        all_files = list(Path(data_dir).glob('*.hdf5'))
        n_all_files = len(all_files)
        n_train_files = int(n_all_files * (1 - val_split))

        return all_files[:n_train_files], all_files[n_train_files:]

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
        ).to('cuda')

        # calculate psds
        psds = spectral_density(psd_data.double())

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        ).to('cuda')

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


class GlitchDataloader(GwakFileDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        self.train_fnames, self.val_fnames = self.train_val_split(data_dir)
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

    def train_val_split(self, data_dir, val_split=0.2):

        all_files = list(Path(data_dir).glob('*.hdf5'))
        n_all_files = len(all_files)
        n_train_files = int(n_all_files * (1 - val_split))

        return all_files[:n_train_files], all_files[n_train_files:]

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
        ).to('cuda')

        # calculate psds
        psds = spectral_density(psd_data.double())

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        ).to('cuda')

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
        prior: data.BasePrior,
        waveform: torch.nn.Module,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.waveform = waveform
        self.prior = prior

        # Projection parameters
        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        self.phic_prior = Uniform(0, 2 * torch.pi)

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):

        # get detector orientations
        ifos = ['H1', 'L1']
        tensors, vertices = get_ifo_geometry(*ifos)

        # sample from prior and generate waveforms
        parameters = self.prior.sample(batch_size) # dict[str, torch.tensor]

        ra = self.ra_prior.sample((batch_size,))
        dec = self.dec_prior.sample((batch_size,))
        phic = self.phic_prior.sample((batch_size,))

        cross, plus = self.waveform(**parameters)

        # compute detector responses
        responses = compute_observed_strain(
            dec,
            phic,
            ra,
            tensors,
            vertices,
            self.sample_rate,
            cross=cross.float(),
            plus=plus.float()
        ).to('cuda')


        return responses

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
        ).to('cuda')

        # calculate psds
        psds = spectral_density(psd_data.double())

        # Waveform padding
        inj_len = waveforms.shape[-1]
        window_len = splits[1]
        half = int((window_len - inj_len)/2)

        first_half, second_half = half, window_len - half - inj_len

        waveforms = F.pad(
            input=waveforms,
            pad=(first_half, second_half),
            mode='constant',
            value=0
        )

        injected = batch + waveforms * 100

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        ).to('cuda')

        whitened = whitener(injected.double(), psds.double())

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch

            # generate waveforms
            waveforms = self.generate_waveforms(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..

            batch = self.inject(batch, waveforms)

            if self.trainer.training and (self.data_saving_file is not None):

                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())

            if self.trainer.validating and (self.data_saving_file is not None):

                bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())

            return batch


class AugmentationSignalDataloader(GwakBaseDataloader):
    def __init__(
            self,
            signal_class: SignalDataloader,
            prior: data.BasePrior,
            ra_prior=None,
            dec_prior=None,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.signal_class = signal_class
        self.prior = prior

        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        #self.phic_prior = Uniform(0, 2 * torch.pi)

        if ra_prior is not None:
            self.ra_prior = ra_prior
        if dec_prior is not None:
            self.dec_prior = dec_prior

        self.sky_location_augmentation = True
        self.distance_augmentation = False
        self.tc_augmentation = False

    def generate_waveforms_augmented(self, batch_size):
        parameters = self.prior.sample(batch_size) # dict[str, torch.tensor]
        ra = self.ra_prior.sample((batch_size,))
        dec = self.dec_prior.sample((batch_size,))


        aug0 = self.signal_class.generate_waveforms(batch_size, parameters = parameters,
                                                    ra = ra, dec = dec)

        if self.sky_location_augmentation: #reroll
            ra = self.ra_prior.sample((batch_size,))
            dec = self.dec_prior.sample((batch_size,))

        if self.distance_augmentation:
            parameters_regenerated = self.prior.sample(batch_size)
            parameters['distance'] = parameters_regenerated['distance']

        if self.tc_augmentation:
            # prior sets everything to zero, so implement this later
            None

        # and do it again
        aug1 = self.signal_class.generate_waveforms(batch_size, parameters = parameters,
                                                    ra = ra, dec = dec)

        return torch.stack([aug0, aug1])

    def inject_augmented(self, batch, waveforms):
        aug_0, aug_1 = waveforms

        aug_0_injected = self.signal_class.inject(batch, aug_0)
        aug_1_injected = self.signal_class.inject(batch, aug_1)
        return torch.stack([aug_0_injected, aug_1_injected])

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch

            # generate waveforms
            waveforms = self.generate_waveforms_augmented(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..

            batch = self.inject_augmented(batch, waveforms)

            if self.trainer.training and (self.data_saving_file is not None):

                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())

            if self.trainer.validating and (self.data_saving_file is not None):

                bk_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_BK"
                inj_step = f"Validation/Step_{self.trainer.global_validation_step:06d}_INJ"

                self.data_group.create_dataset(bk_step, data = batch.cpu())
                self.data_group.create_dataset(inj_step, data = waveforms.cpu())

            return batch


class BBHDataloader(SignalDataloader):

    def __init__(
        self,
        ringdown_duration: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ringdown_size = int(ringdown_duration * self.sample_rate)

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):

        # get detector orientations
        ifos = ['H1', 'L1']
        tensors, vertices = get_ifo_geometry(*ifos)

        if parameters is None:
            # sample from prior and generate waveforms
            parameters = self.prior.sample(batch_size) # dict[str, torch.tensor]
        if ra is None:
            ra = self.ra_prior.sample((batch_size,))
        if dec is None:
            dec = self.dec_prior.sample((batch_size,))

        cross, plus = self.waveform(**parameters)
        cross, plus = torch.fft.irfft(cross), torch.fft.irfft(plus)
        # Normalization
        cross *= self.sample_rate
        plus *= self.sample_rate

        # roll the waveforms to join
        # the coalescence and ringdown
        cross = torch.roll(cross, -self.ringdown_size, dims=-1)
        plus = torch.roll(plus, -self.ringdown_size, dims=-1)

        # compute detector responses
        responses = compute_observed_strain(
            # parameters['dec'],
            dec,
            parameters['phic'], # psi
            # parameters['ra'],
            ra,
            tensors,
            vertices,
            self.sample_rate,
            cross=cross.float(),
            plus=plus.float()
        ).to('cuda')

        return responses

class MultiSignalDataloader(GwakBaseDataloader):

    def __init__(
        self,
        priors: list[data.BasePrior],
        waveforms: list[torch.nn.Module],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.waveform = waveforms
        self.prior = priors
        self.num_priors = len(priors)

        # Projection parameters
        self.ra_prior =  Uniform(0, 2*torch.pi)
        self.dec_prior = Cosine(-np.pi/2, torch.pi/2)
        self.phic_prior = Uniform(0, 2 * torch.pi)

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):

        # get detector orientations
        ifos = ['H1', 'L1']
        tensors, vertices = get_ifo_geometry(*ifos)

        # sample from prior and generate waveforms (equal number per prior)
        samples_per = self.num_priors * [batch_size//self.num_priors]
        for i in range(batch_size % self.num_priors):
            samples_per[i] += 1

        responses = []

        for i in range(self.num_priors):
            parameters = self.prior[i].sample(samples_per[i]) # dict[str, torch.tensor]

            ra = self.ra_prior.sample((samples_per[i],))
            dec = self.dec_prior.sample((samples_per[i],))
            phic = self.phic_prior.sample((samples_per[i],))

            cross, plus = self.waveform[i](**parameters)

            # compute detector responses
            responses_i = compute_observed_strain(
                dec,
                phic,
                ra,
                tensors,
                vertices,
                self.sample_rate,
                cross=cross.float(),
                plus=plus.float()
            )

            responses.append(responses_i)

        labels = torch.cat([(i+1)*torch.ones(samples_per[i]) for i in range(self.num_priors)]).to('cuda')
        responses = torch.cat(responses,dim=0).to('cuda')
        return responses, labels

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
        ).to('cuda')

        # calculate psds
        psds = spectral_density(psd_data.double())

        # Waveform padding
        inj_len = waveforms.shape[-1]
        window_len = splits[1]
        half = int((window_len - inj_len)/2)

        first_half, second_half = half, window_len - half - inj_len

        waveforms = F.pad(
            input=waveforms,
            pad=(first_half, second_half),
            mode='constant',
            value=0
        )

        injected = batch + waveforms * 100

        # create whitener
        whitener = Whiten(
            self.fduration,
            self.sample_rate,
            highpass = 30,
        ).to('cuda')

        whitened = whitener(injected.double(), psds.double())

        # normalize the input data
        stds = torch.std(whitened, dim=-1, keepdim=True)
        whitened = whitened / stds

        return whitened

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch

            # generate waveforms
            waveforms, labels = self.generate_waveforms(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..

            batch = self.inject(batch, waveforms)

            if self.trainer.training and (self.data_saving_file is not None):

                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"
                label_step = f"Training/Step_{self.trainer.global_step:06d}_LAB"

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
        

class MultiSignalDataloaderV2(GwakBaseDataloader):

    def __init__(
        self,
        signal_classes: list[SignalDataloader],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.signal_classes = signal_classes
        self.num_classes = len(signal_classes)

    def generate_waveforms(self, batch_size, parameters=None, ra=None, dec=None):
        # sample from prior and generate waveforms (equal number per prior)
        samples_per = self.num_classes * [batch_size//self.num_classes]
        for i in range(batch_size % self.num_classes):
            samples_per[i] += 1

        responses = []

        for i in range(self.classes):
            responses_i = self.signal_classes[i].generate_waveforms(samples_per[i])
            responses.append(responses_i)

        labels = torch.cat([(i+1)*torch.ones(samples_per[i]) for i in range(self.num_classes)]).to('cuda')
        responses = torch.cat(responses,dim=0).to('cuda')
        return responses, labels, samples_per

    def inject(self, batch, waveforms, samples_per):
        injections = []
        icurr = 0
        for i in range(self.num_classes):
            inj = self.signal_classes[i].inject(batch[icurr:icurr+samples_per[i]], waveforms[icurr:icurr+samples_per[i]])
            injections.append(inj)
            icurr += samples_per[i]

        injections = torch.cat(injections,dim=0).to('cuda')
        return injections

    def on_after_batch_transfer(self, batch, dataloader_idx):

        if self.trainer.training or self.trainer.validating or self.trainer.sanity_checking:
            # unpack the batch
            [batch] = batch

            # generate waveforms
            waveforms, labels, samples_per = self.generate_waveforms(batch.shape[0])
            # inject waveforms; maybe also whiten data preprocess etc..

            batch = self.inject(batch, waveforms, samples_per)

            if self.trainer.training and (self.data_saving_file is not None):

                # Set a warning that when the global_step exceed 1e6,
                # the data will have duplications.
                # Replace this with a data saving function.
                bk_step = f"Training/Step_{self.trainer.global_step:06d}_BK"
                inj_step = f"Training/Step_{self.trainer.global_step:06d}_INJ"
                label_step = f"Training/Step_{self.trainer.global_step:06d}_LAB"

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
import logging
from collections import OrderedDict


import lal
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import torch.distributions as torchdist
from tqdm import tqdm
import lal
from astropy import units as u
from ml4gw.distributions import Cosine, Sine
from ml4gw.waveforms.conversion import (
    bilby_spins_to_lalsim, 
    chirp_mass_and_mass_ratio_to_components
)
import math

class Constant:

    def __init__(self, val, tensor=True):
        self.val = val
        self.tensor = tensor

    def __repr__(self, cls):
        return self.__name__

    def sample(self, batch_size):

        if self.tensor:
            return torch.full(batch_size, self.val)

        return self.val


class BasePrior:

    def __init__(self):
        self.params = OrderedDict()
        self.sampled_params = OrderedDict()

    def sample(self, batch_size):

        self.sampled_params = OrderedDict()
        
        for k in self.params.keys():

            if type(self.params[k]) == Constant:
                    self.sampled_params[k] = self.params[k].val
            else:
                self.sampled_params[k] = self.params[k].sample((batch_size,))

        return self.sampled_params
        
class LogUniform(torchdist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(torchdist.Uniform(math.log(lb), math.log(ub)),
                                         torchdist.ExpTransform())

class SineGaussianHighFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()

        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(25, 75),
            frequency = Uniform(512, 1024),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01)
        )


class SineGaussianLowFrequency(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        self.params = OrderedDict(
            hrss = Uniform(1e-21, 2e-21),
            quality = Uniform(25, 75),
            frequency = Uniform(64, 512),
            phase = Uniform(0, 2 * torch.pi),
            eccentricity = Uniform(0, 0.01)
        )

class SineGaussianBBC(BasePrior):

    def __init__(self):
    # this is a super wide range for all the signals with converted amplitude to hrss here: https://git.ligo.org/bursts/burst-pipeline-benchmark/-/wikis/o4b_1/Waveforms-O4b-1
        super().__init__()
        self.params = OrderedDict(
            hrss = LogUniform(1e-24, 5e-20), 
            quality = Uniform(3, 700),
            frequency = Uniform(14, 3067),
            phase = Uniform(0, torch.pi),
            eccentricity = Uniform(0, 1)
        )

class GaussianBBC(BasePrior):

    def __init__(self):
    # this is a super wide range for all the signals with converted amplitude to hrss here: https://git.ligo.org/bursts/burst-pipeline-benchmark/-/wikis/o4b_1/Waveforms-O4b-1
        super().__init__()
        self.params = OrderedDict(
            hrss = LogUniform(1e-24, 1e-16), 
            polarization = Uniform(0, torch.pi),
            eccentricity = Uniform(0, 1)
        )

class WhiteNoiseBurstBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            frequency = Uniform(55, 1520),
            bandwidth = Uniform(32, 2048),
            eccentricity = Uniform(0, 1),
            phase = Uniform(0, torch.pi),
            int_hdot_squared = LogUniform(7.3e-41, 1.6e-35)
        )

class CuspBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-4.0 / 3.0),
            amplitude = Uniform(4.0e-22, 4.0e-21),
            f_high = Constant(1000)
        )

class KinkBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-5.0 / 3.0),
            amplitude = Uniform(1.4e-21, 1.4e-20),
            f_high = Constant(1000)
        )

class KinkkinkBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-2.0),
            amplitude = Uniform(4.7e-21, 4.7e-20),
            f_high = Constant(1000)
        )

class LAL_BBHPrior(BasePrior):
    
    def __init__(
        self,
        f_min=30,
        f_max=2048,
        duration=2, # duration of the time series
        f_ref=20.0
    ):

        self.priors = {}
        self.bilby_priors = {}
        self.spin_params = {}
        self.sampled_params = {}
        
        self.lal_keys = [
            "inclination", # Transformed inclination angle. (TensorType)
            "s1x", # Spin component x of the first BH. (TensorType)
            "s1y", # Spin component y of the first BH. (TensorType)
            "s1z", # Spin component z of the first BH. (TensorType)
            "s2x", # Spin component x of the second BH. (TensorType)
            "s2y", # Spin component y of the second BH. (TensorType)
            "s2z", # Spin component z of the second BH. (TensorType)
        ]

        # Frequency series in Hz. (TensorType)
        self.sampled_params["fs"] = torch.arange(f_min, f_max, 1 / duration) 
        
        # Chirp mass in solar masses. (TensorType)
        self.priors['chirp_mass'] = Uniform(15, 30) 
        
        # Mass ratio m1/m2. (TensorType)
        self.priors['mass_ratio'] = Uniform(0.5, 0.99) 
        
        # # Luminosity distance in Mpc.(TensorType)
        self.priors["distance"] = Uniform(50, 200) # dist_mpc
        
        # Coalescence time. (TensorType)
        self.priors["tc"] = Constant(0) 
        
        # Phase of the two polarlization
        self.priors['phic'] = Uniform(0, 2 * torch.pi) # psi
        
        # ----- Spin & incl parameters (Bilby parameters) -----
        # Inclination in bilby setup
        self.bilby_priors['theta_jn'] = Sine() 
        
        # Spin phase angle
        self.bilby_priors['phi_jl'] = Uniform(0, 2 * torch.pi) 
        
        # Primary object tilt
        self.bilby_priors['tilt_1'] = Sine(0, torch.pi) 
        
        # Secondary object tilt
        self.bilby_priors['tilt_2'] = Sine(0, torch.pi) 
        
        # Relative spin azimuthal angle
        self.bilby_priors['phi_12'] = Uniform(0, 2 * torch.pi) 
        
        # Primary dimensionless spin magnitude
        self.bilby_priors['a_1'] = Uniform(0, 0.99) 
        
        # Secondary dimensionless spin magnitude
        self.bilby_priors['a_2'] = Uniform(0, 0.99) 
        
        # Reference frequency in Hz. *****(float)*****
        self.sampled_params["f_ref"] = Constant(f_ref).sample((1,)).numpy() 
        
        # Uniform(0, 2*np.pi) # Reference phase. (TensorType) #(Bilby) Orbital phase
        self.bilby_priors['phiRef'] = Constant(0) 

        self.sample_keys = self.priors.keys()
        
    def sample(self, batch_size): # translator
        
        for key in self.priors.keys():
            
            self.sampled_params[key] = self.priors[key].sample((batch_size,))

        for key in self.bilby_priors.keys():
            
            self.spin_params[key] = self.bilby_priors[key].sample((batch_size,)) 
        
        mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(
            self.sampled_params['chirp_mass'],
            self.sampled_params['mass_ratio']
        )
        
        lal_spins = bilby_spins_to_lalsim(
            theta_jn=self.spin_params['theta_jn'], 
            phi_jl=self.spin_params['phi_jl'], 
            tilt_1=self.spin_params['tilt_1'], 
            tilt_2=self.spin_params['tilt_2'], 
            phi_12=self.spin_params['phi_12'], 
            a_1=self.spin_params['a_1'], 
            a_2=self.spin_params['a_2'], 
            mass_1=mass_1, 
            mass_2=mass_2, 
            f_ref=self.sampled_params['f_ref'][0], 
            phi_ref=self.spin_params['phiRef'], 
        )
        
        for i, key in enumerate(self.lal_keys):

            self.sampled_params[key] = lal_spins[i]
        
        return self.sampled_params


class BBHPrior(BasePrior):

    def __init__(self):
    # something with sample method that returns dict that maps
    # parameter name to tensor of parameter names
        super().__init__()
        # taken from bilby.gw.prior.BBHPriorDict()
        self.params = dict(
            mass_ratio = Uniform(0.5, 0.99), # Uniform(0.125, 1),
            chirp_mass = Uniform(15, 30), # Uniform(25, 100),
            theta_jn = Sine(),
            phase = Constant(0), # Uniform(0, 2 * torch.pi),
            a_1 = Uniform(0, 0.99),
            a_2 = Uniform(0, 0.99),
            tilt_1 = Sine(0, torch.pi),
            tilt_2 = Sine(0, torch.pi),
            phi_12 = Uniform(0, 2 * torch.pi),
            phi_jl = Uniform(0, 2 * torch.pi),
            reference_frequency = Constant(20.0, tensor=False), #Constant(50.0, tensor=False),
            # CHECK THIS: time of coallesence and fs
            tc = Constant(0),
            fs = Constant(2048),
            dist_mpc = Uniform(50, 200),
            ra = Uniform(0, 2 * torch.pi),
            dec = Cosine(),
            psi = Uniform(0, torch.pi)
        )

    def sample(self, batch_size):

        for k in self.params.keys():
            self.sampled_params[k] = self.params[k].sample((batch_size,))

        self.sampled_params['mass_2'] = self.sampled_params['chirp_mass'] * (1 + self.sampled_params['mass_ratio']) ** 0.2 / self.sampled_params['mass_ratio']**0.6
        self.sampled_params['mass_1'] = self.sampled_params['mass_ratio'] * self.sampled_params['mass_2']

        # if self.sampled_params['mass_2'] > self.sampled_params['mass_1']:
        #     self.sampled_params['mass_1'], self.sampled_params['mass_2'] = self.sampled_params['mass_2'], self.sampled_params['mass_1']
        #     self.sampled_params['mass_ratio'] = 1 / self.sampled_params['mass_ratio']


        # # correct units
        # self.sampled_params['mass_2'] *= lal.MSUN_SI
        # self.sampled_params['mass_1'] *= lal.MSUN_SI

        # convert from Bilby convention to Lalsimulation
        self.sampled_params['incl'], self.sampled_params['s1x'], self.sampled_params['s1y'], \
        self.sampled_params['s1z'], self.sampled_params['s2x'], self.sampled_params['s2y'], \
        self.sampled_params['s2z'] = transform_precessing_spins(
            self.sampled_params['theta_jn'], self.sampled_params['phi_jl'],
            self.sampled_params['tilt_1'],
            self.sampled_params['tilt_2'], self.sampled_params['phi_12'],
            self.sampled_params['a_1'], self.sampled_params['a_2'],
            self.sampled_params['mass_1'], self.sampled_params['mass_2'],
            self.sampled_params['reference_frequency'], self.sampled_params['phase']
            )

        self.sampled_params['incl'] = torch.Tensor(self.sampled_params['incl'])
        self.sampled_params['s1x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1x'])
        self.sampled_params['s1y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s1y'])
        self.sampled_params['s1z'] = torch.Tensor(self.sampled_params['s1z'])
        self.sampled_params['s2x'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2x'])
        self.sampled_params['s2y'] = Constant(0).sample((batch_size,)) # torch.Tensor(self.sampled_params['s2y'])
        self.sampled_params['s2z'] = torch.Tensor(self.sampled_params['s2z'])

        self.sampled_params['f_ref'] = self.sampled_params['reference_frequency']
        self.sampled_params['phiRef'] = self.sampled_params['phase']

        self.sampled_params['dist_mpc'] = (self.sampled_params['dist_mpc'] * u.Mpc).to("m").value # ???

        logger = logging.getLogger(__name__)

        for k in self.sampled_params.keys():
            if type(self.sampled_params[k])==float:
                logger.info(f'The shape of {k} is {self.sampled_params[k]}')
            else:
                logger.info(f'The shape of {k} is {self.sampled_params[k].shape}')
        
        return self.sampled_params

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
    def __init__(self, low, high):
        low_tensor = torch.tensor(low, dtype=torch.float64)
        high_tensor = torch.tensor(high, dtype=torch.float64)
        base_dist = torchdist.Uniform(torch.log(low_tensor), torch.log(high_tensor))
        super().__init__(base_dist, torchdist.ExpTransform())
        self.low = low_tensor
        self.high = high_tensor

class IntUniform(torchdist.Distribution):
    """Discrete uniform on {low, …, high‑1} with the same .sample(...) API."""
    arg_constraints = {}

    def __init__(self, low: int, high: int, validate_args=None):
        assert low < high, "`low` must be < `high`"
        self.low, self.high = int(low), int(high)
        super().__init__(torch.Size(), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        return torch.randint(
            self.low, self.high, sample_shape, dtype=torch.long
        )

    # (optional) rarely used, but keeps some libraries happy
    def log_prob(self, value):
        return torch.full_like(
            value.float(),
            -torch.log(torch.tensor(self.high - self.low, dtype=torch.float32)),
        )


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
            hrss = LogUniform(1.1e-23, 1.0e-21), 
            quality = Uniform(3, 700),
            frequency = Uniform(30, 2048),
            phase = Uniform(0, torch.pi),
            eccentricity = Uniform(0, 1)
        )

class MultiSineGaussianBBC(BasePrior):
    def __init__(self):
        super().__init__()
        n_max = 10
        p = OrderedDict()
        p["n_components"] = IntUniform(1, n_max + 1)          # 1 … n_max

        for i in range(1, n_max + 1):
            p[f"hrss_{i}"]        = LogUniform(1.6e-23, 1.5e-22)
            p[f"quality_{i}"]     = torchdist.Uniform(3.0, 700.0)
            p[f"frequency_{i}"]   = torchdist.Uniform(30, 2048.0)
            p[f"phase_{i}"]       = torchdist.Uniform(0.0, torch.pi)
            p[f"eccentricity_{i}"]= torchdist.Uniform(0.0, 1.0)

        self.params = p

class GaussianBBC(BasePrior):

    def __init__(self):
    # this is a super wide range for all the signals with converted amplitude to hrss here: https://git.ligo.org/bursts/burst-pipeline-benchmark/-/wikis/o4b_1/Waveforms-O4b-1
        super().__init__()
        self.params = OrderedDict(
            hrss = LogUniform(1.6e-23, 2.0e-16),
            duration = Uniform(0.001, 0.1) # This may be too flat in a one second or shorter window
            # duration = Uniform(0.001, 0.02) # this is the duration of the gaussian in seconds
        )

class WhiteNoiseBurstBBC(BasePrior):
    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            time_envelope = Uniform(2e-2,2),
            frequency = Uniform(40, 1500),
            bandwidth = Uniform(10, 200),
            eccentricity = Uniform(0, 1),
            phase = Uniform(0, torch.pi),
            int_hdot_squared = LogUniform(3.0e-40, 2.5e-34),
        )

class CuspBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-4.0 / 3.0),
            amplitude = Uniform(4.0e-22, 4.0e-21),
            f_high = Uniform(40, 1000)
        )

class KinkBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-5.0 / 3.0),
            amplitude = Uniform(1.4e-21, 1.4e-20),
            f_high = Uniform(40, 1000)
        )

class KinkkinkBBC(BasePrior):

    def __init__(self):
        super().__init__()
        self.params = OrderedDict(
            power = Constant(-2.0),
            amplitude = Uniform(4.7e-21, 4.7e-20),
            f_high = Uniform(2047.9, 2048)
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
        self.priors['chirp_mass'] = Uniform(5, 65)
        
        # Mass ratio m1/m2. (TensorType)
        self.priors['mass_ratio'] = Uniform(0.5, 0.99) 
        
        # # Luminosity distance in Mpc.(TensorType)
        self.priors["distance"] = Uniform(100, 1375) # dist_mpc
        
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


class FakeGlitchPrior(BasePrior):
    def __init__(self,selected_signals=None):
        super().__init__()
        all_priors = {
            "MultiSineGaussian":MultiSineGaussianBBC,
            "BBH":LAL_BBHPrior,
            "Gaussian":GaussianBBC,
            "Cusp":CuspBBC,
            "Kink":KinkBBC,
            "Kinkkink":KinkkinkBBC,
            "WhiteNoiseBurst":WhiteNoiseBurstBBC,
            "CCSN":MultiSineGaussianBBC # dummy value, not used for CCSN. we use Andy's custom implementation
        }
        if selected_signals is None:
            self.selected_signals = list(all_priors.keys())
        else:
            self.selected_signals = selected_signals
        self.selected_priors = {}
        for p in self.selected_signals:
            if p not in all_priors.keys():
                print(f"Unrecogized prior name {p}, skipping. Please modify the code to add it if needed.")
            else:
                self.selected_priors[p] = all_priors[p]()

    def num_per_signal(self,batch_size):
        num_per = batch_size//len(self.selected_signals)
        rem = batch_size%len(self.selected_signals)
        nums = [num_per for _ in range(len(self.selected_signals))]
        for i in range(rem):
            nums[i] += 1
        return nums
    
    def sample(self, batch_size):
        nums = self.num_per_signal(batch_size)
        sampled_params = []
        for nsample, signal in zip(nums,self.selected_signals):
            sampled = self.selected_priors[signal].sample(nsample)
            sampled_params.append(sampled)
        
        return sampled_params
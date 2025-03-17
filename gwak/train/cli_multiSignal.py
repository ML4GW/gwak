import logging

from lightning.pytorch.cli import LightningCLI

from ml4gw import waveforms


def sum_args(a, b):
    return float(a) + float(b)

class GwakMultiSignalCLI(LightningCLI):
    def before_instantiate_classes(self):
        sample_rate = self.config["fit.data.init_args.sample_rate"]
        fduration = self.config['fit.data.init_args.fduration']
        kernel_length = self.config['fit.data.init_args.kernel_length']
        
        for i in range(len(self.config['fit.data.init_args.waveforms'])):
            
            if "sample_rate" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['sample_rate'] = sample_rate
            
            if "duration" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['duration'] = fduration+kernel_length
        
        return

def cli_main(args=None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Started')

    cli = GwakMultiSignalCLI(
        save_config_kwargs={'overwrite': True},
        args=args
    )


if __name__ == '__main__':
    cli_main()


from gwak.data.prior import SineGaussianBBC, LAL_BBHPrior, GaussianBBC, CuspBBC, KinkBBC, KinkkinkBBC, WhiteNoiseBurstBBC
from ml4gw.waveforms import SineGaussian, IMRPhenomPv2, Gaussian, GenerateString, WhiteNoiseBurst

signal_classes = [
    "SineGaussian",
    "BBH",
    "Gaussian",
    "Cusp",
    "Kink",
    "KinkKink",
    "WhiteNoiseBurst"
]
priors = [
    SineGaussianBBC,
    LAL_BBHPrior,
    GaussianBBC,
    CuspBBC,
    KinkBBC,
    KinkkinkBBC,
    WhiteNoiseBurstBBC
]
waveforms = [
    SineGaussian(
        sample_rate=sample_rate,
        duration=duration
    ),
    IMRPhenomPv2(),
    Gaussian(
        sample_rate=sample_rate,
        duration=duration
    ),
    GenerateString(
        sample_rate=sample_rate
    ),
    GenerateString(
        sample_rate=sample_rate
    ),
    GenerateString(
        sample_rate=sample_rate
    ),
    WhiteNoiseBurst(
        sample_rate=sample_rate,
        duration=duration
    )
]
extra_kwargs = [
    None,
    {"ringdown_duration":0.9},
    None,
    None,
    None,
    None,
    None
]
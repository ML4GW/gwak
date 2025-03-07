import logging

from lightning.pytorch.cli import LightningCLI

from ml4gw import waveforms


def sum_args(a, b):
    return float(a) + float(b)

class GwakMultiSignalCLI(LightningCLI):
    def before_instantiate_classes(self):
        
        #for k,v in self.config.items():
        #    print(k,v)
        
        sample_rate = self.config["fit.data.init_args.sample_rate"]
        fduration = self.config['fit.data.init_args.fduration']
        kernel_length = self.config['fit.data.init_args.kernel_length']
        
        for i in range(len(self.config['fit.data.init_args.waveforms'])):
            
            if "sample_rate" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['sample_rate'] = sample_rate
            
            if "duration" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['duration'] = fduration+kernel_length
        
        #print("new config:")
        #for k,v in self.config.items():
        #    print(k,v)
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
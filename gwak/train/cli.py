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
            if self.config['fit.data.init_args.signal_classes'][i] in ["Background", "Glitch"]: continue
            
            if "sample_rate" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['sample_rate'] = sample_rate
            
            if "duration" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['duration'] = fduration+kernel_length

        # harmonize lr between optimizer and scheduler if applicable
        if self.config['fit.model.class_path'] == "models.Tarantula":
            print("Making LR scheduling update for tarantula")
            tot_steps = self.config['fit.trainer.max_epochs'] * self.config['fit.data.init_args.batches_per_epoch']
            self.config['fit.model.init_args.total_steps'] = tot_steps

        # make sure length is right if we're using S4 SSM
        if self.config['fit.model.class_path'] == "models.Crayon":
            self.config['fit.model.init_args.num_timesteps'] = int(sample_rate * kernel_length)
                
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
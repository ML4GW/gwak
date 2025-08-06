import logging

from lightning.pytorch.cli import LightningCLI

from ml4gw import waveforms


def sum_args(a, b):
    return float(a) + float(b)

class GwakMultiSignalCLI(LightningCLI):
    def before_instantiate_classes(self):
        sample_rate = getattr(self.config,"fit.data.init_args.sample_rate",None)
        fduration = getattr(self.config,'fit.data.init_args.fduration',None)
        kernel_length = getattr(self.config,'fit.data.init_args.kernel_length',None)
        if sample_rate is None or fduration is None or kernel_length is None:
            print("Sample rate, fduration, and kernel length =", sample_rate, fduration, kernel_length,", make sure this is right for your purposes!")
        
        waveforms = getattr(self.config, 'fit.data.init_args.waveforms', [])
        for i in range(len(waveforms)):
            if self.config['fit.data.init_args.signal_classes'][i] in ["Background", "Glitch", "CCSN", "FakeGlitch"]: continue
            
            if "sample_rate" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['sample_rate'] = sample_rate
            
            if "duration" in self.config['fit.data.init_args.waveforms'][i]['init_args'].keys():
                self.config['fit.data.init_args.waveforms'][i]['init_args']['duration'] = fduration+kernel_length

        # harmonize lr between optimizer and scheduler if applicable
        if self.config['fit.model.class_path'] == "cl_models.Tarantula" or self.config['fit.model.class_path'] == "cl_models.iTransformer":
            print("Making LR scheduling update for tarantula")
            batches_per_epoch = self.config['fit.data.init_args.batches_per_epoch'] if 'fit.data.init_args.batches_per_epoch' in self.config.keys() else self.config['fit.trainer.limit_train_batches']
            tot_steps = self.config['fit.trainer.max_epochs'] * batches_per_epoch
            self.config['fit.model.init_args.total_steps'] = tot_steps

        # make sure length is right if we're using S4 SSM
        if self.config['fit.model.class_path'] == "cl_models.Crayon":
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
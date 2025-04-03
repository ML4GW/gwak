import logging

from lightning.pytorch.cli import LightningCLI

from ml4gw import waveforms


def sum_args(a, b):
    return float(a) + float(b)


class GwakSignalCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        parser.link_arguments(
            'data.init_args.signal_class.init_args.sample_rate',
            'data.init_args.signal_class.init_args.waveform.init_args.sample_rate',
            apply_on='parse'
        )
        parser.link_arguments(
            ('data.init_args.signal_class.init_args.kernel_length', 'data.init_args.signal_class.init_args.fduration'),
            'data.init_args.signal_class.init_args.waveform.init_args.duration',
            compute_fn=sum_args,
            apply_on='parse'
        )


def cli_main(args=None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Started')

    cli = GwakSignalCLI(
        save_config_kwargs={'overwrite': True},
        args=args
    )


if __name__ == '__main__':
    cli_main()
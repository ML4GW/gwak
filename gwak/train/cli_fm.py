import logging
from lightning.pytorch.cli import LightningCLI


def cli_main(args=None):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Started')

    cli = LightningCLI(
        save_config_kwargs={'overwrite': True},
        args=args
    )


if __name__ == '__main__':
    cli_main()
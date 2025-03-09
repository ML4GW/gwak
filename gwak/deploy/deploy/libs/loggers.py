import sys
import logging


def gwak_logger(
    log_file,
    log_level=logging.DEBUG,
    log_format="%(asctime)s %(name)s %(levelname)s:\t%(message)s",
    date_format="%H:%M:%S"
):

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format=log_format,
        datefmt=date_format,
        level=log_level,
        force=True
    )

    # Get the root logger
    logger = logging.getLogger()

    # Ensure console output matches file format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level) 

    # Apply the same formatter for console logs
    formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Prevent duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)
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
        
        
class Pathfinder:

    def __init__(
        self,
        dir_var: str = "GWAK_PATH",
        path_function_name: Optional[str] = None,
        target_path: Optional[str] = None,
    ):

        # A name for the path function, used in logging
        self.path_function_name = path_function_name or "project_path"

        try:
            project_path = os.environ[f"{dir_var}"]

        except KeyError:

            logging.error(f"Can not find {dir_var} please export {dir_var} to bash variable!")
            sys.exit()

        except Exception as e:
            logging.error(f"{type(e).__name__}")
            sys.exit()

        self.project_path = Path(project_path)


        if target_path is not None:
            self.project_path = self.project_path / target_path

    def __call__(
        self,
        additional_path: Optional[str] = None,
        verbose: bool = False,
    ) -> Path:

        if additional_path is not None:

            return self.project_path / additional_path

        if verbose:
            logging.info(f"Resolved {self.path_function_name} at:")
            logging.info(f"    {self.project_path}")
        return self.project_path

class project_data_dir(Pathfinder):

    def __init__(
        self,
    ):
        super().__init__(
            dir_var="GWAK_DATA_DIR",
            path_function_name="data directory",
        )

class project_output_dir(Pathfinder):

    def __init__(
        self,
    ):
        super().__init__(
            dir_var="GWAK_OUTPUT_DIR",
            path_function_name="output directory",
        )

class project_image_dir(Pathfinder):

    def __init__(
        self,
    ):
        super().__init__(
            dir_var="GWAK_IMAGE_DIR",
            path_function_name="image directory",
        )
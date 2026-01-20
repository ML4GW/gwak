import sys

from jsonargparse import ArgumentParser, ActionConfigFile

subcommands_keys = [
    "export",
    "condor_client", 
    "infer", 
    "infer_condor", 
    "deploy", 
    "post_analyze",
    "resolve_O4_bbc"
]

# Keys to skip during resolving subcommands (export, infer, deploy,...)
skip_keys = [
    "project", 
    "output_dir", 
    "run_name", 
    "cl_config", 
    "fm_config",
    "Tb"
]

def build_parser(
    subcommands=subcommands_keys,
    action="store",
    skip_keys=skip_keys
):

    parser = ArgumentParser(prog="cli")
    parser.add_argument("subcommand", choices=subcommands)
    parser.add_argument("--config", action=action)
    for key in skip_keys:
        parser.add_argument(f"--{key}", required=False)

    return parser

def main(args=None):

    parser = build_parser(skip_keys=skip_keys)
    subcommand = parser.parse_args(sys.argv[1:]).subcommand 

    if subcommand == "export":
        from deploy.export import export as main_cli

    if subcommand == "condor_client":
        from deploy.condor_infer_module import infer as main_cli

    if subcommand == "infer":
        from deploy.infer_module import infer as main_cli

    if subcommand == "infer_condor":
        from deploy.condor_handler import condor_infer_wrapper as main_cli

    if subcommand == "infer_slurm":
        from deploy.slurm_handeler import slurm_infer_wrapper as main_cli

    if subcommand == "post_analyze":
        from deploy.analyzer import scan as main_cli

    if subcommand == "resolve_O4_bbc":
        from deploy.analyzer import resolve_bbc as main_cli

    # Create subparser
    subparser = build_parser(action=ActionConfigFile)
    subparser.add_function_arguments(
        main_cli,
        skip=skip_keys
    )

    # Parse and instantiate classes
    args = subparser.parse_args()
    args = subparser.instantiate_classes(args)

    delattr(args, "subcommand")
    main_cli(**args)

if __name__ == "__main__":
    main()
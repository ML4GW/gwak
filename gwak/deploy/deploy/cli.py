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
# The skipped keys should only by string types variables. 
# Avoid passing non string type variables to skip list. 
# Otherwise, you would have to add an additinaol type check to 
skip_keys = [
    "project",
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

def export_args_hook():

    import yaml
    from deploy.libs import gwak_dir

    export_cfg = gwak_dir()(append_path="gwak/deploy/deploy/config/export.yaml")

    with open(export_cfg) as f:
        export_args = yaml.safe_load(f)

    return dict(
        ifos = export_args.get("ifos"),
        psd_length = export_args.get("psd_length"),
        stride_batch_size=export_args.get("stride_batch_size"),
        sample_rate=export_args.get("sample_rate"),
        inference_rate=export_args.get("inference_rate"),
    )

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
    if subcommand == "infer_condor":
        for key, value in export_args_hook().items():
            setattr(args, key, value)
    delattr(args, "subcommand")
    main_cli(**args)

if __name__ == "__main__":
    main()
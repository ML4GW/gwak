import sys

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser(
    subcommands=["export", "infer", "deploy"],
    action="store",
    skip_keys=["project", "output_dir", "run_name"]
):

    parser = ArgumentParser(prog="cli")
    parser.add_argument("subcommand", choices=subcommands)
    parser.add_argument("--config", action=action)
    for key in skip_keys:
        parser.add_argument(f"--{key}", required=False)

    return parser

def main(args=None):

    skip_keys = ["project", "output_dir", "run_name"]
    parser = build_parser(skip_keys=skip_keys)
    subcommand = parser.parse_args(sys.argv[1:]).subcommand 

    if subcommand == "export":
        from deploy.export import export
        
        subparser = build_parser(action=ActionConfigFile)
        subparser.add_function_arguments(
            export, 
            skip=skip_keys
        ) 
        args = subparser.parse_args()
        args = subparser.instantiate_classes(args)
        delattr(args, "subcommand")

        export(**args)

    if subcommand == "infer":
        from deploy.infer_module import infer
        
        subparser = build_parser(action=ActionConfigFile)
        subparser.add_function_arguments(
            infer, 
            skip=skip_keys
        ) 
        args = subparser.parse_args()
        args = subparser.instantiate_classes(args)
        delattr(args, "subcommand")

        infer(**args)

    if subcommand == "deploy":
        from deploy.slurm_handeler import slurm_infer_wrapper    
        
        subparser = build_parser(action=ActionConfigFile)

        subparser.add_function_arguments(
            slurm_infer_wrapper, 
            skip=skip_keys
        )
        args = subparser.parse_args()
        args = subparser.instantiate_classes(args)
        delattr(args, "subcommand")

        slurm_infer_wrapper(**args)
    
    
if __name__ == "__main__":
    main()
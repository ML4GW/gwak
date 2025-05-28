from deploy.slurm_handeler import slurm_infer_wrapper

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(slurm_infer_wrapper)
    
    return parser

def main(args=None):
    
    parser = build_parser()
    args = parser.parse_args()
    args = args.as_dict()

    slurm_infer_wrapper(**args)
    
    
if __name__ == "__main__":
    main()
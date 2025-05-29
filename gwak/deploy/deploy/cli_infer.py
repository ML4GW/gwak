from deploy.infer_module import infer

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser():
    
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(infer)
    
    return parser

def main(args=None):
    
    parser = build_parser()
    args = parser.parse_args()
    args = args.as_dict()

    infer(**args)
    
    
if __name__ == "__main__":
    main()
from background import gwak_background

from jsonargparse import ArgumentParser, ActionConfigFile


def build_parser():
    
    parser = ArgumentParser(default_env=True)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(gwak_background)
    
    return parser

def main(args=None):
    
    parser = build_parser()
    args = parser.parse_args()
    args = args.as_dict()
    
    gwak_background(**args)
    
    
if __name__ == "__main__":
    main()

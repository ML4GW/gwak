from background import make_seg_list, gwak_background

from jsonargparse import ArgumentParser, ActionConfigFile

SUBCOMMANDS = {
    "make_seg_list": make_seg_list,
    "get_strain": gwak_background,
}


def build_parser():

    parser = ArgumentParser(default_env=True)
    parser.add_argument("--config", action=ActionConfigFile)

    subcommands = parser.add_subcommands(dest="subcommand")
    for name, main_cli in SUBCOMMANDS.items():
        subparser = ArgumentParser()
        subparser.add_function_arguments(main_cli)
        subcommands.add_subcommand(name, subparser)

    return parser

def main(args=None):

    parser = build_parser()
    args = parser.parse_args(args)

    subcommand = args.subcommand
    main_cli = SUBCOMMANDS[subcommand]
    main_cli(**args[subcommand].as_dict())


if __name__ == "__main__":
    main()

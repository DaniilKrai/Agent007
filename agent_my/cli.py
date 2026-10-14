import argparse
from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Console coding agent with tool calling",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"coding-agent {__version__}",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
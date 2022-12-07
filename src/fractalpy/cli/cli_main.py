"""This module provides the Command Line Interface for this package."""

import click

from .. import __version__
from .commands import cmd_julia, cmd_mandelbrot


@click.group()
@click.version_option(__version__)
@click.pass_context
def cli_main(ctx):
    pass


# Add the commands to the main group
cli_main.add_command(cmd_mandelbrot.mandelbrot)
cli_main.add_command(cmd_julia.julia)


# We need to define main function in this module for mpire's pool function to work
def main():
    cli_main(max_content_width=150)

"""This module provides the Command Line Interface for this package."""

from importlib.metadata import version

import click

from .commands import cmd_julia, cmd_mandelbrot


@click.group()
# @click.version_option(version(__package__))
@click.pass_context
def cli_main(ctx):
    pass


cli_main.add_command(cmd_mandelbrot.mandelbrot)
cli_main.add_command(cmd_julia.julia)

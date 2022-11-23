"""This module provides the Command Line Interface for this package."""

from importlib.metadata import version

import click
import collections

from .. import __version__
from .commands import cmd_julia, cmd_mandelbrot


class OrderedGroup(click.Group):
    """This class is for reordering the listing of sub-commands in the group to the order that they appear"""

    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands


@click.group(cls=OrderedGroup)
@click.version_option(__version__)
@click.pass_context
def cli_main(ctx):
    pass


cli_main.add_command(cmd_mandelbrot.mandelbrot)
cli_main.add_command(cmd_julia.julia)


# We need to define main in here for multiprocessing to work properly with mpire
def main():
    cli_main(max_content_width=150)

from typing import get_args, get_type_hints

import click
from fractalpy.cli.helper import get_default_args
from fractalpy.fractals import fractals as frac

from . import cmd_fractal_base as cmd

mandel_type_hints = get_type_hints(frac.Mandelbrot.__init__)
mandel_default_args = get_default_args(frac.Mandelbrot.__init__)


@click.group()
@click.pass_context
@click.option('--limits',
              nargs=4,
              type=get_args(mandel_type_hints['limits']),
              default=mandel_default_args['limits'],
              show_default=True,
              help="Lower and upper limits for the ranges of x values and y values."
              )
@cmd.needs_options
def mandelbrot(ctx, limits, npts, threshold, cmap, setcolor, pallet_len, shift):
    """Commands related to the Mandelbrot set."""
    ctx.obj = frac.Mandelbrot(limits=limits,
                              n_pts=npts,
                              threshold=threshold,
                              color_map=cmap,
                              c_set=setcolor,
                              pallet_len=pallet_len,
                              color_map_shift=shift
                              )


mandelbrot.add_command(cmd.plot_fractal)
mandelbrot.add_command(cmd.save_fractal)
mandelbrot.add_command(cmd.zoom_fractal)

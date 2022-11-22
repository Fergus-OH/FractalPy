from typing import get_args, get_type_hints

import click

from fractalpy.cli.helper import get_default_args
from fractalpy.fractals import fractals as frac

from . import cmd_fractal_base as cmd

julia_type_hints = get_type_hints(frac.Julia.__init__)
julia_default_args = get_default_args(frac.Julia.__init__)


# @cli_main.group()
@click.group()
@click.pass_context
@click.option('-c',
              type=julia_type_hints['c'],
              default=julia_default_args['c'],
              show_default=True,
              help="c value"
              )
@click.option('--limits',
              nargs=4,
              type=get_args(julia_type_hints['limits']),
              default=julia_default_args['limits'],
              show_default=True,
              help="limits"
              )
@cmd.needs_options
def julia(ctx, c, limits, npts, threshold, cmap, setcolor, pallet_len, shift):
    """Commands relating to the Julia set"""
    ctx.obj = frac.Julia(c=c,
                         limits=limits,
                         n_pts=npts,
                         threshold=threshold,
                         color_map=cmap,
                         c_set=setcolor,
                         pallet_len=pallet_len,
                         color_map_shift=shift
                         )


julia.add_command(cmd.plot_fractal)
julia.add_command(cmd.save_fractal)
julia.add_command(cmd.zoom_fractal)

spin_type_hints = get_type_hints(frac.Julia.spin)
spin_default_args = get_default_args(frac.Julia.spin)


@julia.command('spin')
@click.pass_context
@click.option('--filename',
              type=spin_type_hints['filename'],
              default=None,
              show_default=True,
              help='output filename'
              )
@click.option('--extension',
              type=spin_type_hints['extension'],
              default=spin_default_args['extension'],
              show_default=True,
              help='output extension type'
              )
# @click.option('--frame_subdir',
#               type=zoom_type_hints['frame_subdir'],
#               default='frames',
#               show_default=True,
#               help='directory to store frames'
#               )
@click.option('--n_frames',
              type=spin_type_hints['n_frames'],
              default=spin_default_args['n_frames'],
              show_default=True,
              help='Number of frames for video'
              )
@click.option('--fps',
              type=spin_type_hints['fps'],
              default=spin_default_args['fps'],
              show_default=True,
              help='framerate of video in frames per second'
              )
@click.option('--n_jobs',
              type=spin_type_hints['n_jobs'],
              default=spin_default_args['n_jobs'],
              show_default=True,
              help='number of processors for multiprocessing frame generation'
              )
def spin_julia(ctx, filename, extension, frame_subdir, n_frames, fps, n_jobs):
    """create a video of rotating the parameter c"""
    ctx.obj.spin(filename, extension, frame_subdir, n_frames, fps, n_jobs)

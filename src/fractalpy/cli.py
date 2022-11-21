"""This module provides the Command Line Interface for this package."""
from functools import wraps
from importlib.metadata import version
from typing import get_args, get_type_hints

import click

from . import fractals as frac
from .helper import get_default_args

base_type_hints = get_type_hints(frac.FractalBase.__init__)
mandel_type_hints = get_type_hints(frac.Mandelbrot.__init__)
julia_type_hints = get_type_hints(frac.Julia.__init__)

base_default_args = get_default_args(frac.FractalBase.__init__)
mandel_default_args = get_default_args(frac.Mandelbrot.__init__)
julia_default_args = get_default_args(frac.Julia.__init__)


def needs_options(f):
    @wraps(f)
    @click.option('--npts',
                  type=base_type_hints['n_pts'],
                  default=base_default_args['n_pts'],
                  show_default=True,
                  help="npts value"
                  )
    @click.option('--threshold',
                  type=base_type_hints['threshold'],
                  default=base_default_args['threshold'],
                  show_default=True,
                  help="threshold"
                  )
    @click.option('--cmap',
                  type=base_type_hints['color_map'],
                  default=base_default_args['color_map'],
                  show_default=True,
                  help="color map"
                  )
    @click.option('--setcolor',
                  type=base_type_hints['c_set'],
                  default=base_default_args['c_set'],
                  show_default=True,
                  help="set color of the set"
                  )
    @click.option('--pallet_len',
                  type=base_type_hints['pallet_len'],
                  default=base_default_args['pallet_len'],
                  show_default=True,
                  help="pallet length"
                  )
    @click.option('--shift',
                  type=base_type_hints['color_map_shift'],
                  default=base_default_args['color_map_shift'],
                  show_default=True,
                  help="color_map_shift"
                  )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
@click.version_option(version(__package__))
@click.pass_context
def cli(ctx):
    pass


@cli.group()
@click.pass_context
@click.option('--limits',
              nargs=4,
              type=get_args(mandel_type_hints['limits']),
              default=mandel_default_args['limits'],
              show_default=True,
              help="limits"
              )
@needs_options
def mandelbrot(ctx, limits, npts, threshold, cmap, setcolor, pallet_len, shift):
    """Commands relating to the Mandelbrot set"""
    ctx.obj = frac.Mandelbrot(limits=limits,
                              n_pts=npts,
                              threshold=threshold,
                              color_map=cmap,
                              c_set=setcolor,
                              pallet_len=pallet_len,
                              color_map_shift=shift
                              )


@cli.group()
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
@needs_options
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


plot_type_hints = get_type_hints(frac.FractalBase.plot)
plot_default_args = get_default_args(frac.FractalBase.plot)


# These are essentially a base commands
@click.command(name='plot')
@click.pass_context
@click.option('--fig_size',
              type=plot_type_hints['fig_size'],
              default=plot_default_args['fig_size'],
              show_default=True,
              help="size of figure"
              )
@click.option("--axis",
              is_flag=True,
              default=False,
              show_default=True,
              help="Show axis"
              )
@click.option("--nticks",
              type=plot_type_hints['n_ticks'],
              default=plot_default_args['n_ticks'],
              show_default=True,
              help="Number of ticks"
              )
def plot_fractal(ctx, fig_size, axis, nticks):
    """plot the set"""
    ctx.obj.plot(fig_size=fig_size, axis=axis, n_ticks=nticks)


save_type_hints = get_type_hints(frac.FractalBase.save)
save_default_args = get_default_args(frac.FractalBase.save)


@click.command('save')
@click.pass_context
@click.option('--filename',
              type=save_type_hints['filename'],
              default=save_default_args['filename'],
              help="filename of image"
              )
@click.option('--extension',
              type=save_type_hints['extension'],
              default=save_default_args['extension'],
              help="extension of image"
              )
def save_fractal(ctx, filename, extension):
    """save an image of the set"""
    ctx.obj.save(filename=filename, extension=extension)


zoom_type_hints = get_type_hints(frac.FractalBase.zoom)
zoom_default_args = get_default_args(frac.FractalBase.zoom)


@click.command('zoom')
@click.pass_context
@click.option('--magnitude',
              '-m',
              type=zoom_type_hints['m'],
              default=None,     # see function
              show_default=True,
              help='magnitude of zoom to target location'
              )
@click.option('--target',
              nargs=2,
              type=zoom_type_hints['target'],
              default=None,     # see function
              show_default=True,
              help='target location for zoom'
              )
@click.option('--filename',
              type=zoom_type_hints['filename'],
              default=zoom_default_args['filename'],
              show_default=True,
              help='output filename'
              )
@click.option('--extension',
              type=zoom_type_hints['extension'],
              default=zoom_default_args['extension'],
              show_default=True,
              help='output extension type'
              )
@click.option('--frame_subdir',
              type=zoom_type_hints['frame_subdir'],
              default=zoom_default_args['frame_subdir'],
              show_default=True,
              help='directory to store frames'
              )
@click.option('--n_frames',
              type=zoom_type_hints['n_frames'],
              default=zoom_default_args['n_frames'],
              show_default=True,
              help='Number of frames for video'
              )
@click.option('--fps',
              type=zoom_type_hints['fps'],
              default=zoom_default_args['fps'],
              show_default=True,
              help='framerate of video in frames per second'
              )
@click.option('--n_jobs',
              type=zoom_type_hints['n_jobs'],
              default=zoom_default_args['n_jobs'],
              show_default=True,
              help='number of processors for multiprocessing frame generation'
              )
@click.option('--preview',
              is_flag=True,
              default=False,
              show_default=True,
              help="Preview target location"
              )
@click.option('--nticks',
              default=5,
              type=int,
              show_default=True,
              help="Number of axes ticks (useful for preview)"
              )
def zoom_fractal(ctx,
                 magnitude,
                 target,
                 filename,
                 extension,
                 frame_subdir,
                 n_frames,
                 fps,
                 n_jobs,
                 preview,
                 nticks
                 ):
    """create a video of zooming into the set"""
    if preview:
        ctx.obj.x_min, ctx.obj.x_max, ctx.obj.y_min, ctx.obj.y_max = ctx.obj.get_target_ranges(m=magnitude,
                                                                                               target=target
                                                                                               )

        ctx.obj.plot(axis='on', n_ticks=nticks)
    else:

        ctx.obj.magnitude = magnitude if magnitude is not None else get_default_args(ctx.obj.zoom)['m']
        ctx.obj.target = target if target is not None else get_default_args(ctx.obj.zoom)['target']

        ctx.obj.zoom(filename=filename,
                     extension=extension,
                     frame_subdir=frame_subdir,
                     n_frames=n_frames,
                     fps=fps,
                     n_jobs=n_jobs
                     )


mandelbrot.add_command(plot_fractal)
mandelbrot.add_command(save_fractal)
mandelbrot.add_command(zoom_fractal)

julia.add_command(plot_fractal)
julia.add_command(save_fractal)
julia.add_command(zoom_fractal)


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

"""This module provides the FRACTAL CLI."""

import click
import os
from fractal import fractal as frac
from functools import wraps

# from fractal import __app_name__, __version__


def needs_options(f):
    @wraps(f)
    @click.option('--npts',
                  default=500,
                  show_default=True,
                  type=int,
                  help="npts value"
                  )
    @click.option('--threshold',
                  default=500,
                  show_default=True,
                  type=int,
                  help="npts value"
                  )
    @click.option('--cmap',
                  default='hsv',
                  show_default=True,
                  type=str,
                  help="color map"
                  )
    @click.option('--setcolor',
                  default='black',
                  show_default=True,
                  type=str,
                  help="set color of the set"
                  )
    @click.option('--pallet_len',
                  default=250,
                  show_default=True,
                  type=int,
                  help="pallet length"
                  )
    @click.option('--shift',
                  default=0,
                  show_default=True,
                  type=int,
                  help="color_map_shift"
                  )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
# @click.version_option(__version__)
@click.pass_context
def cli(ctx):
    pass


@cli.group()
@click.pass_context
@click.option('--ranges',
              default=((-2, 1), (-1.5, 1.5)),
              show_default=True,
              type=tuple,
              help="range"
              )
@needs_options
def mandelbrot(ctx, ranges, npts, threshold, cmap, setcolor, pallet_len, shift):
    """docstring"""
    ctx.obj = frac.Mandelbrot(x_ran=ranges[0],
                              y_ran=ranges[1],
                              n_pts=npts,
                              threshold=threshold,
                              color_map=cmap,
                              c_set=setcolor,
                              pallet_len=pallet_len,
                              color_map_shift=shift
                              )


@cli.group()
@click.pass_context
@click.option('--ranges',
              default=((-1.5, 1.5), (-1.5, 1.5)),
              show_default=True,
              type=tuple,
              help="range"
              )
@click.option('-c',
              default=-0.79 + 0.15j,
              show_default=True,
              type=complex,
              help="c value"
              )
@needs_options
def julia(ctx, ranges, c, npts, threshold, cmap, setcolor, pallet_len, shift):
    """Commands for HTTPS"""
    ctx.obj = frac.Julia(c=c,
                         x_ran=ranges[0],
                         y_ran=ranges[1],
                         n_pts=npts,
                         threshold=threshold,
                         color_map=cmap,
                         c_set=setcolor,
                         pallet_len=pallet_len,
                         color_map_shift=shift
                         )


# These are essentially a base commands
@click.command(name='plot')
@click.pass_context
@click.option('--dpi',
              default=None,
              show_default=True,
              type=int,
              help="DPI of image"
              )
@click.option('--fig_size',
              default=None,
              show_default=True,
              type=float,
              help="size of figure"
              )
@click.option("--axis",
              is_flag=True,
              default=False,
              show_default=True,
              help="Show axis"
              )
def plot_fractal(ctx, axis, dpi, fig_size):
    """docstring"""
    ctx.obj.plot(axis=axis, dpi=dpi, fig_size=fig_size)


@click.command('save')
@click.pass_context
@click.option('--filename',
              default=None,
              type=str,
              help="filename of image"
              )
@click.option('--extension',
              default='png',
              type=str,
              help="extension of image"
              )
def save_fractal(ctx, filename, extension):
    """docstring"""
    ctx.obj.save(filename, extension)


@click.command('zoom')
@click.pass_context
@click.option('--target',
              default=(6e+4, -1.186592e+0, -1.901211e-1),
              type=tuple
              )
@click.option('--filename',
              default=None
              )
@click.option('--extension',
              default='gif'
              )
@click.option('--frame_subdir',
              default='frames'
              )
@click.option('--n_frames',
              default=120
              )
@click.option('--fps',
              default=60
              )
@click.option('--n_jobs',
              default=os.cpu_count()
              )
def zoom_fractal(ctx, target, filename, extension, frame_subdir, n_frames, fps, n_jobs):
    """docstring"""
    ctx.obj.zoom(target=target,
                 filename=filename,
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


@julia.command('spin')
@click.pass_context
@click.option('--filename',
              default=None
              )
@click.option('--extension',
              default='gif'
              )
@click.option('--frame_subdir',
              default='frames'
              )
@click.option('--n_frames',
              default=120
              )
@click.option('--fps',
              default=60
              )
@click.option('--n_jobs',
              default=os.cpu_count()
              )
def spin_julia(ctx, filename, extension, frame_subdir, n_frames, fps, n_jobs):
    """docstring"""
    ctx.obj.spin(filename, extension, frame_subdir, n_frames, fps, n_jobs)

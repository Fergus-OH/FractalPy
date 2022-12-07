from functools import wraps
from typing import get_args, get_type_hints

import click
from fractalpy.cli.helper import get_default_args
from fractalpy.fractals import fractals as frac

base_type_hints = get_type_hints(frac.FractalBase.__init__)
base_default_args = get_default_args(frac.FractalBase.__init__)

plot_type_hints = get_type_hints(frac.FractalBase.plot)
plot_default_args = get_default_args(frac.FractalBase.plot)

zoom_type_hints = get_type_hints(frac.FractalBase.zoom)
zoom_default_args = get_default_args(frac.FractalBase.zoom)

save_type_hints = get_type_hints(frac.FractalBase.save)
save_default_args = get_default_args(frac.FractalBase.save)


def needs_options(f):
    @wraps(f)
    @click.option('--npts',
                  type=base_type_hints['n_pts'],
                  default=base_default_args['n_pts'],
                  show_default=True,
                  help="Number of points along the x-axis."
                  )
    @click.option('--threshold',
                  type=base_type_hints['threshold'],
                  default=base_default_args['threshold'],
                  show_default=True,
                  help="Number of iterations required to determine if a point is in the set."
                  )
    @click.option('--cmap',
                  type=base_type_hints['color_map'],
                  default=base_default_args['color_map'],
                  show_default=True,
                  help="Specify Matplotlib colormap."
                  )
    @click.option('--setcolor',
                  type=base_type_hints['c_set'],
                  default=base_default_args['c_set'],
                  show_default=True,
                  help="Color of the plotted set."
                  )
    @click.option('--pallet_len',
                  type=base_type_hints['pallet_len'],
                  default=base_default_args['pallet_len'],
                  show_default=True,
                  help="Periodic length of colormap pallet."
                  )
    @click.option('--shift',
                  type=base_type_hints['color_map_shift'],
                  default=base_default_args['color_map_shift'],
                  show_default=True,
                  help="Shift of colormap."
                  )
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


# These are commands that can be used for all fractal classes
@click.command(name='plot')
@click.pass_context
@click.option('--fig_size',
              type=plot_type_hints['fig_size'],
              default=plot_default_args['fig_size'],
              show_default=True,
              help="Size of figure plot."
              )
@click.option("--axis",
              is_flag=True,
              default=False,
              show_default=True,
              help="Show axis flag."
              )
@click.option("--nticks",
              type=plot_type_hints['n_ticks'],
              default=plot_default_args['n_ticks'],
              show_default=True,
              help="Number of ticks along axes."
              )
def plot_fractal(ctx, fig_size, axis, nticks):
    """Plot the set."""
    ctx.obj.plot(fig_size=fig_size, axis=axis, n_ticks=nticks)


@click.command('save')
@click.pass_context
@click.option('--filename',
              type=save_type_hints['filename'],
              default=save_default_args['filename'],
              help="Filename of output image."
              )
@click.option('--extension',
              type=save_type_hints['extension'],
              default=save_default_args['extension'],
              help="Format of output image."
              )
def save_fractal(ctx, filename, extension):
    """Save an image of the set."""
    ctx.obj.save(filename=filename, extension=extension)


@click.command('zoom')
@click.pass_context
@click.option('--magnitude',
              '-m',
              type=zoom_type_hints['m'],
              default=None,  # see function
              show_default=True,
              help='Magnitude of zoom to target location.'
              )
@click.option('--target',
              nargs=2,
              type=get_args(zoom_type_hints['target']),
              default=None,  # see function
              show_default=True,
              help='Target location for zoom (Use --preview to inspect default).'
              )
@click.option('--filename',
              type=zoom_type_hints['filename'],
              default=zoom_default_args['filename'],
              show_default=True,
              help='Output filename.'
              )
@click.option('--extension',
              type=zoom_type_hints['extension'],
              default=zoom_default_args['extension'],
              show_default=True,
              help='Output format.'
              )
@click.option('--frame_subdir',
              type=zoom_type_hints['frame_subdir'],
              default=zoom_default_args['frame_subdir'],
              show_default=True,
              help='Directory to store frames.'
              )
@click.option('--n_frames',
              type=zoom_type_hints['n_frames'],
              default=zoom_default_args['n_frames'],
              show_default=True,
              help='Number of frames for video/gif.'
              )
@click.option('--fps',
              type=zoom_type_hints['fps'],
              default=zoom_default_args['fps'],
              show_default=True,
              help='Framerate of video/gif in frames per second.'
              )
@click.option('--n_jobs',
              type=zoom_type_hints['n_jobs'],
              default=zoom_default_args['n_jobs'],
              show_default=True,
              help='Number of processors for multiprocessing frame generation (default is cpu count).'
              )
@click.option('--preview',
              is_flag=True,
              default=False,
              show_default=True,
              help="Preview target location flag."
              )
@click.option('--nticks',
              default=5,
              type=int,
              show_default=True,
              help="Number of axes ticks (useful for preview)."
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
    """Create a video that zooms in to the set."""
    ctx.obj.magnitude = magnitude if magnitude is not None else get_default_args(ctx.obj.zoom)['m']
    ctx.obj.target = target if target is not None else get_default_args(ctx.obj.zoom)['target']
    if preview:
        ctx.obj.x_min, ctx.obj.x_max, ctx.obj.y_min, ctx.obj.y_max = ctx.obj.get_target_ranges(m=ctx.obj.magnitude,
                                                                                               target=ctx.obj.target
                                                                                               )
        ctx.obj.plot(axis='on', title=f'm={ctx.obj.magnitude:.1e}; target={ctx.obj.target}', n_ticks=nticks)

    else:
        ctx.obj.zoom(filename=filename,
                     extension=extension,
                     frame_subdir=frame_subdir,
                     n_frames=n_frames,
                     fps=fps,
                     n_jobs=n_jobs
                     )

"""This module provides the base fractal class for this package."""

import os
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Union

import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from mpire import WorkerPool

working_path = Path().resolve()


class FractalBase(ABC):
    """An abstract class to represent a base class for fractals."""

    def __init__(self,
                 limits: tuple[float, float, float, float],
                 n_pts: int = 1000,
                 threshold: int = 1000,
                 color_map: str = 'hsv',
                 c_set: str = 'black',
                 pallet_len: int = 250,
                 color_map_shift: int = 0
                 ):
        """Initialises Fractal with either the Mandelbrot set or Julia set along with default attributes.

        Args:
            limits (tuple, optional): Tuple of minimum and maximum values along x-axis and y-axis.
            n_pts (int, optional): Number of points along y-axis. Defaults to 1000.
            threshold (int, optional): Number of iterations before point determined to be in the set. Defaults to 1000.
            color_map (str, optional): Color map for plots. Defaults to 'hsv'.
            pallet_len (int, optional): Length of periodicity for color pallet. Defaults to 250.
            color_map_shift (int, optional): Length to shift color pallet. Defaults to 0.
        """

        self.x_min, self.x_max, self.y_min, self.y_max = limits

        self.n_pts = n_pts
        self.threshold = threshold

        self.set_color = c_set
        self._color_map = None
        self.color_map = color_map
        self.pallet_len = pallet_len
        self.color_map_shift = color_map_shift

    @property
    def ratio(self):
        """A property for the aspect ratio"""
        x_len = abs(self.x_max - self.x_min)
        y_len = abs(self.y_max - self.y_min)
        ratio = x_len / y_len
        return ratio

    @property
    def x_y_ranges(self):
        """A property for the ranges of the x-points and the y-points"""
        x_arr = np.linspace(self.x_min, self.x_max, ceil(self.n_pts * self.ratio))
        y_arr = np.linspace(self.y_max, self.y_min, self.n_pts)
        return x_arr, y_arr

    @property
    def color_map(self):
        return self._color_map

    @color_map.setter
    def color_map(self, c_map):
        """Sets the color map with a mask for the values in the fractalpy set"""
        new_c_map = cm.get_cmap(c_map).copy()
        new_c_map.set_bad(color=self.set_color)
        self._color_map = new_c_map

    @property
    @abstractmethod
    def color_chart(self):
        """The color chart of the calculated set"""

    @abstractmethod
    def plot(self,
             fig_size: float = 5,
             axis: Union[str, bool, tuple[float, float, float, float]] = 'off',
             title: str = '',
             n_ticks: int = 5
             ):
        """Plots the calculated set.

        Args:
            fig_size (int, optional): Size of figure in inches of x-axis for plot. Defaults to 4.
            axis (tuple[float, float, float, float] or bool or string, optional): Axis parameters for plot. Defaults to 'off'.
            title (str, optional): title of plot. Defaults to ''.
            n_ticks (int, optional): Number of ticks for axes. Defaults to 5.
        """
        assert type(n_ticks) == int and n_ticks > 0, 'n_ticks must be a positive integer'
        assert fig_size > 0, 'fig_size must be a positive float or integer'

        fig_size = (fig_size * self.ratio, fig_size)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.imshow(self.color_chart % self.pallet_len,  # Modulo length of pallet for periodicity
                  origin='upper',
                  cmap=self.color_map,
                  vmin=0,
                  vmax=self.pallet_len,
                  aspect='equal')
        ax.axis(axis)
        ax.set_title(title)

        # Set axes ticks. No if statement here due to possible variance in type(axis)
        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.linspace(x_start, x_end, n_ticks))
        ax.set_xticklabels(np.linspace(self.x_min, self.x_max, n_ticks), rotation=90)

        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.linspace(y_start, y_end, n_ticks))
        ax.set_yticklabels(np.linspace(self.y_min, self.y_max, n_ticks))
        plt.show()

    @abstractmethod
    def save(self,
             filename: str = '',
             extension: str = 'png'
             ):
        """Saves an image of the calculated set in the 'images' directory.

        Args:
            filename (_type_): The filename of the saved image. Defaults to None.
            # frame_iter (str, optional): The frame iteration number. Defaults to ''.
            extension (str, optional): The extension to save the image as. Defaults to 'png'.
        """
        self._make_dir(os.path.join('FractalPy_Outputs', 'images'))
        fname = os.path.join('FractalPy_Outputs', 'images', f'{filename}.{extension}')
        plt.imsave(fname=fname,
                   arr=self.color_chart % self.pallet_len,
                   origin='upper',
                   cmap=self.color_map,
                   vmin=0,
                   vmax=self.pallet_len,
                   format=extension)
        print(f'Image saved at {os.path.join(working_path, fname)}')

    @staticmethod
    def _make_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # directory already exists
            pass

    @abstractmethod
    def zoom(self,
             m: float,
             target: tuple[float, float],
             filename: str = '',
             extension: str = 'gif',
             frame_subdir: str = 'frames',
             n_frames: int = 120,
             fps: int = 60,
             n_jobs: int = os.cpu_count()
             ):
        """Compiles a video after generating a sequence of images, beginning with the object's current co-ordinate
        frame and finishing at the target location frame.

        Args:
            filename (str, optional): Name of output file. Defaults to None.
            extension (str, optional): Extension of the output file format. Defaults to 'gif'.
            frame_subdir (str, optional): Directory to save image frames. Defaults to 'frames'.
            m (float, optional): the zoom magnitude.
            target (tuple[float, float], optional): Target location for the end of zoom.
                The first tuple entry is  location with x, y coordinates. Defaults to (6e+4, -1.186592e+0, -1.901211e-1).
                # TODO fix the defaults here. it depends on which class
            n_frames (int, optional): Number of total frames compiled. n_frames-2 frames are compiled intermediately
                between the initial frame and target frame. Defaults to 120.
            fps (int, optional): Number of frames per second for output video. This determines the smoothness between
                frames and affects the length of the output video. Defaults to 60.
            n_jobs (int, optional): Number of workers for parallel processing. Defaults to os.cpu_count().
        """
        self._make_dir(os.path.join('FractalPy_Outputs', 'frames'))

        # Get target ranges from target parameters
        x_target_min, x_target_max, y_target_min, y_target_max = self.get_target_ranges(m, target)

        # TODO think about how to describe this. This is the most complex and important line in the package
        geom = np.flip(1 - (np.geomspace(1, m, n_frames) - 1) / (m - 1))

        # Create a geometric sequence of frames such that corresponding zoom is smooth
        x_ranges = [(x0, x1) for (x0, x1) in
                    zip(self.x_min + geom * (x_target_min - self.x_min),
                        self.x_max + geom * (x_target_max - self.x_max))]
        y_ranges = [(y0, y1) for (y0, y1) in
                    zip(self.y_min + geom * (y_target_min - self.y_min),
                        self.y_max + geom * (y_target_max - self.y_max))]

        if n_jobs == 1:
            for i in range(n_frames):
                self._single_zoom_frame(i, x_ranges[i], y_ranges[i])

        else:
            # MULTIPROCESSING
            """
            inputs = zip(range(n_frames), x_ranges, y_ranges)
            with Pool() as p:
                list(tqdm(p.imap_unordered(self._single_zoom_frame, inputs), total=n_frames))
    
            inputs = zip(range(n_frames), x_ranges, y_ranges)
            p_umap(self._single_zoom_frame, inputs)
            """

            # Pack up frame data to be carried out by workers
            inputs = zip(range(n_frames), x_ranges, y_ranges)

            # Assign workers to generate image frames in a multiprocessing configuration
            with WorkerPool(n_jobs=n_jobs) as pool:
                pool.map(self._single_zoom_frame, inputs, progress_bar=True, iterable_len=n_frames)

        # Assign default filename TODO: decide on consistency with where to determine default filename and fix the details of the filename along with adding the magnitude
        if not filename:
            filename = str(f'zoom_{target}_{self.threshold}thresh_{self.n_pts}pts_{n_frames}frames_{fps}fps')
        filename = str(filename).replace('.', ',').replace(' ', '')

        # Compile video of generated image frames
        if extension == 'gif':
            self._build_gif(filename=filename,
                            n_frames=n_frames,
                            fps=fps,
                            end_buffer=2 * fps
                            )
        else:
            self._build_vid(filename=filename,
                            extension=extension,
                            n_frames=n_frames,
                            fps=fps
                            )

    def get_target_ranges(self, m: float, target: tuple[float, float]):
        """Gets the x range and y range for the target point at corresponding zoom magnitude"""
        (x, y) = target

        x_target_len = abs(self.x_max - self.x_min) / m
        x_target = (x - x_target_len / 2, x + x_target_len / 2)

        y_target_len = abs(self.y_max - self.y_min) / m
        y_target = (y - y_target_len / 2, y + y_target_len / 2)

        return x_target[0], x_target[1], y_target[0], y_target[1]

    def _single_zoom_frame(self, i, x_cur, y_cur):
        """Generates image frame of current x range and y range"""
        self.x_min, self.x_max = x_cur
        self.y_min, self.y_max = y_cur
        self._save_frame(frame_iter=i)

    def _save_frame(self, frame_iter, extension='png'):
        fname = os.path.join('FractalPy_Outputs', 'frames', f'frame{frame_iter}.{extension}')
        plt.imsave(fname=fname,
                   arr=self.color_chart % self.pallet_len,
                   origin='upper',
                   cmap=self.color_map,
                   vmin=0,
                   vmax=self.pallet_len,
                   format=extension)

    @classmethod
    def _build_gif(cls, filename, n_frames, fps, end_buffer):
        """Compiles gif from images located in the frame subdirectory"""
        print('Compiling gif...')

        cls._make_dir(os.path.join('FractalPy_Outputs', 'gifs'))
        save_rel_path = os.path.join('FractalPy_Outputs', 'gifs', f'{filename}.gif')

        with iio.get_writer(save_rel_path, mode='I', fps=fps) as writer:
            for frame in [os.path.join('FractalPy_Outputs', 'frames', f'frame{i}.png') for i in range(n_frames)]:
                image = iio.v3.imread(frame)
                writer.append_data(image)
            for _ in range(end_buffer):
                writer.append_data(image)

        print(f'Completed, gif saved at {os.path.join(working_path, save_rel_path)}')

    @classmethod
    def _build_vid(cls, filename, extension, n_frames, fps):
        """Compiles video from images located in the frame subdirectory"""
        print(f'Compiling {extension} video...')

        # Makes filename compatible with ffmpeg
        filename = filename.replace('(', '\(').replace(')', '\)')

        cls._make_dir(os.path.join('FractalPy_Outputs', 'videos'))
        frame_dir = os.path.join('FractalPy_Outputs', 'frames', 'frame%d.png')
        save_rel_path = os.path.join('FractalPy_Outputs', 'videos', f'{filename}.{extension}')

        list_of_commands = [f'-framerate {fps} -i',
                            f'{frame_dir}',
                            f'-frames:v {n_frames}',
                            f'-c:v libx265',
                            f'-vtag hvc1',
                            f'-filter:v "scale=in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv"',
                            f'-pix_fmt:v "yuv420p"',
                            f'-colorspace:v "bt709"',
                            f'-color_primaries:v "bt709"',
                            f'-color_trc:v "bt709"',
                            f'-color_range:v "tv"',
                            f'-c:a copy',
                            f'-r {fps}',
                            f'{save_rel_path}'
                            ]
        command = "ffmpeg " + " ".join(list_of_commands)
        os.system(command)

        print(f'Completed, video saved at {os.path.join(working_path, save_rel_path)}')

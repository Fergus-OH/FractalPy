import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
import imageio as iio
import os
from mpire import WorkerPool
import numba as nb
from math import ceil


class Fractal:
    """A class to represent the Mandelbrot set or Julia set fractals."""
    def __init__(self, julia=False, c=(-0.79 + 0.15j), x_ran=None, y_ran=None, n_pts=1000, threshold=1000, c_map='hsv',
                 pallet_len=250, shift=0):
        """Initialises Fractal with either the Mandelbrot set or Julia set along with default attributes.

        Args:
            julia (bool, optional): Sets mode to Julia set if true and Mandelbrot set if False. Defaults to False.
            c (tuple, optional): Julia set parameter. Defaults to (-0.79 + 0.15j).
            x_ran (tuple, optional): Tuple of minimum and maximum values along x-axis. Defaults to None.
            y_ran (tuple, optional): Tuple of minimum and maximum values along y-axis. Defaults to None.
            n_pts (int, optional): Number of points along y-axis. Defaults to 1000.
            threshold (int, optional): Number of iterations before point determined to be in the set. Defaults to 1000.
            c_map (str, optional): Color map for plots. Defaults to 'hsv'.
            pallet_len (int, optional): Length of periodicity for color pallet. Defaults to 250.
            shift (int, optional): Length to shift color pallet. Defaults to 0.
        """

        print('Initialising object...')
        self.julia = julia
        self.c = c if julia else None
        self.x_ran, self.y_ran = self._get_default_ranges(x_ran, y_ran)
        self.n_pts = n_pts
        self.threshold = threshold
        self.c_map = self._get_c_map(c_map)
        self.pallet_len = pallet_len
        self.shift = shift
        self.color_chart = self._determine_color_chart()
        print('Object initialised.')

    def _get_default_ranges(self, x_ran, y_ran):
        if not self.julia:
            x_ran = (-2, 1) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        else:
            x_ran = (-1.5, 1.5) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        return x_ran, y_ran

    def _determine_color_chart(self):
        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, ceil(self.n_pts * x_len / y_len))
        y_arr = np.linspace(y_max, y_min, self.n_pts)

        @nb.jit(nopython=True, parallel=True)
        def _mandel_chart(threshold):
            c_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in nb.prange(len(y_arr)):
                for j in nb.prange(len(x_arr)):
                    c = complex(x_arr[j], y_arr[i])
                    z = 0.0j
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            c_chart[i, j] = k
                            break
            return c_chart

        @nb.jit(nopython=True, parallel=True)
        def _julia_chart(threshold, c):
            c_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in nb.prange(len(y_arr)):
                for j in nb.prange(len(x_arr)):
                    z = complex(x_arr[j], y_arr[i])
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            c_chart[i, j] = k
                            break
            return c_chart

        color_chart = _mandel_chart(threshold=self.threshold) if not self.julia \
            else _julia_chart(threshold=self.threshold, c=self.c)

        color_chart = np.ma.masked_where(color_chart == -1, color_chart)
        color_chart += self.shift
        return color_chart

    def plot(self, axis='off', fig_size=None, dpi=150):
        """Plots the set calculated according to the objects parameters.

        Args:
            axis (str, optional): Axis parameters for plot. Defaults to 'off'.
            fig_size (tuple, optional): Size of figure in inches for plot. Defaults to None.
            dpi (int, optional): DPI of plot. Defaults to 100.
        """
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        ax.imshow(self.color_chart % self.pallet_len, origin='upper', cmap=self.c_map, vmin=0, vmax=self.pallet_len,
                  aspect='equal')
        ax.axis(axis)

        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.linspace(x_start, x_end, 5))
        ax.set_xticklabels(np.linspace(self.x_ran[0], self.x_ran[1], 5), rotation=60)

        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.linspace(y_start, y_end, 5))
        ax.set_yticklabels(np.linspace(self.y_ran[0], self.y_ran[1], 5))
        plt.show()

    def save(self, subdir='', filename=None, frame_iter='', extension='png'):
        """Saves the image of the set in the './images' directory.

        Args:
            subdir (str, optional): A subdirectory of './images' to save the image within. Defaults to ''.
            filename (_type_, optional): The filename of the saved image. Defaults to None.
            frame_iter (str, optional): The frame iteration number. Defaults to ''.
            extension (str, optional): The extension to save the image as. Defaults to 'png'.
        """
        self._make_dir(os.path.join('images', subdir))
        # setting the default filename
        if not filename:
            filename = str(f'Mandelbrot_{self.n_pts}pts_{self.threshold}threshold') if not self.julia \
                else str(f'Julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold')
            filename = str(filename).replace('.', ',').replace(' ', '')

        plt.imsave(fname=f'images/{subdir}/{filename}{frame_iter}.{extension}', arr=self.color_chart % self.pallet_len,
                   origin='upper', cmap=self.c_map, vmin=0, vmax=self.pallet_len, format=extension)

    @staticmethod
    def _get_c_map(c_map):
        new_c_map = cmx.get_cmap(c_map).copy()
        # Creating a new color map that maps masked values to black
        new_c_map.set_bad(color='black')
        return new_c_map

    @staticmethod
    def _make_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # directory already exists
            pass

    def zoom(self, filename=None, extension='gif', frame_subdir='frames', target=(6e+4, -1.186592e+0, -1.901211e-1),
             n_frames=120,
             fps=60, n_jobs=os.cpu_count()):
        """Compiles a video after generating a sequence of images, beginning with the object's current co-ordinate
        frame and finishing at the target location.

        Args:
            filename (str, optional): Name of output file. Defaults to None.
            extension (str, optional): Extension of the output file format. Defaults to 'gif'.
            frame_subdir (str, optional): Directory to save image frames. Defaults to 'frames'.
            target (tuple, optional): Target location for the end of zoom. The first tuple entry is the zoom magnitude
                of the target location with x, y coordinates. Defaults to (6e+4, -1.186592e+0, -1.901211e-1).
            n_frames (int, optional): Number of total frames compiled. n_frames-2 frames are compiled intermediately
                between the initial frame and target frame. Defaults to 120.
            fps (int, optional): Number of frames per second for output video. This determines the smoothness between
                frames and affects the length of the output video. Defaults to 60.
            n_jobs (int, optional): Number of workers for parallel processing. Defaults to os.cpu_count().
        """

        # Unpacking zoom scale and target co-ordinates from target parameters
        # (m, x, y) = target

        # x_target_len = abs(self.x_ran[1] - self.x_ran[0]) / m
        # x_target = (x - x_target_len / 2, x + x_target_len / 2)

        # y_target_len = abs(self.y_ran[1] - self.y_ran[0]) / m
        # y_target = (y - y_target_len / 2, y + y_target_len / 2)

        x_target, y_target = self.get_target_ranges(target)
        m = target[0]

        # Creating a geometric sequence for frames to correspond to smooth zooming
        geom = np.flip(1 - (np.geomspace(1, m, n_frames) - 1) / (m - 1))
        x_ranges = [(x0, x1) for (x0, x1) in
                    zip(self.x_ran[0] + geom * (x_target[0] - self.x_ran[0]),
                        self.x_ran[1] + geom * (x_target[1] - self.x_ran[1]))]
        y_ranges = [(y0, y1) for (y0, y1) in
                    zip(self.y_ran[0] + geom * (y_target[0] - self.y_ran[0]),
                        self.y_ran[1] + geom * (y_target[1] - self.y_ran[1]))]

        # MULTIPROCESSING
        """
        inputs = zip(range(n_frames), x_ranges, y_ranges)
        with Pool() as p:
            list(tqdm(p.imap_unordered(self._single_zoom_frame, inputs), total=n_frames))

        inputs = zip(range(n_frames), x_ranges, y_ranges)
        p_umap(self._single_zoom_frame, inputs)
        """

        inputs = zip(range(n_frames), x_ranges, y_ranges)
        with WorkerPool(n_jobs=n_jobs) as pool:
            pool.map(self._single_zoom_frame, inputs, progress_bar=True, iterable_len=n_frames)

        if not filename:
            filename = str(f'zoom_{target}_{self.threshold}thresh_{self.n_pts}pts_{n_frames}frames_{fps}fps')
        filename = str(filename).replace('.', ',').replace(' ', '')

        if extension == 'gif':
            self._build_gif(filename=filename, frame_subdir=frame_subdir, n_frames=n_frames, fps=fps,
                            end_buffer=2 * fps)
        else:
            self._build_vid(filename=filename, extension=extension, frame_subdir=frame_subdir, n_frames=n_frames,
                            fps=fps)

    def _single_zoom_frame(self, i, x_cur, y_cur):
        self.x_ran = x_cur
        self.y_ran = y_cur

        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames', frame_iter=i)

    def get_target_ranges(self, target):
        (m, x, y) = target

        x_target_len = abs(self.x_ran[1] - self.x_ran[0]) / m
        x_target = (x - x_target_len / 2, x + x_target_len / 2)

        y_target_len = abs(self.y_ran[1] - self.y_ran[0]) / m
        y_target = (y - y_target_len / 2, y + y_target_len / 2)

        return x_target, y_target

    def spin(self, filename=None, extension='gif', frame_subdir='frames', n_frames=60, fps=60, n_jobs=os.cpu_count()):
        """Creates a sequence of images beginning with the objects current co-ordinate frame and finishing at the target
         location.

        Args:
            filename (_type_, optional): _description_. Defaults to None.
            extension (str, optional): _description_. Defaults to 'gif'.
            frame_subdir (str, optional): _description_. Defaults to 'frames'.
            n_frames (int, optional): _description_. Defaults to 60.
            fps (int, optional): _description_. Defaults to 60.
            n_jobs (_type_, optional): _description_. Defaults to os.cpu_count().
        """

        a_ran = np.linspace(0, 2 * np.pi, n_frames)
        modulus = np.abs(self.c)
        c_ran = modulus * np.exp(1j * a_ran)

        # multiprocessing
        inputs = zip(range(n_frames), c_ran)
        with WorkerPool(n_jobs=n_jobs) as pool:
            pool.map(self._single_spin_frame, inputs, progress_bar=True, iterable_len=n_frames)

        if not filename:
            filename = str(f'spin_{self.c}_{self.threshold}thresh_{self.n_pts}pts_{n_frames}frames_{fps}fps')
        filename = str(filename).replace('.', ',').replace(' ', '')

        if extension == 'gif':
            self._build_gif(filename=filename, frame_subdir=frame_subdir, n_frames=n_frames, fps=fps, end_buffer=0)
        else:
            self._build_vid(filename=filename, extension=extension, frame_subdir=frame_subdir, n_frames=n_frames,
                            fps=fps)

    def _single_spin_frame(self, i, c):
        self.c = c
        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames', frame_iter=i)

    @staticmethod
    def _build_gif(filename, frame_subdir, n_frames, fps, end_buffer):
        """Compiles gif from images located in the frame subdirectory"""
        print('Compiling gif...')
        Fractal._make_dir('gifs')
        with iio.get_writer(f'gifs/{filename}.gif', mode='I', fps=fps) as writer:
            for frame in [f'images/{frame_subdir}/frame{i}.png' for i in range(n_frames)]:
                image = iio.v3.imread(frame)
                writer.append_data(image)
            for _ in range(end_buffer):
                writer.append_data(image)
        print(f'Completed, gif saved at \'gifs/{filename}.gif\'')

    @staticmethod
    def _build_vid(filename, extension, frame_subdir, n_frames, fps):
        """Compiles video from images located in the frame subdirectory"""
        Fractal._make_dir('videos')
        filename = filename.replace('(', '\(').replace(')', '\)')
        list_of_commands = [
            f'-framerate {fps} -i',
            f'./images/{frame_subdir}/frame%d.png',
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
            f'./videos/{filename}.{extension}'
        ]
        command = "ffmpeg " + " ".join(list_of_commands)
        os.system(command)

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
import imageio as iio
import os
from mpire import WorkerPool
import numba as nb


class Mandelbrot:
    """_summary_
    """

    def __init__(self, julia=False, c=(-0.79 + 0.15j), x_ran=None, y_ran=None, n_pts=1000, threshold=1000, c_map='hsv',
                 pallet_len=250):
        """_summary_

        Args:
            julia (bool, optional): Sets mode to Julia set if true and Mandelbrot set if False. Defaults to False.
            c (tuple, optional): _description_. Defaults to (-0.79 + 0.15j).
            x_ran (tuple, optional): Tuple of minimum and maximum values along x-axis. Defaults to None.
            y_ran (tuple, optional): Tuple of minimum and maximum values along y-axis. Defaults to None.
            n_pts (int, optional): Number of points along x-axis. Defaults to 1000.
            threshold (int, optional): Number of iterations before determined as being in the set. Defaults to 1000.
            c_map (str, optional): Color map for plotting. Defaults to 'hsv'.
            pallet_len (int, optional): Periodic length of pallet. Defaults to 250.
        """
        self.julia = julia
        self.c = c if julia else None
        self.x_ran, self.y_ran = self._get_default_ranges(x_ran, y_ran)
        self.n_pts = n_pts
        self.threshold = threshold
        self.c_map = self._get_c_map(c_map)
        self.pallet_len = pallet_len
        self.color_chart = self._determine_color_chart()
        print('Object initialised, call plot() method to plot image or save() method to save in images directory...')

    def _get_default_ranges(self, x_ran, y_ran):
        """_summary_

        Args:
            x_ran (_type_): _description_
            y_ran (_type_): _description_

        Returns:
            _type_: _description_
        """
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

        x_arr = np.linspace(x_min, x_max, self.n_pts)
        y_arr = np.linspace(y_max, y_min, int(self.n_pts * y_len / x_len))

        # Retrieve attributes for jit function

        @nb.jit(nopython=True)
        def _mandel_chart(threshold):
            c_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in range(len(y_arr)):
                for j in range(len(x_arr)):
                    c = complex(x_arr[j], y_arr[i])
                    z = 0.0j
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            c_chart[i, j] = k
                            break
            return c_chart

        @nb.jit(nopython=True)
        def _julia_chart(threshold, c):
            c_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in range(len(y_arr)):
                for j in range(len(x_arr)):
                    z = complex(x_arr[j], y_arr[i])
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            c_chart[i, j] = k
                            break
            return c_chart

        """
        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, self.n_pts)
        y_arr = np.linspace(y_max, y_min, int(self.n_pts * y_len / x_len))

        color_chart = _mandel_chart(x_arr=x_arr, y_arr=y_arr, threshold=self.threshold) if not self.julia\
            else _julia_chart(x_arr=x_arr, y_arr=y_arr, threshold=self.threshold, c=self.c)
        """

        color_chart = _mandel_chart(threshold=self.threshold) if not self.julia \
            else _julia_chart(threshold=self.threshold, c=self.c)
        # if not self.julia:
        #     _mandel_chart()
        # else:
        #     _julia_chart()

        color_chart = np.ma.masked_where(color_chart == -1, color_chart)
        return color_chart

    # Plot and save features
    def plot(self, axis='off', fig_size=None, dpi=100):
        """_summary_

        Args:
            axis (str, optional): _description_. Defaults to 'off'.
            fig_size (_type_, optional): _description_. Defaults to None.
            dpi (int, optional): _description_. Defaults to 100.
        """
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        # c_map = self._get_c_map(self.c_map)
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
        """_summary_

        Args:
            subdir (str, optional): _description_. Defaults to ''.
            filename (_type_, optional): _description_. Defaults to None.
            frame_iter (str, optional): _description_. Defaults to ''.
            extension (str, optional): _description_. Defaults to 'png'.
        """
        # c_map = self._get_c_map(self.c_map)
        self._make_dir(os.path.join('images', subdir))
        # setting the default filename
        if not filename:
            filename = str(f'Mandelbrot_{self.n_pts}pts_{self.threshold}threshold') if not self.julia \
                else str(f'Julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold')
            filename = str(filename).replace('.', ',').replace(' ', '')

        """
        if not self.julia:
            filename = str(f'Mandelbrot_{self.n_pts}pts_{self.threshold}threshold')\
                if not filename else str(filename)
        else:
            filename = str(f'Julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold')\
            if not filename else str(filename)
        filename = filename.replace('.', ',').replace(' ', '')
        """

        plt.imsave(fname=f'images/{subdir}/{filename}{frame_iter}.{extension}', arr=self.color_chart % self.pallet_len,
                   origin='upper', cmap=self.c_map, vmin=0, vmax=self.pallet_len, format=extension)

    @staticmethod
    def _get_c_map(c_map):
        """Determines color map for plotting

        Args:
            c_map (_type_): Colormap

        Returns:
            _type_: Colormap for masked data points
        """
        new_c_map = cmx.get_cmap(c_map).copy()
        new_c_map.set_bad(color='black')
        return new_c_map

    @staticmethod
    def _make_dir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            # directory already exists
            pass

    # Zoom and spin features
    def zoom(self, filename=None, extension='gif', frame_subdir='frames', target=(6e+4, -1.186592e+0, -1.901211e-1),
             n_frames=120,
             fps=60, n_jobs=os.cpu_count()):
        """_summary_

        Args:
            filename (_type_, optional): Name of output gif file. Defaults to None.
            extension (, optional): Extension. Defaults to 'gif'.
            frame_subdir (str, optional): Directory to save frames. Defaults to 'frames'.
            target (tuple, optional): Target zoom location. Defaults to (6e+4, -1.186592e+0, -1.901211e-1).
            n_frames (int, optional): Number of frames. Defaults to 120.
            fps (int, optional): Frames per second for output gif. Defaults to 60.
            n_jobs (_type_, optional): Number of workers for parallel processing. Defaults to os.cpu_count().
        """

        # manager_port_nr=None,
        # if manager_port_nr:
        #     connect_to_dashboard(manager_port_nr=manager_port_nr, manager_host='localhost')

        (m, x, y) = target

        x_target_len = abs(self.x_ran[1] - self.x_ran[0]) / m
        x_target = (x - x_target_len / 2, x + x_target_len / 2)

        y_target_len = abs(self.y_ran[1] - self.y_ran[0]) / m
        y_target = (y - y_target_len / 2, y + y_target_len / 2)

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
        """_summary_

        Args:
            i (_type_): _description_
            x_cur (_type_): _description_
            y_cur (_type_): _description_
        """
        # def _single_zoom_frame(self, inputs):
        # i, self.x_ran, self.y_ran = inputs
        self.x_ran = x_cur
        self.y_ran = y_cur

        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames', frame_iter=i)

    def spin(self, filename=None, extension='gif', frame_subdir='frames', n_frames=60, fps=60, n_jobs=os.cpu_count()):
        """_summary_

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
        """_summary_

        Args:
            i (_type_): _description_
            c (_type_): _description_
        """
        self.c = c
        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames', frame_iter=i)

    # Creating gif
    @staticmethod
    def _build_gif(filename, frame_subdir, n_frames, fps, end_buffer):
        """Compiles gif from frames

        Args:
            filename (_type_): _description_
            frame_subdir (_type_): _description_
            n_frames (_type_): _description_
            fps (_type_): _description_
            end_buffer (_type_): _description_
        """
        print('Compiling gif...')
        Mandelbrot._make_dir('gifs')
        with iio.get_writer(f'gifs/{filename}.gif', mode='I', fps=fps) as writer:
            for frame in [f'images/{frame_subdir}/frame{i}.png' for i in range(n_frames)]:
                image = iio.v3.imread(frame)
                writer.append_data(image)
            for _ in range(end_buffer):
                writer.append_data(image)
        print(f'Completed, gif saved at \'gifs/{filename}.gif\'')

    @staticmethod
    def _build_vid(filename, extension, frame_subdir, n_frames, fps):
        """_summary_

        Args:
            filename (_type_): _description_
            extension (_type_): _description_
            frame_subdir (_type_): _description_
            n_frames (_type_): _description_
            fps (_type_): _description_
        """
        Mandelbrot._make_dir('videos')
        filename = filename.replace('(', '\(').replace(')', '\)')
        cmd = f'ffmpeg -framerate {fps} -i ' \
              f'./images/{frame_subdir}/frame%d.png ' \
              f'-frames:v {n_frames} ' \
              f'-c:v libx265 ' \
              f'-vtag hvc1 ' \
              f'-filter:v "scale=in_color_matrix=auto:in_range=auto:out_color_matrix=bt709:out_range=tv" ' \
              f'-pix_fmt:v "yuv420p" ' \
              f'-colorspace:v "bt709" ' \
              f'-color_primaries:v "bt709" ' \
              f'-color_trc:v "bt709" ' \
              f'-color_range:v "tv" ' \
              f'-c:a copy ' \
              f'-r {fps} ' \
              f'./videos/{filename}.{extension} '
        os.system(f'/bin/bash -c "{cmd}"')

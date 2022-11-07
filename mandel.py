import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np
import imageio as iio
import os
from mpire import WorkerPool
from mpire.dashboard import connect_to_dashboard
import warnings
import numba as nb
from p_tqdm import p_umap
from multiprocessing import Pool
from tqdm import tqdm

# Maybe see if this can be changed to just overflow error, not the category.
warnings.filterwarnings("ignore", category=RuntimeWarning)


# connect_to_dashboard(8099)


class Mandelbrot:
    def __init__(self, julia=False, c=(-0.79 + 0.15j), x_ran=None, y_ran=None, n_pts=1000, threshold=1000):
        self.julia = julia
        self.c = c if julia else None
        self.x_ran, self.y_ran = self._get_default_ranges(x_ran, y_ran)
        self.n_pts = n_pts
        self.threshold = threshold
        self.color_chart = self._determine_color_chart()
        print('Object initialised, call plot() method to plot image or save() method to save in images directory...')

    def _get_default_ranges(self, x_ran, y_ran):
        if not self.julia:
            x_ran = (-2, 1) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        else:
            x_ran = (-1.5, 1.5) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        return x_ran, y_ran

    def _get_default_dirs(self, filename, frame_subdir):
        if not filename:
            filename = str(f'mandelbrot_{self.x_ran}_{self.y_ran}_{self.n_pts}') if not self.julia else str(
                f'julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',')
        return filename, frame_subdir

    def _determine_color_chart(self):
        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, self.n_pts)
        y_arr = np.linspace(y_max, y_min, int(self.n_pts * y_len / x_len))

        color_chart = self._speed(x_arr=x_arr, y_arr=y_arr, threshold=self.threshold, julia=self.julia, c=self.c)
        color_chart = np.ma.masked_where(color_chart == -1, color_chart)
        return color_chart


    def _speed(self, x_arr, y_arr, threshold, julia, c):
        @nb.jit(nopython=True)
        def mandel_chart(x_arr, y_arr, threshold):
            color_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in range(len(y_arr)):
                for j in range(len(x_arr)):
                    c = complex(x_arr[j], y_arr[i])
                    z = 0.0j
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            color_chart[i, j] = k
                            break
            return color_chart

        @nb.jit(nopython=True)
        def julia_chart(x_arr, y_arr, threshold, c):
            color_chart = np.zeros((len(y_arr), len(x_arr))) - 1
            for i in range(len(y_arr)):
                for j in range(len(x_arr)):
                    z = complex(x_arr[j], y_arr[i])
                    for k in range(threshold):
                        z = z * z + c
                        if (z.real * z.real + z.imag * z.imag) >= 4:
                            color_chart[i, j] = k
                            break
            return color_chart
        return mandel_chart(x_arr, y_arr, threshold) if not julia else julia_chart(x_arr, y_arr, threshold, c)

    @staticmethod
    def _get_c_map(c_map):
        new_c_map = cmx.get_cmap(c_map).copy()
        new_c_map.set_bad(color='black')
        return new_c_map

    def plot(self, c_map='hsv', pallet_len=250, axis='off', fig_size=None, dpi=100):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        c_map = self._get_c_map(c_map)
        ax.imshow(self.color_chart % pallet_len, origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, aspect='equal')
        ax.axis(axis)

        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.linspace(x_start, x_end, 5))
        ax.set_xticklabels(np.linspace(self.x_ran[0], self.x_ran[1], 5), rotation=60)

        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.linspace(y_start, y_end, 5))
        ax.set_yticklabels(np.linspace(self.y_ran[0], self.y_ran[1], 5))
        plt.show()

    def save(self, subdir='', filename=None, frame_iter='', extension='png', c_map='hsv', pallet_len=250):
        c_map = self._get_c_map(c_map)
        # setting the default filename
        if not self.julia:
            filename = str(f'Mandelbrot_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',')\
                if not filename else str(filename)
        else:
            filename = str(f'Julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',')\
                if not filename else str(filename)
        plt.imsave(fname=f'images/{subdir}{filename}{frame_iter}.{extension}', arr=self.color_chart % pallet_len,
                   origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, format=extension)

    def zoom(self, filename=None, frame_subdir='frames', target=(6e+4, -1.186592e+0, -1.901211e-1), n_frames=120,
             manager_port_nr=None, n_jobs=os.cpu_count()):
        filename = str(filename) if filename else str(f'zoom_{target}_{n_frames}_frames')
        if manager_port_nr:
            connect_to_dashboard(manager_port_nr=manager_port_nr, manager_host='localhost')

        (m, x, y) = target

        x_target_len = abs(self.x_ran[1] - self.x_ran[0]) / m
        x_target = (x - x_target_len / 2, x + x_target_len / 2)

        y_target_len = abs(self.y_ran[1] - self.y_ran[0]) / m
        y_target = (y - y_target_len / 2, y + y_target_len / 2)

        geom = np.flip(1 - (np.geomspace(1, m, n_frames) - 1) / (m - 1))
        x_ranges = [(x0, x1) for (x0, x1) in
                    zip(self.x_ran[0] + geom * (x_target[0] - self.x_ran[0]), self.x_ran[1] + geom * (x_target[1] - self.x_ran[1]))]
        y_ranges = [(y0, y1) for (y0, y1) in
                    zip(self.y_ran[0] + geom * (y_target[0] - self.y_ran[0]), self.y_ran[1] + geom * (y_target[1] - self.y_ran[1]))]

        # MULTIPROCESSING

        # inputs = zip(range(n_frames), x_ranges, y_ranges)
        # with Pool() as p:
        #     list(tqdm(p.imap_unordered(self._single_frame, inputs), total=n_frames))

        # inputs = zip(range(n_frames), x_ranges, y_ranges)
        # p_umap(self._single_frame, inputs)

        inputs = zip(range(n_frames), x_ranges, y_ranges)
        with WorkerPool(n_jobs=n_jobs) as pool:
            pool.map(self._single_frame, inputs, progress_bar=True, iterable_len=n_frames)

        self.build_gif(filename=filename, frame_subdir=frame_subdir, n_frames=n_frames)

    def _single_frame(self, i, x_cur, y_cur):
    # def _single_frame(self, inputs):
        # i, self.x_ran, self.y_ran = inputs
        self.x_ran = x_cur
        self.y_ran = y_cur

        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames/', frame_iter=i)

    def spin(self, filename=None, frame_subdir='frames', n_frames=60, n_jobs=os.cpu_count()):
        filename = str(filename) if filename else str(f'spin_{self.c}_{n_frames}_frames')

        a_ran = np.linspace(0, 2 * np.pi, n_frames)
        modulus = np.abs(self.c)
        c_ran = modulus * np.exp(1j * a_ran)
        inputs = zip(range(n_frames), c_ran)
        with WorkerPool(n_jobs=n_jobs) as pool:
            pool.map(self._single_spin_frame, inputs, progress_bar=True, iterable_len=n_frames)
        self.build_gif(filename=filename, frame_subdir=frame_subdir, n_frames=n_frames)

    def _single_spin_frame(self, i, c):
        self.c = c
        self.color_chart = self._determine_color_chart()
        self.save(filename='frame', subdir='frames/', frame_iter=i)

    @staticmethod
    def build_gif(filename, frame_subdir, n_frames):
        with iio.get_writer(f'videos/{filename}.gif', mode='I') as writer:
            for frame in [f'images/{frame_subdir}/frame{i}.png' for i in range(n_frames)]:
                image = iio.v3.imread(frame)
                writer.append_data(image)
        print(f'Completed, video saved at \'videos/{filename}.gif\'')

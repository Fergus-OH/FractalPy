"""This module provides the usage classes for this package."""

import os

import numba as nb
import numpy as np
from mpire import WorkerPool

from .fractal_base import FractalBase


class Mandelbrot(FractalBase):
    """A class to represent the Mandelbrot set."""

    def __init__(self, limits: tuple[float, float, float, float] = (-2, 1, -1.5, 1.5), **kwargs):
        super().__init__(limits, **kwargs)
        self._color_chart = None

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _calculate_mandelbrot(x_y_ranges, threshold):
        x_arr, y_arr = x_y_ranges
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

    @property
    def color_chart(self):
        # To avoid over complication, we don't define the jit function as a class method
        color_chart = self._calculate_mandelbrot(x_y_ranges=self.x_y_ranges, threshold=self.threshold)
        color_chart = np.ma.masked_where(color_chart == -1, color_chart)
        color_chart += self.color_map_shift
        return color_chart

    def plot(self, **kwargs):
        super().plot(**kwargs)

    def save(self, filename='', **kwargs):
        if not filename:
            filename = str(f'{self.__class__.__name__}_{self.n_pts}pts_{self.threshold}threshold')
        super().save(filename, **kwargs)

    def zoom(self, m=6e+4, target=(-1.186592e+0, -1.901211e-1), **kwargs):
        super().zoom(m, target, **kwargs)


class Julia(FractalBase):
    """A class to represent the Julia set."""

    def __init__(self,
                 c: complex = -0.79 + 0.15j,
                 limits: tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
                 **kwargs
                 ):
        super().__init__(limits, **kwargs)
        self.c = c

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _calculate_julia(x_y_ranges, threshold, c):
        x_arr, y_arr = x_y_ranges
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

    @property
    def color_chart(self):
        color_chart = self._calculate_julia(x_y_ranges=self.x_y_ranges, threshold=self.threshold, c=self.c)
        color_chart = np.ma.masked_where(color_chart == -1, color_chart)
        color_chart += self.color_map_shift
        return color_chart

    def plot(self, **kwargs):
        super().plot(**kwargs)

    def save(self, filename='', **kwargs):
        if not filename:
            filename = str(f'{self.__class__.__name__}_{self.c}_{self.n_pts}pts_{self.threshold}threshold')
            filename = str(filename).replace('.', ',').replace(' ', '')  # TODO: is the second replace necessary anymore
        super().save(filename=filename, **kwargs)

    def zoom(self, m=5, target=(.5, .5), **kwargs):
        super().zoom(m, target, **kwargs)

    def spin(self,
             filename: str = None,
             extension: str = 'gif',
             n_frames: int = 60,
             fps: int = 60,
             n_jobs: int = os.cpu_count()
             ):
        """Creates a sequence of images beginning with the objects current co-ordinate frame and finishing at the target
         location.

        Args:
            filename (_type_, optional): _description_. Defaults to None.
            extension (str, optional): _description_. Defaults to 'gif'.
            n_frames (int, optional): _description_. Defaults to 60.
            fps (int, optional): _description_. Defaults to 60.
            n_jobs (_type_, optional): _description_. Defaults to os.cpu_count().
        """

        a_ran = np.linspace(0, 2 * np.pi, n_frames)
        modulus = np.abs(self.c)
        c_ran = modulus * np.exp(1j * a_ran)

        if n_jobs == 1:
            for i in range(n_frames):
                self._single_spin_frame(i, c_ran[i])

        else:
            # multiprocessing
            inputs = zip(range(n_frames), c_ran)
            with WorkerPool(n_jobs=n_jobs) as pool:
                pool.map(self._single_spin_frame, inputs, progress_bar=True, iterable_len=n_frames)

        if not filename:
            filename = str(f'spin_{self.c}_{self.threshold}thresh_{self.n_pts}pts_{n_frames}frames_{fps}fps')
        filename = str(filename).replace('.', ',').replace(' ', '')

        if extension == 'gif':
            self._build_gif(filename=filename,
                            n_frames=n_frames,
                            fps=fps,
                            end_buffer=0
                            )
        else:
            self._build_vid(filename=filename,
                            extension=extension,
                            n_frames=n_frames,
                            fps=fps
                            )

    def _single_spin_frame(self, i, c):
        self.c = c
        self._save_frame(frame_iter=i)

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np


class Mandelbrot:
    def __init__(self, mode='mandelbrot', c=(-0.79 + 0.15j),
                 x_ran=(-2, 1), y_ran=(-1.5, 1.5), n_pts=1000, threshold=1000):
        self.mode = mode
        self.c = c if self.mode == 'julia' else ''
        self.x_ran = x_ran
        self.y_ran = y_ran
        self.n_pts = n_pts
        self.threshold = threshold
        self.color_chart = self._determine_color_chart()
        print('Object initialised, call plot() method to plot the image or save() method to save in images directory...')

    def _determine_color_chart(self):
        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, self.n_pts)
        y_arr = np.linspace(y_min, y_max, int(self.n_pts * y_len / x_len))

        grid = np.array([x_arr + y*1j for y in reversed(y_arr)]).flatten()
        color_chart = np.zeros(grid.shape)

        # # MANDELBROT
        # self.grid = self.grid.flatten()
        # self.color_chart = self.color_chart.flatten()
        # zz = np.zeros(self.grid.shape) * 0j
        # inds = ~np.isinf(np.abs(zz))
        # for _ in range(self.threshold):
        #     zz[inds] = np.power(zz[inds], 2) + self.grid[inds]
        #     inds = ~np.isinf(np.abs(zz))
        #     self.color_chart[inds] += 1
        # self.color_chart[inds] = 0
        # self.color_chart = self.color_chart.reshape((y_arr.shape[0], x_arr.shape[0]))
        # self.color_chart = np.ma.masked_where(self.color_chart == 0, self.color_chart)


        # # JULIA
        # zz = self.grid.flatten()
        # self.color_chart = self.color_chart.flatten()
        # inds = ~np.isinf(np.abs(zz))
        # for _ in range(self.threshold):
        #     zz[inds] = np.power(zz[inds], 2) + self.c
        #     inds = ~np.isinf(np.abs(zz))
        #     self.color_chart[inds] += 1
        # self.color_chart[inds] = 0
        # self.color_chart = self.color_chart.reshape((y_arr.shape[0], x_arr.shape[0]))
        # self.color_chart = np.ma.masked_where(self.color_chart == 0, self.color_chart)

        if self.mode == 'mandelbrot':
            zz = np.zeros(grid.shape) * 0j
            c = grid

        elif self.mode == 'julia':
            zz = grid.flatten()
            c = np.full(grid.shape, self.c)

        inds = ~np.isinf(np.abs(zz))
        for _ in range(self.threshold):
            zz[inds] = np.power(zz[inds], 2) + c[inds]
            inds = ~np.isinf(np.abs(zz))
            color_chart[inds] += 1
        color_chart[inds] = 0
        color_chart = color_chart.reshape((y_arr.shape[0], x_arr.shape[0]))
        color_chart = np.ma.masked_where(color_chart == 0, color_chart)

        return color_chart




    # def _iteration(self, c):
    #     z = 0
    #     for j in range(self.threshold):
    #         z = np.power(z, 2) + c
    #         if np.isnan(np.abs(z)):
    #             return j
    #     return 0

    # def _mandelbrot(self):
    #     for i in range(self.grid.shape[0]):
    #         for j in range(self.grid.shape[1]):
    #             if self.mode == 'mandelbrot':
    #                 pt_color = self._iteration(complex(0, 0), self.grid[i, j])
    #                 self.color_chart[i, j] = pt_color
    #             elif self.mode == 'julia':
    #                 pt_color = self._iteration(self.grid[i, j], self.c)
    #                 self.color_chart[i, j] = pt_color
    #     # masking colorchart
    #     self.color_chart = np.ma.masked_where(self.color_chart == 0, self.color_chart)

    def _get_cmap(self, c_map):
        new_c_map = cmx.get_cmap(c_map).copy()
        new_c_map.set_bad(color='black')
        return new_c_map
    
    def plot(self, c_map='hsv', pallet_len=250, axis='off', fig_size=None, dpi=100):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        c_map = self._get_cmap(c_map)
        # ax.imshow(self.color_chart, origin='upper', cmap=c_map, vmin=0, vmax=self.threshold, aspect='equal')
        ax.imshow(self.color_chart%pallet_len, origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, aspect='equal')
        ax.axis(axis)

        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.linspace(x_start, x_end, 5))
        ax.set_xticklabels(np.linspace(self.x_ran[0], self.x_ran[1], 5), rotation=60)

        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.linspace(y_start, y_end, 5))
        ax.set_yticklabels(np.linspace(self.y_ran[0], self.y_ran[1], 5))
        plt.show()

    def save(self, filename=None, extension='png', c_map='hsv', pallet_len=250):
        c_map = self._get_cmap(c_map)
        # setting the default filename
        filename = str(f'{self.mode}{self.c}_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',') if not filename else str(filename)
        plt.imsave(fname='images/'+filename+f'.{extension}', arr=self.color_chart%pallet_len, origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, format=extension)

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np


class Mandelbrot:
    def __init__(self, mode='mandelbrot', c=(-0.79 + 0.15j),
                 x_ran=(-2, 1), y_ran=(-1.5, 1.5), n_pts=1000, threshold=300):
        self.mode = mode
        self.c = c if self.mode == 'julia' else ''
        self.x_ran = x_ran
        self.y_ran = y_ran
        self.n_pts = n_pts
        self.threshold = threshold

        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, self.n_pts)
        y_arr = np.linspace(y_min, y_max, int(self.n_pts * y_len / x_len))

        self.grid = np.array([x_arr + y*1j for y in reversed(y_arr)])
        self.color_chart = np.zeros(self.grid.shape)

        self.mandelbrot()
        print('Object initialised, call plot() method to plot result...')

    def iteration(self, z, c):
        for j in range(self.threshold):
            z = z**2 + c
            if np.isnan(abs(z)):
                return False, j
        return True, 0

    def mandelbrot(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.mode == 'mandelbrot':
                    flag, pt_color = self.iteration(complex(0, 0), self.grid[i, j])
                    self.color_chart[i, j] = pt_color
                elif self.mode == 'julia':
                    flag, pt_color = self.iteration(self.grid[i, j], self.c)
                    self.color_chart[i, j] = pt_color
    
    def plot(self, save=False, filename=None, extension='png', c_map='prism', axis='off', fig_size=None, dpi=100):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self.color_chart = np.ma.masked_where(self.color_chart==0, self.color_chart)
        c_map = cmx.get_cmap(c_map).copy()
        c_map.set_bad(color='black')
        ax.imshow(self.color_chart, origin='upper', cmap=c_map, vmin=0, vmax=self.threshold, aspect='equal')
        ax.axis(axis)
        plt.show()
        if save:
            filename = str(f'{self.mode}{self.c}_{self.n_pts}pts_{dpi}dpi').replace('.', ',') if not filename else str(filename)
            fig.savefig('images/'+filename+f'.{extension}', format=extension,
                        dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

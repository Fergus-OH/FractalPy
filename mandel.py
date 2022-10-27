import matplotlib.pyplot as plt
import numpy as np

class Mandelbrot:
    def __init__(self, mode='mandelbrot', c=(-0.79 + 0.15j), x_ran=[-2, 1], y_ran=[-1.5, 1.5], npts=1000, threshold=300):
        self.mode = mode
        self.c = c if self.mode=='julia' else _
        self.x_ran = x_ran
        self.y_ran = y_ran
        self.npts = npts
        self.threshold = threshold

        x_min, x_max = self.x_ran
        y_min, y_max = self.y_ran

        x_len = abs(x_max - x_min)
        y_len = abs(y_max - y_min)

        x_arr = np.linspace(x_min, x_max, self.npts)
        y_arr = np.linspace(y_min, y_max, int(self.npts*y_len/x_len))

        self.grid = np.array([x_arr + y*1j for y in reversed(y_arr)])
        self.colorchart = np.zeros(self.grid.shape)

        self.mandelbrot()
        print('Object initialised, plotting result...')
        self.plot()

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
                    flag, pt_color = self.iteration(complex(0,0), self.grid[i,j])
                    self.colorchart[i,j] = pt_color
                elif self.mode == 'julia':
                    flag, pt_color = self.iteration(self.grid[i,j], self.c)
                    self.colorchart[i,j] = pt_color
    
    def plot(self, save=False, format='png', cmap='prism', axis='off', figsize=None, dpi=100):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(self.colorchart, origin='upper', cmap=cmap, aspect='equal')
        ax.axis(axis)
        plt.show()
        if save:
            filename = str(f'{self.mode}{self.c}_{self.npts}pts_{dpi}dpi').replace('.',',')
            fig.savefig('images/'+filename+f'.{format}', format=format, dpi=fig.dpi, bbox_inches='tight', pad_inches=0)


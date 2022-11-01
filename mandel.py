import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import numpy as np

import imageio


class Mandelbrot:
    def __init__(self, julia=False, c=(-0.79 + 0.15j),
                 x_ran=None, y_ran=None, n_pts=1000, threshold=1000):
        self.julia = julia
        self.c = c if julia else None

        x_ran, y_ran = self._get_default_rans(x_ran, y_ran)
        self.x_min, self.x_max = x_ran
        self.y_min, self.y_max = y_ran

        self.n_pts = n_pts
        self.threshold = threshold
        self.color_chart = self._determine_color_chart()
        print('Object initialised, call plot() method to plot the image or save() method to save in images directory...')

    def _get_default_rans(self, x_ran, y_ran):
        if not self.julia:
            x_ran = (-2, 1) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        else:
            x_ran = (-1.5, 1.5) if not x_ran else x_ran
            y_ran = (-1.5, 1.5) if not y_ran else y_ran
        return x_ran, y_ran

    def _determine_color_chart(self):
        x_len = abs(self.x_max - self.x_min)
        y_len = abs(self.y_max - self.y_min)

        x_arr = np.linspace(self.x_min, self.x_max, self.n_pts)
        y_arr = np.linspace(self.y_min, self.y_max, int(self.n_pts * y_len / x_len))

        grid = np.array([x_arr + y*1j for y in reversed(y_arr)]).flatten()
        color_chart = np.zeros(grid.shape)

        if not self.julia:
            zz = np.zeros(grid.shape) * 0j
            c = grid

        else:
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

    def _get_cmap(self, c_map):
        new_c_map = cmx.get_cmap(c_map).copy()
        new_c_map.set_bad(color='black')
        return new_c_map
    
    def plot(self, c_map='hsv', pallet_len=250, axis='off', fig_size=None, dpi=100):
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        c_map = self._get_cmap(c_map)
        ax.imshow(self.color_chart%pallet_len, origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, aspect='equal')
        ax.axis(axis)

        x_start, x_end = ax.get_xlim()
        ax.set_xticks(np.linspace(x_start, x_end, 5))
        ax.set_xticklabels(np.linspace(self.x_min, self.x_max, 5), rotation=60)

        y_start, y_end = ax.get_ylim()
        ax.set_yticks(np.linspace(y_start, y_end, 5))
        ax.set_yticklabels(np.linspace(self.y_min, self.y_max, 5))
        plt.show()

    def save(self, filename=None, extension='png', c_map='hsv', pallet_len=250):
        c_map = self._get_cmap(c_map)
        # setting the default filename
        if not self.julia:
            filename = str(f'Mandelbrot_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',') if not filename else str(filename)
        else:
            filename = str(f'Julia_{self.c}_{self.n_pts}pts_{self.threshold}threshold').replace('.', ',') if not filename else str(filename)
        plt.imsave(fname='images/'+filename+f'.{extension}', arr=self.color_chart%pallet_len, origin='upper', cmap=c_map, vmin=0, vmax=pallet_len, format=extension)

    def zoom(self, filename='zoomVid', frame_subdir='zoom', target=(6e+4,-1.186592e+0,-1.901211e-1), n_frames=120, start_frame=0):
        (m, x, y) = target

        x_target_len = abs(self.x_max - self.x_min)/m
        x_target = (x - x_target_len/2, x + x_target_len/2)

        y_target_len = abs(self.y_max - self.y_min)/m
        y_target = (y - y_target_len/2, y + y_target_len/2)

        geom = np.flip(1-(np.geomspace(1,m,n_frames)-1)/(m-1))
        x_ranges = [(x0,x1) for (x0,x1) in zip(self.x_min + geom*(x_target[0]-self.x_min), self.x_max + geom*(x_target[1]-self.x_max))]
        y_ranges = [(y0,y1) for (y0,y1) in zip(self.y_min + geom*(y_target[0]-self.y_min), self.y_max + geom*(y_target[1]-self.y_max))]

        for i in range(start_frame, n_frames):
            self.x_ran = x_ranges[i]
            self.y_ran = y_ranges[i]
            self.color_chart = self._determine_color_chart()
            self.save(f'frames/{frame_subdir}/image{i}')
            print(f'iteration {i} completed...')

        self.build_gif(filename=filename, frame_dir=frame_subdir, n_frames=n_frames)

    def spin(self, filename='spinVid', frame_subdir='spin', n_frames=60, start_frame=0):
        a_ran = np.linspace(0, 2*np.pi, n_frames)
        modulus = np.abs(self.c)
        for i in range(start_frame, n_frames):
            a = a_ran[i]
            self.c = modulus*np.exp(1j*a)
            self.color_chart = self._determine_color_chart()
            self.save(f'frames/{frame_subdir}/image{i}')
            print(f'iteration {i} completed...')

        self.build_gif(filename=filename, frame_dir=frame_subdir, n_frames=n_frames)

    def build_gif(self, filename, frame_dir, n_frames):
        with imageio.get_writer(f'videos/{filename}.gif', mode='I') as writer:
            for filename in [f'images/frames/{frame_dir}/image{i}.png' for i in range(n_frames)]:
                image = imageio.imread(filename)
                writer.append_data(image)
        print('completed. Video saved at...')

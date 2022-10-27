import matplotlib.pyplot as plt
import numpy as np

def mandelbrotIteration(c, threshold):
    z = complex(0,0)
    for j in range(threshold):
        z = z**2 + c
        if np.isnan(abs(z)):
            return False, j
    return True, 0

def mandelbrot(threshold=300, x_ran=[-2, 1], y_ran=[-1.5, 1.5], npts=1000):
    x_min, x_max = x_ran
    y_min, y_max = y_ran

    x_len = abs(x_max - x_min)
    y_len = abs(y_max - y_min)

    x_arr = np.linspace(x_min, x_max, npts)
    y_arr = np.linspace(y_min, y_max, npts)


    grid = np.array([x_arr + y*1j for y in y_arr])
    colorchart = np.zeros(grid.shape)


    mand = np.array([])
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            flag, pt_color = mandelbrotIteration(grid[i,j], threshold)
            colorchart[i,j] = pt_color
            if flag:
                mand = np.append(mand, grid[i,j])


    ### plotting
    fig, ax = plt.subplots(figsize=(x_len+1, y_len+1), dpi=300)
    plt.imshow(colorchart, origin='lower')
    plt.show()

mandelbrot()
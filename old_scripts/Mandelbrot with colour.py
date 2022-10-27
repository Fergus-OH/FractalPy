import matplotlib.pyplot as plt
import numpy as np

def mandelbrotIteration(c, threshold):
    z = complex(0,0)
    for j in range(threshold):
        z = z**2 + c
        if np.isnan(abs(z)):
            return False, j
    return True, np.inf

def mandelbrot(threshold=300, x_ran=[-2, 1], y_ran=[-1.5, 1.5], res=100):
    x_min, x_max = x_ran
    y_min, y_max = y_ran

    x_len = abs(x_max - x_min)
    y_len = abs(y_max - y_min)

    #some adjusting needs to be done here
    unit = 1/res

    # x_arr = np.arange(x_min, x_max, unit)
    # y_arr = np.arange(y_min, y_max, unit)

    x_arr = np.linspace(x_min, x_max, 1000)
    y_arr = np.linspace(y_min, y_max, 1000)

    grid = np.array([x_arr + y*1j for y in y_arr]).flatten()
    colorchart = np.zeros(len(grid))

    mand = np.array([])
    for i in range(len(grid)):
        flag, pt_color = mandelbrotIteration(grid[i], threshold)
        colorchart[i] = pt_color
        if flag:
            mand = np.append(mand, grid[i])

    X1 = np.array([x.real for x in grid])
    Y1 = np.array([x.imag for x in grid])


    ### plotting
    fig, ax = plt.subplots(figsize=(x_len+1, y_len+1), dpi=100)
    # px = 1/plt.rcParams['figure.dpi'] 
    # num_points = ((.5*unit)**2)/px
    num_points = .5*unit*fig.dpi
    marker_size = num_points**2

    ax.scatter(X1, Y1, marker='.', s=marker_size, c=colorchart, cmap='Spectral')
    ax.axis([x_min, x_max, y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    # fig.tight_layout()
    plt.show()

    ax.axis('off')
    fig.savefig('boo', bbox_inches='tight', dpi=fig.dpi)


    # X = np.array([x.real for x in mand])
    # Y = np.array([x.imag for x in mand])

    # plt.scatter(X,Y, s=0.5, color='black')
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()

# mandelbrot(x_ran=[-0.22, -0.219], y_ran=[-0.70, -0.699])
mandelbrot()
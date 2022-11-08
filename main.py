from mandel import Mandelbrot


def __main__():
    x_init = (-3, 3)
    y_init = (-1.6875, 1.6875)

    # target = (62407000, -0.743643930055, -0.131825895901)
    target = (10004407000, -0.7436439059192348, -0.131825896951)

    (m, x, y) = target
    x_l = abs(x_init[1] - x_init[0])/m
    x_target = (x - x_l/2, x + x_l/2)
    y_l = abs(y_init[1] - y_init[0])/m
    y_target = (y - y_l/2, y + y_l/2)
    # plot target
    # Mandelbrot(x_ran=x_target, y_ran=y_target, threshold=5000).plot(dpi=100, axis='on')
    # Mandelbrot(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=360, pallet_len=25).zoom(n_frames=100, target=target)
    # Mandelbrot(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=3456).zoom(target=target, n_frames=1000)
    Mandelbrot(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=1920).zoom(target=target, n_frames=5000)
    # Mandelbrot(julia=True, threshold=1000, n_pts=3456, pallet_len=25).spin(n_frames=500)

if __name__ == '__main__':
    __main__()

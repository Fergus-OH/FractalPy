from fractal import Fractal


def main():
    x_init = (-3, 3)
    y_init = (-1.6875, 1.6875)

    target = (10004407000, -0.7436439059192348, -0.131825896951)



    # plot target
    # Mandelbrot(x_ran=x_target, y_ran=y_target, threshold=5000).plot(dpi=100, axis='on')
    # Fractal(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=360, pallet_len=25).zoom(n_frames=100, target=target)
    # Fractal(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=2160).zoom(target=target, n_frames=1000)
    Fractal(x_ran=x_init, y_ran=y_init, threshold=5000, n_pts=480).zoom(target=target, n_frames=100, fps=30)
    # Fractal(julia=True, threshold=1000, n_pts=3456, pallet_len=25).spin(n_frames=500)


if __name__ == '__main__':
    main()

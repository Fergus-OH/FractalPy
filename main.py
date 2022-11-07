from mandel import Mandelbrot


def __main__():
    target = (62407000, -0.743643900055, -0.131825890901)
    Mandelbrot(x_ran=(-3,1), y_ran=(-1,1.5), threshold=3000).zoom(n_frames=100, target=target)

if __name__ == '__main__':
    __main__()

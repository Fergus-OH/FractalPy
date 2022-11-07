from mandel import Mandelbrot

target = (62407000, -0.743643900055, -0.131825890901)
Mandelbrot(x_ran=(-3,1), y_ran=(-1,1.5)).zoom(n_frames=100, target=target)


# import unittest
# from unittest.mock import patch
# from fractalpy import Mandelbrot, Julia
#
#
# class TestMandelbrot(unittest.TestCase):
#
#     @patch('matplotlib.pyplot.show')
#     def test_plot(self, mock_show):
#         # Create a new FractalBase instance with default values
#         fractal = Mandelbrot()
#
#         # Call the plot method
#         fractal.plot()
#
#         # Verify that the plot method calls the correct functions in the imageio and matplotlib modules
#         mock_show.assert_called_once()
#
#     @patch('matplotlib.pyplot.imsave')
#     def test_save(self, mock_imsave):
#         fractal = Mandelbrot(n_pts=20, threshold=50)
#         fractal.save()
#         mock_imsave.assert_called_once()
#
#     @patch('imageio.get_writer')
#     @patch('imageio.v3.imread')
#     @patch('matplotlib.pyplot.imsave')
#     def test_zoom(self, mock_imsave, mock_imread, mock_get_writer):
#         n_frames = 5
#         fractal = Mandelbrot(n_pts=20, threshold=50)
#         fractal.zoom(n_frames=n_frames, n_jobs=1)
#
#         assert mock_imsave.call_count == n_frames
#         assert mock_imread.call_count == n_frames
#         mock_get_writer.assert_called_once()
#
#     @patch('os.system')
#     @patch('mpire.WorkerPool.map')
#     def test_ffmpeg(self, mock_map, mock_system):
#         n_frames = 2
#         fractal = Mandelbrot(n_pts=20, threshold=50)
#         fractal.zoom(n_frames=n_frames, extension='mp4')
#         mock_map.assert_called_once()
#         mock_system.assert_called_once()
#
#
# class TestJulia(unittest.TestCase):
#     @patch('matplotlib.pyplot.show')
#     def test_plot(self, mock_show):
#         # Create a new FractalBase instance with default values
#         fractal = Julia(n_pts=20, threshold=50)
#
#         # Call the plot method
#         fractal.plot()
#
#         # Verify that the plot method calls the correct functions in the imageio and matplotlib modules
#         mock_show.assert_called_once()
#
#     @patch('matplotlib.pyplot.imsave')
#     def test_save(self, mock_imsave):
#         fractal = Julia(n_pts=20, threshold=50)
#         fractal.save()
#         mock_imsave.assert_called_once()
#
#     @patch('imageio.get_writer')
#     @patch('imageio.v3.imread')
#     @patch('matplotlib.pyplot.imsave')
#     def test_zoom(self, mock_imsave, mock_imread, mock_get_writer):
#         n_frames = 2
#         fractal = Julia(n_pts=20, threshold=50)
#         fractal.zoom(n_frames=n_frames, n_jobs=1)
#
#         assert mock_imsave.call_count == n_frames
#         assert mock_imread.call_count == n_frames
#         mock_get_writer.assert_called_once()
#
#     @patch('imageio.get_writer')
#     @patch('imageio.v3.imread')
#     @patch('matplotlib.pyplot.imsave')
#     def test_spin(self, mock_imsave, mock_imread, mock_get_writer):
#         n_frames = 2
#         fractal = Julia(n_pts=20, threshold=50)
#         fractal.spin(n_frames=n_frames, n_jobs=1)
#
#         assert mock_imsave.call_count == n_frames
#         assert mock_imread.call_count == n_frames
#         mock_get_writer.assert_called_once()
#
#     @patch('os.system')
#     @patch('mpire.WorkerPool.map')
#     def test_ffmpeg(self, mock_map, mock_system):
#         n_frames = 2
#         fractal = Julia(n_pts=20, threshold=50)
#         fractal.spin(n_frames=n_frames, extension='mp4')
#
#         mock_map.assert_called_once()
#         mock_system.assert_called_once()
#
#
# # RUN with: NUMBA_DISABLE_JIT=1 pytest --cov-report term-missing --cov

import unittest
from unittest.mock import patch

import fractalpy
from click.testing import CliRunner
from fractalpy.cli.cli_main import cli_main


class TestCli(unittest.TestCase):

    @patch('fractalpy.cli.cli_main.cli_main')
    def test_init(self, mock):
        fractalpy.cli.cli_main.main()
        mock.assert_called_once()


class TestMandelbrot(unittest.TestCase):
    runner = CliRunner()
    npts = 20
    n_frames = 5
    threshold = 50

    @patch('matplotlib.pyplot.show')
    def test_plot(self, mock_show):
        self.runner.invoke(cli_main, ['mandelbrot',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'plot'
                                      ])
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.imsave')
    def test_save(self, mock_imsave):
        self.runner.invoke(cli_main, ['mandelbrot',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'save'
                                      ])
        mock_imsave.assert_called_once()

    @patch('imageio.get_writer')
    @patch('imageio.v3.imread')
    @patch('matplotlib.pyplot.imsave')
    def test_zoom(self, mock_imsave, mock_imread, mock_get_writer):
        self.runner.invoke(cli_main, ['mandelbrot',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'zoom',
                                      '--n_frames', f'{self.n_frames}',
                                      '--n_jobs', '1',
                                      ])
        assert mock_imsave.call_count == self.n_frames
        assert mock_imread.call_count == self.n_frames
        mock_get_writer.assert_called_once()

    @patch('os.system')
    @patch('mpire.WorkerPool.map')
    def test_ffmpeg_pool(self, mock_map, mock_system):
        self.runner.invoke(cli_main, ['mandelbrot',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'zoom',
                                      '--n_frames', f'{self.n_frames}',
                                      '--extension', 'mp4'
                                      ])

        mock_map.assert_called_once()
        mock_system.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_zoom_preview(self, mock_show):
        self.runner.invoke(cli_main, ['mandelbrot',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'zoom',
                                      '--preview'
                                      ])
        mock_show.assert_called_once()


class TestJulia(unittest.TestCase):
    runner = CliRunner()
    npts = 20
    n_frames = 5
    threshold = 50

    @patch('matplotlib.pyplot.show')
    def test_plot(self, mock_show):
        self.runner.invoke(cli_main, ['julia',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'plot'
                                      ])
        mock_show.assert_called_once()

    #
    @patch('matplotlib.pyplot.imsave')
    def test_save(self, mock_imsave):
        self.runner.invoke(cli_main, ['julia',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'save'
                                      ])
        mock_imsave.assert_called_once()

    @patch('imageio.get_writer')
    @patch('imageio.v3.imread')
    @patch('matplotlib.pyplot.imsave')
    def test_zoom(self, mock_imsave, mock_imread, mock_get_writer):
        self.runner.invoke(cli_main, ['julia',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'zoom',
                                      '--n_frames', f'{self.n_frames}',
                                      '--n_jobs', '1'
                                      ])
        assert mock_imsave.call_count == self.n_frames
        assert mock_imread.call_count == self.n_frames
        mock_get_writer.assert_called_once()

    @patch('imageio.get_writer')
    @patch('imageio.v3.imread')
    @patch('matplotlib.pyplot.imsave')
    def test_spin(self, mock_imsave, mock_imread, mock_get_writer):
        self.runner.invoke(cli_main, ['julia',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'spin',
                                      '--n_frames', f'{self.n_frames}',
                                      '--n_jobs', '1'
                                      ])
        assert mock_imsave.call_count == self.n_frames
        assert mock_imread.call_count == self.n_frames
        mock_get_writer.assert_called_once()

    @patch('os.system')
    @patch('mpire.WorkerPool.map')
    def test_spin_ffmpeg_pool(self, mock_map, mock_system):
        self.runner.invoke(cli_main, ['julia',
                                      '--npts', f'{self.npts}',
                                      '--threshold', f'{self.threshold}',
                                      'spin',
                                      '--n_frames', f'{self.n_frames}',
                                      '--extension', 'mp4'
                                      ])
        mock_map.assert_called_once()
        mock_system.assert_called_once()

# RUN with: NUMBA_DISABLE_JIT=1 pytest --cov-report term-missing --cov

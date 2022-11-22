"""Top-level script for this package."""

from importlib.metadata import version

from .fractals.fractals import Julia, Mandelbrot

__version__ = version(__package__)

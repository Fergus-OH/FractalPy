# FractalPy
![pypi](https://img.shields.io/pypi/v/FractalPy)
![tag](https://img.shields.io/github/v/tag/Fergus-OH/FractalPy)
![python_version](https://img.shields.io/pypi/pyversions/FractalPy)
![licence](https://img.shields.io/github/license/Fergus-OH/FractalPy)
![checks](https://img.shields.io/github/checks-status/Fergus-OH/FractalPy/main)
[![codecov](https://codecov.io/gh/Fergus-OH/FractalPy/branch/main/graph/badge.svg?token=XWYUNL7XIE)](https://codecov.io/gh/Fergus-OH/FractalPy)

<p align="center">
  <img src= "https://raw.githubusercontent.com/Fergus-OH/mandelbrot-julia-sets/numba/assets/Mandelbrot_4320pts_1000threshold.png" width="800">
</p>

Consider the recurrence relation $z_{n+1} = z_n^2 + c$ where $c$ is a complex number.
The Mandelbrot set is a fractal, defined by the set of complex numbers $c$ for which this recurrence relation, with initial value $z_0 = 0$, does not diverge.
Another interesting type of set, which are related to the Mandelbrot set, are Julia sets and are defined for a specific complex number $c$.
To keep things brief, we will just establish the definition of a filled-in Julia set and do so in the following way:
The filled-in Julia set of a complex number $c$ is the set of initial values $z_0$ for which the previously mentioned recurrence relation does not diverge.
Not every filled-in Julia set is a fractal, but for almost all complex numbers $c$, they are.
This project contains an implementation to generate images and videos relating to the Mandelbrot set and Julia sets.

[//]: # (<img src="https://raw.githubusercontent.com/Fergus-OH/FractalPy/numba/assets/zoom_&#40;-1,186592,-0,1901211&#41;)

[//]: # (_1000thresh_360pts_60frames_15fps.gif" width="400">)

<p align="center">
  <img src="https://raw.githubusercontent.com/Fergus-OH/FractalPy/numba/assets/zoom_(-1,186592,-0,1901211)_1000thresh_360pts_60frames_15fps-min.gif" width="400">
  <img src="https://raw.githubusercontent.com/Fergus-OH/FractalPy/numba/assets/spin_(-0,79+0,15j)_1000thresh_360pts_110frames_30fps.gif" width="400">
</p>




[//]: # (<img src="https://raw.githubusercontent.com/Fergus-OH/mandelbrot-julia-sets/numba/assets/zoom_&#40;10004407000,-0,7436439059192348,-0,131825896951&#41;_5000thresh_480pts_300frames_30fps.gif" width="500">)
[//]: # (<img src="https://raw.githubusercontent.com/Fergus-OH/mandelbrot-julia-sets/numba/assets/julia_spin2.gif" width="500">)
  


## Installation
Before installing the `FractalPy` package, it is recommended to create and activate a virtual environment with `python 3.10`.
This can be done with conda by running the following commands in a terminal
```
$ conda create --name fractal python==3.10
```

```
$ conda activate fractal
```
Now the package and it's dependencies can be installed in the virtual environment, `fractal`, using pip
```
$ pip install fractalpy
```

For an editable installation from the source, first clone the repository and install with the following
```
$ conda activate fractal
$ pip install -e .
```

## Usage
There are two ways of using `FractalPy`.
The package can be imported to a python script with

```python
import fractalpy as frac

# Plot the Mandelbrot set
frac.Mandelbrot().plot()

# Plot the Julia set
frac.Julia().plot()
```

The package also offers a command line interface that can be immediately accessed in the terminal with
```
fractalpy --help
```

For example, we can create a gif of zooming into the mandelbrot set with the following command:
```
fractalpy mandelbrot zoom
```

If FFmpeg is installed and accessible via the $PATH environment variable, then `FractalPy` can also generate videos, for example
```
fractalpy mandelbrot zoom --extension mp4
```

`FractalPy` makes use of multiprocessing to generate multiple frames simultaneously and also performs the computationally expensive calculations in parallel with `jit`, making it an extremely fast.
<!-- ```
Fractal().
```


A notebook with demos can be found [here](https://nbviewer.org/github/Fergus-OH/mandelbrot-julia-sets/blob/numba/demos.ipynb)

<a href="https://nbviewer.org/github/Fergus-OH/mandelbrot-julia-sets/blob/numba/demos.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Render nbviewer" /></a> -->
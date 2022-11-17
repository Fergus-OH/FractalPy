"""A setuptools based setup module."""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="fractalPy",
    version="1.0.0",
    description="A package to generate videos and images of fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fergus-OH/mandelbrot-julia-sets",
    author="Fergus O'Hanlon",
    author_email="fergusohanlon@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="sample, setuptools, development",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=[
        'Click',
        'imageio',
        'matplotlib',
        'mpire',
        'numba',
        'numpy'
    ],
    entry_points={
        'console_scripts': ['fractalPy = fractalPy.__main__'],
    },
)

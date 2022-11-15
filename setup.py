from setuptools import setup, find_packages

setup(
    name='fractal',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'imageio',
        'matplotlib',
        'mpire',
        'numba',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'yourscript = fractal.cli:cli',
        ],
    },
    license='',
    author='fergusohanlon',
    author_email='',
    description=''
)

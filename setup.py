"""
Installs the histogram tools module.
"""

import numpy
from os.path import join, abspath, dirname
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='histeq',
    version='0.1.0',
    description='Image Histogram Equalization Tools',
    long_description=open(join(abspath(dirname(__file__)), 'README.md'), encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    #url='...',
    author='Jeffrey Bush',
    author_email='bushj@moravian.edu',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics :: Editors :: Raster-Based',
        'Topic :: Scientific/Engineering :: Image Recognition',
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='image-processing histogram-equalization contrast-enhancement',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.6, <4',
    install_requires=['imageio', 'numpy>=1.15', 'scipy>=1', 'cython>=0.28'],
    extras_require={'opt': ['pyfftw', 'PyWavelets', 'cupy']},
    ext_modules=cythonize('hist/exact/*.pyx'),
    include_dirs=[numpy.get_include()],
    package_data={'hist': ['*.hpp','*.pxd','*.pyx']},
    #entry_points={
    #    'console_scripts': [
    #        'metric_battery=hist:metric_battery:main',
    #    ],
    #},
)

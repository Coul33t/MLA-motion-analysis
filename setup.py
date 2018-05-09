#!/usr/bin/env python

from distutils.core import setup

setup(name='ML MLA',
      version='0.5',
      description='Machine Learning for the Motion Learning Analytics platform',
      author='Quentin Couland',
      author_email='quentin.couland@gmail.com',
      url='https://github.com/Coul33t/ml_mla',
      packages=['mla'],
      install_requires=['numpy', 'matplotlib', 'scipy', 'sklearn', 'fastdtw'],
     )
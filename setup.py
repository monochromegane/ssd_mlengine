from setuptools import find_packages
from setuptools import setup

setup(
    name='trainer',
    packages=['trainer'],
    install_requires=['keras==1.2.2', 'numpy', 'pandas', 'pillow', 'scipy', 'h5py']
)

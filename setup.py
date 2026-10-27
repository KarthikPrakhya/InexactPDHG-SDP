from skbuild import setup
from setuptools import find_packages

setup(
    name='InexactPDHG-SDP',
    version='1.0.0',
    description='Solving Convex Composite Optimization Problems with Inexact PDHG with focus on Trace-bounded SDPs',
    author='Karthik Prakhya',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'pybind11>=2.6.0',
        'scikit-build',
    ],
)

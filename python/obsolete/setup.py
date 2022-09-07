from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.install import install
from os.path import dirname, realpath, join, isfile
from collections import namedtuple
import setuptools
import os
import sys

root_dir = realpath(dirname(__file__))
sys.tracebacklimit = 0
a = dict()
_version_path = "src/_version.py"
info_path = "src/info.txt"
exec(open(_version_path).read(), globals(), a)

if '__version__' in a:
    __version__ = a['__version__']
if '__cuda_version__' in a:
    __cuda_version__ = a['__cuda_version__']
if '__cudnn_version__' in a:
    __cudnn_version__ = a['__cudnn_version__']
if '__ompi_version__' in a:
    __ompi_version__ = a['__ompi_version__']
if '__author__' in a:
    __author__ = a['__author__']
if '__email__' in a:
    __email__ = a['__email__']

cuda_version = ''.join(__cuda_version__.split('.'))
ompi_version = '_'.join(__ompi_version__.split('.'))

whl_suffix = ''
if 'WHEEL_SUFFIX' in os.environ:
    whl_suffix += os.environ['WHEEL_SUFFIX']

with open(info_path) as f:
    install_cuda_version = f.readlines()


class ObsoleteException(Exception):
    def __init__(self, msg):
        self.msg = msg


class PreInstallCommand(install):
    """Pre-installation for installation mode."""

    def run(self):
        raise ObsoleteException(('This wheel is obsoleted. Please install nnabla_ext_cuda{} as its successor with NCCL2 and OpenMPI support.').format(
            install_cuda_version[0].strip('\n')))


ExtConfig = namedtuple('ExtConfig',
                       ['package_dir', 'packages', 'package_data',
                        'ext_modules', 'ext_opts'])


def get_setup_config(root_dir):
    packages = []
    package_dir = {}
    package_data = {}
    ext_modules = []

    global cuda_version
    global ompi_version

    pkg_name = 'nnabla_ext_cuda{}_nccl2_mpi{}'.format(
        cuda_version, ompi_version)

    pkg_info = dict(
        name=pkg_name,
        description='A CUDA({}) and cuDNN({}) extension of NNabla'.format(
            __cuda_version__, __cudnn_version__),
        version=__version__,
        author=__author__,
        author_email=__email__,
        url="https://github.com/sony/nnabla-ext-cuda",
        license='Apache License 2.0',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS :: MacOS X'
        ],

        platforms=['CUDA {}'.format(__cuda_version__),
                   'cuDNN {}'.format(__cudnn_version__)],
        keywords="deep learning artificial intelligence machine learning neural network cuda",
    )
    return pkg_info, ExtConfig(package_dir, packages, package_data, ext_modules, {})


pkg_info, cfg = get_setup_config(root_dir)

setuptools.setup(
    project_urls={
        "Bug Tracker": "https://github.com/nnabla/nnabla/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    cmdclass={
        'install': PreInstallCommand,
    },
    **pkg_info
)

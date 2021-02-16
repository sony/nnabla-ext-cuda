# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from setuptools import setup, find_packages
from distutils.extension import Extension
import os
from os.path import dirname, realpath, join, isfile, splitext
from collections import namedtuple
import copy
import shutil
import subprocess
import sys

root_dir = realpath(dirname(__file__))
a = dict()

__version__ = None
__email__ = None
exec(open(os.path.join(root_dir, 'src', 'nnabla_ext',
                       'cuda', '_version.py')).read(), globals(), a)
if '__version__' in a:
    __version__ = a['__version__']
if '__cuda_version__' in a:
    __cuda_version__ = a['__cuda_version__']
if '__cudnn_version__' in a:
    __cudnn_version__ = a['__cudnn_version__']
if '__author__' in a:
    __author__ = a['__author__']
if '__email__' in a:
    __email__ = a['__email__']
assert(__version__ is not None)
assert(__author__ is not None)
assert(__email__ is not None)

setup_requires = [
    'numpy',
    'Cython',  # Requires python-dev.
]

whl_suffix = ''
if 'WHEEL_SUFFIX' in os.environ:
    whl_suffix += os.environ['WHEEL_SUFFIX']

install_requires = [
    'pynvml',
    'setuptools',
    'nnabla{}=={}'.format(whl_suffix, __version__),
]

LibInfo = namedtuple('LibInfo', ['file_name', 'path', 'name'])
ExtConfig = namedtuple('ExtConfig',
                       ['package_dir', 'packages', 'package_data',
                        'ext_modules', 'ext_opts'])


def get_libinfo():
    from six.moves.configparser import ConfigParser

    # Parse setup.cfg
    path_cfg = join(dirname(__file__), "setup.cfg")
    if not isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)

    # Read cpu lib info
    cpu_lib = LibInfo(None,
                      cfgp.get("cmake", "cpu_target_file"),
                      cfgp.get("cmake", "cpu_target_name"))
    print("CPU Library name:", cpu_lib.name)
    print("CPU Library file:", cpu_lib.path)

    # Read cuda lib info
    cuda_lib = LibInfo(cfgp.get("cmake", "cuda_target_file_name"),
                       cfgp.get("cmake", "cuda_target_file"),
                       cfgp.get("cmake", "cuda_target_name"))
    print("CUDA Library name:", cuda_lib.name)
    print("CUDA Library file name:", cuda_lib.file_name)
    print("CUDA Library file:", cuda_lib.path)

    if 'INCLUDE_CUDA_CUDNN_LIB_IN_WHL' in os.environ and os.environ['INCLUDE_CUDA_CUDNN_LIB_IN_WHL'] == 'True':
        print("CUDA/cuDNN libraries will include into wheel package.")
        libs = [cfgp.get("cmake", "cuda_toolkit_root_dir"),
                os.path.dirname(cfgp.get("cmake", "cudnn_include_dir"))]
    else:
        libs = None

    return cpu_lib, cuda_lib, libs


def get_cpu_extopts(lib):
    import numpy as np
    include_dir = realpath(join(dirname(__file__), '../include'))
    ext_opts = dict(
        include_dirs=[include_dir, np.get_include()],
        libraries=[lib.name],
        library_dirs=[dirname(lib.path)],
        language="c++",
        # The below definition breaks build. Use -Wcpp instead.
        # define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    if sys.platform != 'win32':
        ext_opts.update(dict(
            extra_compile_args=[
                '-std=c++11', '-Wno-sign-compare', '-Wno-unused-function', '-Wno-cpp'],
            runtime_library_dirs=['$ORIGIN/'],
        ))
    else:
        ext_opts.update(dict(extra_compile_args=['/W0']))
    return ext_opts


def get_cpu_include_dir():
    from six.moves.configparser import ConfigParser

    # Parse setup.cfg
    path_cfg = join(dirname(__file__), "setup.cfg")
    if not isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)

    # Read cpu lib info
    cpu_include_dir = cfgp.get("cmake", "cpu_include_dir")
    print("CPU Include directory:", cpu_include_dir)

    return cpu_include_dir


def get_cpu_cython_path():
    from six.moves.configparser import ConfigParser

    # Parse setup.cfg
    path_cfg = join(dirname(__file__), "setup.cfg")
    if not isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)

    # Read cuda header info
    cpu_cython_path = cfgp.get("cmake", "cpu_cython_path")
    print("CPU Cython path:", cpu_cython_path)

    return cpu_cython_path


def get_cuda_include_dir():
    from six.moves.configparser import ConfigParser

    # Parse setup.cfg
    path_cfg = join(dirname(__file__), "setup.cfg")
    if not isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)

    # Read cpu lib info
    cuda_include_dir = os.path.join(
        cfgp.get("cmake", "cuda_toolkit_root_dir"), "include")
    print("CUDA Include directory:", cuda_include_dir)

    return cuda_include_dir


def cuda_config(root_dir, cuda_lib, ext_opts, lib_dirs):
    # With CUDA
    src_dir = join(root_dir, 'src')
    path_cuda_pkg = join(src_dir, 'nnabla_ext', 'cuda')
    cuda_pkg = "nnabla_ext.cuda"
    package_dir = {cuda_pkg: path_cuda_pkg}
    packages = [
        cuda_pkg,
        cuda_pkg + '.experimental',
    ]

    cuda_lib_out = join(path_cuda_pkg, cuda_lib.file_name)
    shutil.copyfile(cuda_lib.path, cuda_lib_out)
    package_data = {cuda_pkg: [cuda_lib.file_name]}

    nnabla_ext_cuda_root = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..'))
    os.makedirs(os.path.join(path_cuda_pkg, 'doc/third_party'), exist_ok=True)
    for fn in ['LICENSE',
               'NOTICE.md',
               os.path.join('third_party', 'LICENSES.md')]:
        shutil.copyfile(os.path.join(nnabla_ext_cuda_root, fn),
                        os.path.join(path_cuda_pkg, 'doc', fn))
        package_data["nnabla_ext.cuda"].append(os.path.join('doc', fn))

    if lib_dirs is not None:
        if sys.platform.startswith('linux'):
            out = subprocess.check_output(['ldd', cuda_lib_out])
            for l in out.splitlines():
                ls = l.strip().decode('ascii').split()
                if len(ls) >= 3:
                    libname = ls[0]
                    if libname == "libcuda.so.1":
                        continue
                    libfile = ls[2]

                    # Copy libraries into WHL file.
                    # libcu*   : CUDA/cuDNN
                    # libnccl  : NCCL2
                    if libname.startswith('libcu') or \
                            libname.startswith('libnccl'):
                        print('Copying {}'.format(libname))
                        path_out = join(path_cuda_pkg, libname)
                        shutil.copyfile(libfile, path_out)
                        package_data[cuda_pkg].append(libname)

        elif sys.platform == 'win32':
            libdir = dirname(cuda_lib.path)
            libname, _ = splitext(cuda_lib.file_name)
            cuda_ext_lib_file_name = libname + '.lib'
            cuda_ext_lib_path = join(libdir, cuda_ext_lib_file_name)
            cuda_ext_lib_out = join(path_cuda_pkg, cuda_ext_lib_file_name)
            shutil.copyfile(cuda_ext_lib_path, cuda_ext_lib_out)
            package_data[cuda_pkg].append(cuda_ext_lib_file_name)

            def search_dependencies(lib, libs=[]):
                print('Searching libs in {}'.format(lib))
                out = subprocess.check_output(['dumpbin', '/DEPENDENTS', lib])
                for l in out.splitlines():
                    l = l.strip().decode('ascii')
                    if l[:2] == 'cu' and l[-4:] == '.dll':
                        copied = False
                        for d in lib_dirs:
                            for currentdir, dirs, files in os.walk(d):
                                if l in files and not copied and l not in libs:
                                    path_in = join(currentdir, l)
                                    if l.lower() == 'cudnn64_8.dll':
                                        for cudnn_lib in files:
                                            if cudnn_lib.lower != 'cudnn64_8.lib':
                                                print(
                                                    'Copying {}'.format(cudnn_lib))
                                                shutil.copyfile(join(currentdir, cudnn_lib), join(
                                                    path_cuda_pkg, cudnn_lib))
                                                libs.append(cudnn_lib)
                                    else:
                                        libs = search_dependencies(
                                            path_in, libs)
                                    path_out = join(path_cuda_pkg, l)
                                    print('Copying {}'.format(l))
                                    shutil.copyfile(path_in, path_out)
                                    libs.append(l)
                                    copied = True
                        if not copied:
                            print('Shared library {} is not found.'.format(l))
                            sys.exit(-1)
                return libs

            for d in search_dependencies(cuda_lib_out):
                print('LIB: {}'.format(d))
                package_data[cuda_pkg].append(d)

    cuda_ext_opts = copy.deepcopy(ext_opts)
    cuda_ext_opts['libraries'] += [cuda_lib.name]
    cuda_ext_opts['library_dirs'] += [dirname(cuda_lib.path)]
    cuda_ext_opts['include_dirs'] += [get_cpu_include_dir(),
                                      get_cuda_include_dir()]
    ext_modules = [
        Extension(cuda_pkg + '.init',
                  [join(path_cuda_pkg, 'init.pyx')],
                  **cuda_ext_opts),
        Extension(cuda_pkg + '.nvtx',
                  [join(path_cuda_pkg, 'nvtx.pyx')],
                  **cuda_ext_opts),
    ]

    return ExtConfig(package_dir, packages, package_data,
                     ext_modules, cuda_ext_opts)


def cudnn_config(root_dir, cuda_lib, cuda_ext_opts):
    src_dir = join(root_dir, 'src')
    path_cudnn_pkg = join(src_dir, 'nnabla_ext', 'cudnn')
    cudnn_pkg = 'nnabla_ext.cudnn'
    package_dir = {cudnn_pkg: path_cudnn_pkg}
    packages = [cudnn_pkg]
    ext_modules = [
        Extension(cudnn_pkg + '.init',
                  [join(path_cudnn_pkg, 'init.pyx')],
                  **cuda_ext_opts),
    ]

    return ExtConfig(package_dir, packages, {},
                     ext_modules, cuda_ext_opts)


def utils_config(root_dir, cuda_ext_opts):
    src_dir = join(root_dir, 'src')
    path_utils_pkg = join(src_dir, 'nnabla_ext', 'cuda', 'utils')
    utils_pkg = "nnabla_ext.cuda.utils"
    package_dir = {utils_pkg: path_utils_pkg}

    packages = [utils_pkg, ] + [utils_pkg + "." +
                                x for x in find_packages(where=path_utils_pkg)]
    ext_modules = []

    return ExtConfig(package_dir, packages, {},
                     ext_modules, cuda_ext_opts)


def get_setup_config(root_dir):
    cpu_lib, cuda_lib, libs = get_libinfo()

    packages = ['nnabla_ext']
    package_dir = {'nnabla_ext': join(root_dir, 'src', 'nnabla_ext')}
    package_data = {}
    ext_modules = []

    cuda_ext = cuda_config(root_dir, cuda_lib, get_cpu_extopts(cpu_lib), libs)
    packages += cuda_ext.packages
    package_dir.update(cuda_ext.package_dir)
    package_data.update(cuda_ext.package_data)
    ext_modules += cuda_ext.ext_modules

    cudnn_ext = cudnn_config(root_dir, cuda_lib, cuda_ext.ext_opts)
    packages += cudnn_ext.packages
    package_dir.update(cudnn_ext.package_dir)
    package_data.update(cudnn_ext.package_data)
    ext_modules += cudnn_ext.ext_modules

    utils_ext = utils_config(root_dir, cuda_ext.ext_opts)
    packages += utils_ext.packages
    package_dir.update(utils_ext.package_dir)
    package_data.update(utils_ext.package_data)
    ext_modules += utils_ext.ext_modules

    cuda_version = ''.join(__cuda_version__.split('.'))

    if 'WHL_NO_CUDA_SUFFIX' in os.environ and os.environ['WHL_NO_CUDA_SUFFIX'] == 'True':
        cuda_version = ''

    if 'MULTI_GPU_SUFFIX' in os.environ:
        cuda_version += os.environ['MULTI_GPU_SUFFIX']

    pkg_name = 'nnabla_ext-cuda{}'.format(cuda_version)

    if 'WHEEL_SUFFIX' in os.environ:
        pkg_name += os.environ['WHEEL_SUFFIX']

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
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX :: Linux',
            'Operating System :: MacOS :: MacOS X'
        ],
        platforms=['CUDA {}'.format(__cuda_version__),
                   'cuDNN {}'.format(__cudnn_version__)],
        keywords="deep learning artificial intelligence machine learning neural network cuda",
        python_requires='>=3.5',
    )
    return pkg_info, ExtConfig(package_dir, packages, package_data, ext_modules, {})


if __name__ == '__main__':
    from Cython.Build import cythonize

    pkg_info, cfg = get_setup_config(root_dir)

    # Cythonize
    ext_modules = cythonize(cfg.ext_modules,
                            include_path=[get_cpu_cython_path()],
                            compiler_directives={
                                "embedsignature": True,
                                "c_string_type": 'str',
                                "c_string_encoding": "ascii"})

    # Setup
    setup(
        setup_requires=setup_requires,
        install_requires=install_requires,
        ext_modules=ext_modules,
        package_dir=cfg.package_dir,
        packages=cfg.packages,
        package_data=cfg.package_data,
        namespace_packages=['nnabla_ext'],
        **pkg_info)

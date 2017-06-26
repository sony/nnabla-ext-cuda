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

'''
Python API setup script.

This hasn't been designed well yet. Polishing it is future developement.
The current improvement plan is as following:

* Cythonize before publishing. Users do not use Cython, instead just use
generated C++ files to build extensions.

* Remove hard-coded relative paths for library link. Maybe the solution will be
  install NNabla C++ library in order to put them into folders path of which
  are set.
'''
from setuptools import setup
from distutils.extension import Extension
from os.path import dirname, realpath, join, isfile, splitext
import shutil
from collections import namedtuple
import copy
import sys

setup_requires = [
    'numpy>=1.12.0',
    'Cython>=0.24',  # Requires python-dev.
]

install_requires = [
    'nnabla>=0.9.1rc3',
]

LibInfo = namedtuple('LibInfo', ['file_name', 'path', 'name'])
ExtConfig = namedtuple('ExtConfig',
                       ['package_dir', 'packages', 'package_data',
                        'ext_modules', 'ext_opts'])


def get_libinfo():
    from ConfigParser import ConfigParser

    # Parse setup.cfg
    path_cfg = join(dirname(__file__), "setup.cfg")
    if not isfile(path_cfg):
        raise ValueError(
            "`setup.cfg` does not exist. Read installation document and install using CMake.")
    cfgp = ConfigParser()
    cfgp.read(path_cfg)

    # Read cpu lib info
    cpu_lib = LibInfo(cfgp.get("cmake", "cpu_target_file_name"),
                      cfgp.get("cmake", "cpu_target_file"),
                      cfgp.get("cmake", "cpu_target_name"))
    print("CPU Library name:", cpu_lib.name)
    print("CPU Library file name:", cpu_lib.file_name)
    print("CPU Library file:", cpu_lib.path)

    # Read cuda lib info
    cuda_lib = LibInfo(cfgp.get("cmake", "cuda_target_file_name"),
                       cfgp.get("cmake", "cuda_target_file"),
                       cfgp.get("cmake", "cuda_target_name"))
    print("CUDA Library name:", cuda_lib.name)
    print("CUDA Library file name:", cuda_lib.file_name)
    print("CUDA Library file:", cuda_lib.path)
    cudnn_version = cfgp.get("cmake", "cudnn_version")
    return cpu_lib, cuda_lib, cudnn_version


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


def cuda_config(root_dir, cuda_lib, ext_opts):
    # With CUDA
    src_dir = join(root_dir, 'src')
    path_cuda_pkg = join(src_dir, 'nnabla_ext', 'cuda')
    cuda_pkg = "nnabla_ext.cuda"
    package_dir = {cuda_pkg: path_cuda_pkg}
    packages = [cuda_pkg]

    cuda_lib_out = join(path_cuda_pkg, cuda_lib.file_name)
    shutil.copyfile(cuda_lib.path, cuda_lib_out)
    package_data = {cuda_pkg: [cuda_lib.file_name]}

    if sys.platform == 'win32':
        libdir = dirname(cuda_lib.path)
        libname, _ = splitext(cuda_lib.file_name)
        cuda_ext_lib_file_name = libname + '.lib'
        cuda_ext_lib_path = join(libdir, cuda_ext_lib_file_name)
        cuda_ext_lib_out = join(path_cuda_pkg, cuda_ext_lib_file_name)
        shutil.copyfile(cuda_ext_lib_path, cuda_ext_lib_out)
        package_data[cuda_pkg].append(cuda_ext_lib_file_name)

    cuda_ext_opts = copy.deepcopy(ext_opts)
    cuda_ext_opts['libraries'] += [cuda_lib.name]
    cuda_ext_opts['library_dirs'] += [dirname(cuda_lib.path)]
    ext_modules = [
        Extension(cuda_pkg + '.init',
                  [join(path_cuda_pkg, 'init.pyx')],
                  **cuda_ext_opts),
    ]
    return ExtConfig(package_dir, packages, package_data,
                     ext_modules, cuda_ext_opts)


def cudnn_config(root_dir, cuda_lib, cuda_ext_opts):
    src_dir = join(root_dir, 'src')
    path_cudnn_pkg = join(src_dir, 'nnabla_ext', 'cuda', 'cudnn')
    cudnn_pkg = 'nnabla_ext.cuda.cudnn'
    package_dir = {cudnn_pkg: path_cudnn_pkg}
    packages = [cudnn_pkg]
    ext_modules = [
        Extension(cudnn_pkg + '.init',
                  [join(path_cudnn_pkg, 'init.pyx')],
                  **cuda_ext_opts),
    ]
    return ExtConfig(package_dir, packages, {},
                     ext_modules, cuda_ext_opts)


def get_setup_config(root_dir):
    cpu_lib, cuda_lib, cudnn_version = get_libinfo()

    packages = ['nnabla_ext']
    package_dir = {'nnabla_ext': join(root_dir, 'src', 'nnabla_ext')}
    package_data = {}
    ext_modules = []

    cuda_ext = cuda_config(root_dir, cuda_lib, get_cpu_extopts(cpu_lib))
    packages += cuda_ext.packages
    package_dir.update(cuda_ext.package_dir)
    package_data.update(cuda_ext.package_data)
    ext_modules += cuda_ext.ext_modules

    cudnn_ext = cudnn_config(root_dir, cuda_lib, cuda_ext.ext_opts)
    packages += cudnn_ext.packages
    package_dir.update(cudnn_ext.package_dir)
    package_data.update(cudnn_ext.package_data)
    ext_modules += cudnn_ext.ext_modules

    # Embed signatures in Cython function and classes
    for e in ext_modules:
        e.cython_directives = {"embedsignature": True}
    pkg_info = dict(
        name="nnabla_ext-cuda",
        description='A CUDA and cuDNN extension of NNabla',
        version='0.9.1rc3',
        author_email='nnabla@googlegroups.com',
        url="https://github.com/sony/nnabla",
        license='Apache Licence 2.0',
        classifiers=[
                'Development Status :: 4 - Beta',
                'Intended Audience :: Developers',
                'Intended Audience :: Education',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
                'License :: OSI Approved :: Apache Software License',
                'Programming Language :: Python :: 2.7',
                'Operating System :: Microsoft :: Windows',
                'Operating System :: POSIX :: Linux',
            ],
        keywords="deep learning artificial intelligence machine learning neural network cuda",
        python_requires='>=2.7, <3')
    return pkg_info, ExtConfig(package_dir, packages, package_data, ext_modules, {})


if __name__ == '__main__':
    from Cython.Distutils import build_ext

    root_dir = realpath(dirname(__file__))
    pkg_info, cfg = get_setup_config(root_dir)
    setup(
        cmdclass={"build_ext": build_ext},
        setup_requires=setup_requires,
        install_requires=install_requires,
        ext_modules=cfg.ext_modules,
        package_dir=cfg.package_dir,
        packages=cfg.packages,
        package_data=cfg.package_data,
        namespace_packages=['nnabla_ext'],
        **pkg_info)

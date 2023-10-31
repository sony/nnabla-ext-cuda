# Build CUDA Extension

This document shows how to install CUDA extension on Ubuntu 20.04 LTS. We actually only tested on version: Ubuntu 20.04 LTS, cuda11.6.2, cudnn8.4.0, Python 3.8.10. This procedure should work on other Linux distributions. For a build instruction on Windows, go to:

* [Build on Windows](build_windows.md)


## Prerequisites

In addition to NNabla's requirements, CUDA extension requires CUDA setup has done on your system. If you don't have CUDA on your system, follow the procedure described below.

Download and install CUDA and cuDNN library (both runtime library and development library). Please follow the instruction in the document provided by NVIDIA. Do NOT see any instruction provided by any third party. They are often incorrect or based on old instructions, that could destroy your system.

* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
* [cuDNN library](https://developer.nvidia.com/rdp/cudnn-download) (Registration required)
 

## Build and installation

You needs to [build nnabla](https://github.com/sony/nnabla/blob/master/doc/build/build.md) before build nnabla-ext-cuda.
And, you have to let CMake to know where the source and library of NNabla are located by the following options.

- `NNABLA_DIR`: Root directory of NNabla source code.
- `CPPLIB_LIBRARY`: Path to `libnnabla.so` of nnabla.

The following will build and create a NNabla CUDA extension Python package (`pip` requires `sudo` if it's on your system's Python).

First, get the source of nnabla-ext-cuda and setup for build.

```shell
git clone https://github.com/sony/nnabla-ext-cuda
cd nnabla-ext-cuda
pip install -r python/requirements.txt
mkdir build
cd build
```

Then, build. You can optionally turn off the Python package build by `-DBUILD_PYTHON_PACKAGE=OFF`
and also turn on the C++ utils build by `-DBUILD_CPP_UTILS=ON` in `cmake`. But you must [build C++ utility libraries](https://github.com/sony/nnabla/blob/master/doc/build/build_cpp_utils.md) before you turn on the C++ utils build by `-DBUILD_CPP_UTILS=ON` in `cmake`.

```shell
cmake -DNNABLA_DIR=../../nnabla -DCPPLIB_LIBRARY=../../nnabla/build/lib/libnnabla.so ..
make
```

Note: If the reported error is "`FATAL_ERROR, python [python] not found`" in `cmake`. Please establish a soft connection `python` to `python3.x`, you can refer to the following command:

```shell
sudo ln -s /usr/bin/python3.x /usr/bin/python
```

To install the Python package:

```shell
cd dist
pip uninstall -y nnabla-ext-cuda
pip install nnabla_ext_cuda-{{version}}-{{arch}}.whl
```

(Optional for C++ standalone application with [C++ utility](https://github.com/sony/nnabla/tree/master/doc/build/build_cpp_utils.md) To install C++ library on your system:

```shell
sudo make install  # Set `CMAKE_INSTALL_PREFIX` to install in other place than your system (recommended).
```


## Running unit test

For unit testing, some additional requirements should be installed.

```shell
cd nnabla
sudo apt install -y liblzma-dev libsndfile1
pip install -r python/test_requirements.txt
```

Go to NNabla source root (not nnabla-ext-cuda).

```shell
cd {{nnabla root}}
```

Then run test of CUDA/cuDNN extension.

```shell
export PYTHONPATH={{PATH TO nnabla-ext-cuda}}/python/test:$PYTHONPATH
py.test python/test
```

Note: `py.test` might not in PATH which might cause `py.test: not found`, you can use `python3 -m pytest` to replace `py.test`.

## FAQ

* Q. Why do I need to reboot after installing CUDA/cuDNN?

  * CUDA driver may remain disabled. Therefore, you need to reboot the system and enable the driver.

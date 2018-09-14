# Build CUDA extension on Windows

## Prerequisites

In addition to NNabla's requirements, CUDA extension requires CUDA setup has done on your system. If you don't have CUDA on your system, follow the procedure described below.


Download and install CUDA and cuDNN library (both runtime library and development library). Please follow the instruction in the document provided by NVIDIA. Do NOT see any instruction provided by any third party. They are often incorrect or based on old instructions, that could destroy your system.

* [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
* [cuDNN library](https://developer.nvidia.com/rdp/cudnn-download) (Registration required)

## Build

You needs to [build nnabla](build.md) before build nnabla-ext-cuda.

Open a command prompt, and make sure you are in the conda environment you've created in NNabla's build.

```bat
> activate nnabla
(nnabla) >
```

The following will build and create a NNabla CUDA extension Python package (`pip` requires `sudo` if it's on your system's Python).

```bat
(nnabla) > git clone https://github.com/sony/nnabla-ext-cuda
(nnabla) > cd nnabla-ext-cuda
(nnabla) > mkdir build
(nnabla) > cd build
```

You have to let CMake to know where the source and library of NNabla are located by the following options.

- `NNABLA_DIR`: Root folder of NNabla source code.
- `CPPLIB_LIBRARY`: Path to `libnnabla.dll`

```bat
(nnabla) > cmake -G "Visual Studio 14 Win64" -DNNABLA_DIR=..\..\nnabla -DCPPLIB_LIBRARY=..\..\nnabla\build\bin\Release\nnabla.dll ..
(nnabla) > msbuild ALL_BUILD.vcxproj /p:Configuration=Release
```

You can optionally turn off Python package build by passing `-DBUILD_PYTHON_PACKAGE=OFF` to `cmake`.
To install a Python package:

```bat
(nnabla) > cd dist
(nnabla) > pip install -U nnabla_ext_cuda-<package version>-<package-arch>.whl
```

You can use C++ library (`nnabla_cuda.dll`) in your C++ application by combining with nnabla [C++ utility library](https://github.com/sony/nnabla/tree/master/doc/build/build_cpp_utils_windows.md), by passing include path (`include/`) and the library to your build system.

## Unit test

For unit testing, some additional requirements should be installed.

```bat
(nnabla) > pip install pytest
```

Then run:
```bat
(nnabla) > set PYTHONPATH={{PATH TO nnabla-ext-cuda}}\python\test;%PYTHONPATH%
(nnabla) > py.test nnabla\python\test
```

Deactivate.

```bat
(nnabla) > deactivate
```

## FAQ

* Q. The compiled library and executable run on a compiled PC, but it does not work on another PC.
  * If the installed GPU is different between build PC and target PC, try the following cmake option when you build nnabla-ext-cuda.

```bat
> cmake -G "Visual Studio 14 Win64" -D CUDA_SELECT_NVCC_ARCH_ARG:STRING="All" ..\
```

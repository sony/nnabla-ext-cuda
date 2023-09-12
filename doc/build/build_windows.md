# Build CUDA extension on Windows

## Prerequisites

At first, clone [nnabla](https://github.com/sony/nnabla) and [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda) into same folder.

Then, install CUDA11.0/cuDNN8.0 or CUDA11.4/cuDNN8.2 from following site.
- CUDA
 - https://developer.nvidia.com/cuda-toolkit-archive

Get several versions of cuDNN from following site. (Registration required)
- cuDNN
 - https://developer.nvidia.com/rdp/cudnn-archive

Before building nnabla-ext-cuda, build nnabla with [Instruction](https://github.com/sony/nnabla/blob/master/doc/build/build_windows.md).


### Build cpplib

You can build cpplib for nnabla-ext-cuda with following command.

Note: parameters `CUDA_VERSION` and `CUDNN_VERSION` are options of cuda/cudnn versions.

``` cmd
build-tools\msvc\build_cpplib.bat CUDA_VERSION CUDNN_VERSION
``` 
For examples:

```
build-tools\msvc\build_cpplib.bat 11.0 8
```
or:
```
build-tools\msvc\build_cpplib.bat 11.4 8
```

Tested version

    CUDA/cuDNN: 11.0/8.0 11.4/8.2

### Build wheel
Note: parameters `PYTHON_VERSION`, `CUDA_VERSION` and `CUDNN_VERSION` are options of python, cuda and cudnn versions.
```
build-tools\msvc\build_wheel.bat PYTHON_VERSION CUDA_VERSION CUDNN_VERSION
```
For examples:

```
build-tools\msvc\build_wheel.bat 3.8 11.0 8
```
or:
```
build-tools\msvc\build_wheel.bat 3.8 11.4 8
```

Tested version

    PYTHON: 3.8 3.9 3.10
    CUDA/cuDNN: 11.0/8.0 11.4/8.2

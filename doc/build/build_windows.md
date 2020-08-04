# Build CUDA extension on Windows

## Prerequisites

At first, clone [nnabla](https://github.com/sony/nnabla) and [nnabla-ext-cuda](https://github.com/sony/nnabla-ext-cuda) into same folder.

Then, install CUDA9.0, CUDA10.0 or CUDA10.2 from following site.
- CUDA
 - https://developer.nvidia.com/cuda-toolkit-archive

Get several versions of cuDNN from following site. (Registration required)
- cuDNN
 - https://developer.nvidia.com/rdp/cudnn-download

Make sure that set `CUDNN_PATH` variable. 

Before building nnabla-ext cuda, build nnabla with [Instruction](https://github.com/sony/nnabla/doc/build/build_windows.md).


### Build cpplib

You can build cpplib for nnabla-ext-cuda with following command.

``` cmd
build-tools\msvc\build_cpplib.bat CUDA_VERSION CUDNN_VERSION
```

Tested version

    CUDA: 9.0 10.0 10.2
    cuDNN: 7

### Build wheel
```
build-tools\msvc\build_wheel.bat PYTHON_VERSION CUDA_VERSION CUDNN_VERSION
```

Tested version

    PYTHON: 3.6 3.7 3.8
    CUDA: 9.0 10.0 10.2
    cuDNN: 7

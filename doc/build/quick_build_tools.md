# Quick build tools

Make sure that `nnabla` and `nnabla-builder` are cloned on the same directory.

## Linux

You needs [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and `GNU make`

### Build
```
$ make all
```
`make all` is as same as `make bwd-cpplib bwd-wheel`
After this you can find .whl file in `../nnabla/build_wheel_py??/dist/` `build_wheel_py??_??_?/dist/`

### Build cpplib only
```
$ make bwd-cpplib
```

### Specify CUDA and cuDNN version.

Prepare to specify CUDA, cuDNN, and python version.
```
$ export PYTHON_VERSION_MAJOR=2
$ export PYTHON_VERSION_MINOR=7
$ export CUDA_VERSION_MAJOR=9
$ export CUDA_VERSION_MINOR=0
$ export CUDNN_VERSION=7
```

Then you can get with,
```
$ make all
```

Or you can specify every time.
```
$ make PYTHON_VERSION_MAJOR=2 PYTHON_VERSION_MINOR=7 CUDA_VERSION_MAJOR=9 CUDA_VERSION_MINOR=0 CUDNN_VERSION=7 all
```

## Windows

Install CUDA8.0, CUDA9.0, CUDA9.1 from following site.

- CUDA
 - https://developer.nvidia.com/cuda-toolkit-archive


Get several versions of cuDNN from following site. (Registration required)
- cuDNN
 - https://developer.nvidia.com/rdp/cudnn-download

Make sure that set `CUDNN_PATH` variable. 

### Build cpplib
```
> call build-tools\msvc\build_cpplib.bat
```

### Build wheel
```
> call build-tools\msvc\build_wheel.bat
```

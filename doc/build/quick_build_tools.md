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
$ export PYTHON_VERSION_MAJOR=3
$ export PYTHON_VERSION_MINOR=7
$ export CUDA_VERSION_MAJOR=10
$ export CUDA_VERSION_MINOR=2
$ export CUDNN_VERSION=8
```

Then you can get with,
```
$ make all
```

Or you can specify every time.
```
$ make PYTHON_VERSION_MAJOR=3 PYTHON_VERSION_MINOR=7 CUDA_VERSION_MAJOR=10 CUDA_VERSION_MINOR=2 CUDNN_VERSION=8 all
```

## Windows

Now, this section is same as [Build CUDA extension on Windows](build_windows.md)


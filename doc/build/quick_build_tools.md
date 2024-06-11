# Quick build tools

Make sure that [nnabla](https://github.com/sony/nnabla) and `nnabla-ext-cuda` are cloned on the same level directory.

## Linux

You needs [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) and `GNU make`

Note: workdir in "nnabla-ext-cuda/"

### Build
```
$ cd nnabla-ext-cuda
$ make all
```
`make all` is as same as `make bwd-cpplib bwd-wheel`
After this you can find .whl file in `../nnabla/build_wheel/dist/` and `build_wheel/dist/`

### Build cpplib only
```
$ cd nnabla-ext-cuda
$ make bwd-cpplib
```

### Specify CUDA and cuDNN version.

Prepare to specify CUDA, cuDNN, and python version.
```
$ export PYTHON_VERSION_MAJOR=3
$ export PYTHON_VERSION_MINOR=9
$ export CUDA_VERSION_MAJOR=11
$ export CUDA_VERSION_MINOR=6.2
$ export CUDNN_VERSION=8
```

Then you can get with,
```
$ cd nnabla-ext-cuda
$ make all
```

Or you can specify every time.
```
$ cd nnabla-ext-cuda
$ make PYTHON_VERSION_MAJOR=3 PYTHON_VERSION_MINOR=9 CUDA_VERSION_MAJOR=11 CUDA_VERSION_MINOR=6.2 CUDNN_VERSION=8 all
```

## Windows

Now, this section is same as [Build CUDA extension on Windows](build_windows.md)


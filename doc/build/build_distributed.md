# Build Python Package with distributed execution support powered by NCCL

Distributed execution is only supported on Linux. We tested it on Ubuntu 20.04 LTS, cuda11.4.1, cudnn8.2.4.

## Prerequisites

In addition to [requirements of NNabla without distributed execution](build.md), the build system requires:

* Multiple NVIDIA CUDA-capable GPUs
* [NCCL](https://developer.nvidia.com/nccl): NVIDIA's multi-GPU and multi-node collective communication library optimized for their GPUs.
* Open MPI

### NCCL

In order to use the distributed training, the only difference, when building, is
the procedure described here.

Download [NCCL](https://developer.nvidia.com/nccl) according to your environment,
then install it manually in case of ubuntu20.04,

```shell
sudo dpkg -i nccl-local-repo-ubuntu2004-2.11.4-cuda11.4_1.0-1_amd64.deb
sudo apt-get update
sudo apt-get install libnccl2=2.10.3-1+cuda11.4 libnccl-dev=2.10.3-1+cuda11.4
```

For developer, if you want to use another nccl not publicly distributed,
specify **NCCL_HOME** environment variable as the following.

```shell
export NCCL_HOME=${path}/build
```

Here, we assume the directory structure,

* ${path}/build/include
* ${path}/build/lib


### MPI
Distributed training also depends on MPI, so install it as follows,

```shell
sudo apt-get install libopenmpi-dev
```


## Build and Install

Follow [Build CUDA extension](build.md) with a little modification in CMake's option (`-DWITH_NCCL=ON`).

```shell
cmake -DNNABLA_DIR=../../nnabla -DCPPLIB_LIBRARY=../../nnabla/build/lib/libnnabla.so -DWITH_NCCL=ON ..
```

Note: If the reported error is "`FATAL_ERROR, python [python] not found`" in `cmake`. Please establish a soft connection `python` to `python3.x`, you can refer to the following command:

```shell
sudo ln -s /usr/bin/python3.x /usr/bin/python
```

You can confirm nccl and mpi includes and dependencies found in the output screen,

```
...

CUDA libs: /usr/local/cuda-8.0/lib64/libcudart.so;/usr/local/cuda-8.0/lib64/libcublas.so;/usr/local/cuda-8.0/lib64/libcurand.so;/usr/lib/x86_64-linux-gnu/libnccl.so;/usr/lib/openmpi/lib/libmpi_cxx.so;/usr/lib/openmpi/lib/libmpi.so;/usr/local/cuda/lib64/libcudnn.so
CUDA includes: /usr/local/cuda-8.0/include;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent;/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include;/usr/lib/openmpi/include;/usr/lib/openmpi/include/openmpi;/usr/local/cuda-8.0/include
...
```

## Unit test


Follow the unit test section in [Build CUDA extension](build.md). Now you could see the communicator
test passed.

```
...
...
communicator/test_data_parallel_communicator.py::test_data_parallel_communicator PASSED
...
```


Now you can use **Data Parallel Distributed Training** using multiple GPUs and multiple nodes, please
go to [CIFAR-10 example](https://github.com/sony/nnabla-examples/tree/master/image-classification/cifar10-100) for the usage.

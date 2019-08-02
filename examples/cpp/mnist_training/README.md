# C++ training with MNIST classification model.

## Introduction

This example is the GPU version of MNIST classification model training in C++.
Please execute the CPU version C++ classification model learning sample in `nnabla` repository first.

# Install C++ libraries for GPU

In addition to building CPU version in `nnabla` repository, you need to build Cuda version in `nnabla-ext-cuda`.
Please follow [the installation manual](https://github.com/sony/nnabla-ext-cuda/blob/master/doc/build/build.md).

Also MNIST dataset is required in the same directory.
If you tried the CPU version of this script, you must have downloaded MNIST data in `nnabla` directory.
* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz

Please copy them to this directory.

## Create NNP file of an initialized model for MNIST classification.
You might also have an NNP file of the initialized model in `nnabla/examples/cpp/mnist_training`.
Please copy it to this directory.

## Build MNIST training example in C++ code
You can find an executable file 'mnist_training_cuda' under the build directory located at nnabla-ext-cuda/build/bin.

If you want to build it yourself using Makefile you can refer to the following commands in linux environments.

```shell
export NNABLA_DIR='path to your nnabla directory'
make
```

The above command generates an executable `mnist_training_cuda` at the current directory.

The build file `GNUmakefile` is simple.
It links `libnnabla.so`, `libnnabla_cuda.so`, `libnnabla_utils.so` and `libz.so` with the executable generated from `main.cpp`, and compiles with C++11 option `-std=c++11`.
It also needs to include path to `mnist_training.hpp` which is located in `nnabla/examples/cpp/mnist_training` directory.

## Handwritten digit training
By running the generated example with no argument, you can see the usage documentation.

```shell
./mnist_training_cuda
```

Output:
```

Usage: ./mnist_training_cuda model.nnp

  model.nnp : model file with initialized parameters.

```

The following command executes the training of the initialized model `lenet_initialized.nnp` on MNIST dataset.

```shell

./mnist_training_cuda lenet_initialized.nnp

```

The output file named `parameter.protobuf` contains the learned parameters.

Following process is temporary and at a later date, we will prepare a save function for nnp.

```shell
 cp lenet_initialized.nnp lenet_learned.nnp
 unzip lenet_learned.nnp
 zip lenet_learned.nnp nnp_version.txt network.nntxt parameter.protobuf
```

You will be asked "replace parameter.protobuf?" when unzipping, so please answer "n".

After getting learned.nnp, you can use it as a model file for "mnist_runtime".


## Walk through the example code
[main.cpp][1]
[1]:main.cpp
1. Add NNabla headers.
```c++
  #include <nbla/context.hpp>
  #include <nbla/cuda/cudnn/init.hpp>
  #include <nbla/cuda/init.hpp>
```

2. Initialized cudnn.
```c++
  nbla::init_cudnn();
```
3. Create an execution engine context.
```c++
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};
```

4. Execute training.
```c++
  mnist_training(ctx, argv[1]);
```

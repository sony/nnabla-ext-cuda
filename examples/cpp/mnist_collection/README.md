# C++ training examples with MNIST dataset.

## Introduction

These examples are the GPU version of MNIST training in C++.
Please execute the CPU version C++ training sample in `nnabla` repository first.

# Install C++ libraries for GPU

In addition to building CPU version in `nnabla` repository, you need to build Cuda version in `nnabla-ext-cuda`.
Please follow [the installation manual](https://github.com/sony/nnabla-ext-cuda/blob/master/doc/build/build.md).

Also [MNIST dataset](https://github.com/sony/nnabla/tree/master/examples/cpp/mnist_collection#install-c-libraries) is required in the same level directory.
If you tried the CPU version of this script, you must have downloaded MNIST data in `nnabla` directory.
* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

Please copy them to this directory.

## Build training example in C++ code
You can find executable files `train_lenet_classifier_cuda`, `train_resnet_classifier_cuda`, `train_vae_cuda`, `train_siamese_cuda`, `train_dcgan_cuda` and `train_vat_cuda` under the build directory located at nnabla-ext-cuda/build/bin.

Note: When you build the cuda version in `nnabla-ext-cuda` repository, please turn on the C++ utils build by `-DBUILD_CPP_UTILS=ON` in `cmake`. Otherwise there is no `nnabla-ext-cuda/build/bin` directory.

If you want to build it yourself using Makefile you can refer to the following commands in linux environments.

```shell
export NNABLA_DIR='path to your nnabla directory'
make lenet
```

The above command generates an executable `train_lenet_classifier_cuda` at the current directory.

The makefile `GNUmakefile` is simple.
It links `libnnabla.so`, `libnnabla_cuda.so` and `libz.so` with the executable generated from `main.cpp`, and compiles with C++11 option `-std=c++14`.
It also needs to include path to `lenet_training.hpp` which is located in `nnabla/examples/cpp/mnist_collection` directory.

You can also compile other executables by using make command with proper options.

## Handwritten digit training
By running the generated example with no argument, you can see the usage documentation.

```shell
./train_lenet_classifier_cuda
```

Output:
```
iter: 9, tloss: 2.245488, verr: 0.586719
iter: 19, tloss: 1.886091, verr: 0.426562
iter: 29, tloss: 1.395539, verr: 0.295312
iter: 39, tloss: 0.961441, verr: 0.225000
iter: 49, tloss: 0.688315, verr: 0.146875
iter: 59, tloss: 0.499045, verr: 0.133594
```

You can also run other examples by using each executable.


## Walk through the example code
[train_lenet_classifier_cuda.cpp][1]
[1]:train_lenet_classifier_cuda.cpp
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
  lenet_training(ctx);
```

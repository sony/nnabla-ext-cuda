# C++ infer with semantic segmentation trained model.

## Introduction

This example is infer with semantic segmentation trained model by c++. The default trained model "DeepLabV3-voc-os-16.nnp" download from "https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-os-16.nnp". After builded, it can infer by CPU or CUDA by switch macro "WITH_CUDA". Refer to follow steps to build program and execute infer.

### 1.1 Install NNABLA by offical tutorial

The offical tutorial web is:

https://github.com/nnabla/nnabla/blob/master/doc/build/README.md

Either python package or build c++ utilities are OK. But C++ utilities library will more easier be finded by cmake and make.

### 1.2 Install OpenCV by offical tutorial

The web is:

https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html

If you use cmake with following procedure, libopencv-dev package will be needed to find opencv libraries by find_package().

### 1.3 Install nnabla-ext-cuda by offical tutorial(optional)

The web is:

https://github.com/nnabla/nnabla-ext-cuda/blob/master/doc/build/README.md

Note: the nnabla and nnabla-ext-cuda must in the same folder.

If not install nnabla-ext-cuda the program only infer by CPU.

### 1.4 Install curl library
```shell
sudo apt-get update
sudo apt-get install curl
sudo apt-get install libcurl4-gnutls-dev
```

### 2 Build cpp program

build by cmake

Note: If cmake cannot find nnabla libraries, change the HINT paths to find_library() for nnabla and nnabla_utils in CMakeList.txt.

```shell
cd nnabla-ext-cuda/examples/cpp/semantic_infer
mkdir -p build && cd build
cmake ..     # "cmake .. -DWITH_CUDA=ON" to build with cuda
make
```

build by make

Note: If the make cannot find nnabla nnabla_utils and nnabla_cuda library location, refer to the comment in GNUmakefile.

```shell
cd nnabla-ext-cuda/examples/cpp/semantic_infer
make    # "make cuda" to make with cuda
```

### 3 Run inference

Run it and see hints about the command.

```shell
./semantic DeepLabV3-voc-os-16.nnp test.jpg
```

The program find DeepLabV3-voc-os-16.nnp according to command, if not find it will find DeepLabV3-voc-os-16.nnp under "\~/nnabla_data/nnp_models/semantic_segmentation/", if not find it will download nnp from "https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/DeepLabV3-voc-os-16.nnp", and save to "~/nnabla_data/nnp_models/semantic_segmentation/DeepLabV3-voc-os-16.nnp".

Due to the precision of the nnp model and the input of images, although object recognition are less accurate, the model still can label the object.

NOTE: Please make sure the model is downloaded successfully, if the model is not completely downloaded, it can still be executed, but the result will not be displayed correctly.

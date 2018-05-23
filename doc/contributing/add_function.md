# Adding a new function

## Overview

This document demonstrates how to add CUDA version of `functions` (also called as `layer functions`) to nnabla-ext-cuda.  
The following instruction demonstrates the new layer addition procedure for Ubuntu 16.04 platform.   
However instructions for macOS and Windows platforms are also same except nnabla-ext-cuda building part.  

## Prerequisites

Download the Neural Network Libraries with CUDA extension from https://github.com/sony/nnabla-ext-cuda and build from source.  
For detailed instructions on building nnabla-ext-cuda from source please see [Build Manuals]( https://github.com/sony/nnabla-ext-cuda/blob/master/doc/build/README.md).  
Add new function layer for CPU version. Please see [Add function manual for NNabla]( https://github.com/sony/nnabla/blob/master/doc/contributing/add_function.md) for details.  

## Write a definition in YAML

The nnabla-ext-cuda uses the same [functions.yaml](https://github.com/sony/nnabla/tree/master/build-tools/code_generator/functions.yaml) present in nnabla for new function definition.  
As a first step for adding CUDA layer function, you need to add a type specification of the new function to [function_types.yaml](https://github.com/sony/nnabla-ext-cuda/blob/master/build-tools/code_generator/function_types.yaml) present in nnabla-ext-cuda.   
The syntax of CUDA `function_types.yaml` file remains same as that of nnabla.   
Please see [Add function manual for NNabla]( https://github.com/sony/nnabla/blob/master/doc/contributing/add_function.md) for nnabla `function_types.yaml` syntax details.  

## Generate code template by CMake

After editing the YAML file, next step is to generate the template class for new function for CUDA.  
By running cmake as described in the build instruction (Please refer [Build Manuals](https://github.com/sony/nnabla-ext-cuda/blob/master/doc/build/README.md).),  

```shell
cd build
cmake -DNNABLA_DIR=../../nnabla -DCPPLIB_LIBRARY=../../nnabla/build/lib/libnnabla.so ..
```
new function class template is generated at the following locations.  

* `include/nbla/cuda/function/${snake_name}.hpp`
* `src/nbla/cuda/function/generic/${snake_name}.cu`

## Write your function implementation

The template files generated using cmake contains the class definition for new function.  
The new function class inherits from `${class_name_of_new_function}` defined in [functions.yaml](https://github.com/sony/nnabla/tree/master/build-tools/code_generator/functions.yaml).  
In order to add CUDA version of new layer function, you must implement setup, forward, and backward functions in ${snake_name}.cu file.  
For details on setup, forward, and backward functions please refer [here]( https://github.com/sony/nnabla/blob/master/doc/contributing/add_function.md).  

## Write a test (CUDA)

The tests for nnabla-ext-cuda are mostly entirely included in the nnabla tests.   
For example, the CUDA tests are run in the nnabla Pytests. Therefore, if you have already written a test in the nnabla section, you may skip this section.  
If your new feature is CUDA-specific, write a test inside nnabla-builder/nnabla-ext-cuda/python/test.  

## Add the function to the docs (CUDA)

This is basically the same as the nnabla section.  
Since the APIs for the extension is basically the same as the NNabla core, you may skip this section if you have already written the docs in the nnabla section.  

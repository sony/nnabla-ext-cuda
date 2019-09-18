// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/my_cuda_memset.hpp>

#include <device_launch_parameters.h>

namespace nbla {

__global__ void my_cudaMemset_kernel(size_t count,
                                     unsigned char *devPtr,
                                     unsigned char value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) devPtr[i] = value;
}


void my_cudaMemset(void *devPtr, int value, size_t count) {
  unsigned char *ptr = static_cast<unsigned char*>(devPtr);
  unsigned char val = (unsigned char)value;

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(my_cudaMemset_kernel,
                                 count, ptr, value);
}

}
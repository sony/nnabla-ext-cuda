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

namespace nbla {

int cuda_set_device(int device) {
  int current_device = cuda_get_device();
  if (current_device != device) {
    NBLA_CUDA_CHECK(cudaSetDevice(device));
  }
  return current_device;
}

int cuda_get_device() {
  int current_device;
  NBLA_CUDA_CHECK(cudaGetDevice(&current_device));
  return current_device;
}

std::vector<size_t> cuda_mem_get_info() {
  size_t mf, mt;
  cudaMemGetInfo(&mf, &mt);
  return std::vector<size_t>{mf, mt};
}

cudaDeviceProp cuda_get_current_device_properties() {
  cudaDeviceProp prop;
  int device = cuda_get_device(); // Note: Assuming device is properly set.
  NBLA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop;
}
}

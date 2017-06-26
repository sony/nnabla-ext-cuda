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

#ifndef __NBLA_CUDA_INIT_HPP__
#define __NBLA_CUDA_INIT_HPP__

#include <nbla/cuda/defs.hpp>

#include <string>
#include <vector>

namespace nbla {

using std::vector;
using std::string;

/**
Initialize CUDA features.
*/
NBLA_CUDA_API void init_cuda();

/** Clear all CUDA memory cache
*/
NBLA_CUDA_API void clear_cuda_memory_cache();

/** Get CUDA array classes.
*/
NBLA_CUDA_API vector<string> cuda_array_classes();

/** Set CUDA array classes
*/
NBLA_CUDA_API void _cuda_set_array_classes(const vector<string> &a);

/** Wrapper of cudaDeviceSynchronize
*/
NBLA_CUDA_API void cuda_device_synchronize(int device);

/** Wrapper of cudaGetDeviceCount
*/
NBLA_CUDA_API int cuda_get_device_count();
}
#endif

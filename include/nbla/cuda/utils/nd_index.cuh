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

#ifndef __NBLA_CUDA_UTILS_ND_INDEX_CUH__
#define __NBLA_CUDA_UTILS_ND_INDEX_CUH__

/***
    ND-index to Flat-index functions
 ***/
__forceinline__ __device__ int device_2d_to_flat(int2 nd_index, int stride) {
  return nd_index.x * stride + nd_index.y;
}

__forceinline__ __device__ int device_3d_to_flat(int3 nd_index, int2 stride) {
  return nd_index.x * stride.x + nd_index.y * stride.y + nd_index.z;
}

__forceinline__ __device__ int device_4d_to_flat(int4 nd_index, int3 stride) {
  return nd_index.x * stride.x + nd_index.y * stride.y + nd_index.z * stride.z +
         nd_index.w;
}

/***
    Flat-index to Nd-index functions
 ***/

__forceinline__ __device__ int2 device_flat_to_2d(int index, int stride) {
  auto x = index / stride;
  index -= x * stride;
  auto y = index;
  return make_int2(x, y);
}

__forceinline__ __device__ int3 device_flat_to_3d(int index, int2 stride) {
  auto x = index / stride.x;
  index -= x * stride.x;
  auto y = index / stride.y;
  index -= y * stride.y;
  auto z = index;
  return make_int3(x, y, z);
}

__forceinline__ __device__ int4 device_flat_to_4d(int index, int3 stride) {
  auto x = index / stride.x;
  index -= x * stride.x;
  auto y = index / stride.y;
  index -= y * stride.y;
  auto z = index / stride.z;
  index -= z * stride.z;
  auto w = index;
  return make_int4(x, y, z, w);
}

#endif
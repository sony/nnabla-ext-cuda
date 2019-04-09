// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#ifndef __NBLA_CUDA_UTILS_WARP_SHUFFLE_CUH__
#define __NBLA_CUDA_UTILS_WARP_SHUFFLE_CUH__

#include <nbla/cuda/utils/types.cuh>

namespace nbla {
namespace warp {

///////////////////////////////////////////////////////////////////////////////
//
// SHUFFLE
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ T shuffle(T val, int lane, int width = warpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(0xfffffff, val, lane, width);
#else  // !(CUDA_VERSION >= 9000)
  return __shfl(val, lane, width);
#endif // CUDA_VERSION >= 9000
}

#if CUDA_VERSION < 8000
template <>
__forceinline__ __device__ half shuffle<half>(half val, int lane, int width) {
  // Use uint32 because there is no overload for uint16.
  return half{(unsigned short)(__shfl((unsigned int)val.x, lane, width))};
}
#endif // CUDA_VERSION < 8000

template <>
__forceinline__ __device__ HalfCuda shuffle<HalfCuda>(HalfCuda val, int lane,
                                                      int width) {
#if CUDA_VERSION >= 9000
  return __shfl_sync(0xfffffff, val.h, lane, width);
#else // !(CUDA_VERSION >= 9000)
#if CUDA_VERSION >= 8000
  return __shfl(val.h, lane, width);
#else  // CUDA_VERSION >= 8000
  unsigned int val_ = val.h.x;
  return half{(unsigned short)(__shfl(val_, lane, width))};
#endif // CUDA_VERSION >= 8000
#endif // CUDA_VERSION >= 9000
}

template <>
__forceinline__ __device__ float2 shuffle(float2 val, int lane, int width) {
  return make_float2(shuffle(val.x, lane, width), shuffle(val.y, lane, width));
}

template <>
__forceinline__ __device__ float3 shuffle(float3 val, int lane, int width) {
  return make_float3(shuffle(val.x, lane, width), shuffle(val.y, lane, width),
                     shuffle(val.z, lane, width));
}

template <>
__forceinline__ __device__ float4 shuffle(float4 val, int lane, int width) {
  return make_float4(shuffle(val.x, lane, width), shuffle(val.y, lane, width),
                     shuffle(val.z, lane, width), shuffle(val.w, lane, width));
}

template <>
__forceinline__ __device__ floatint shuffle(floatint val, int lane, int width) {
  floatint buff;
  buff.f = shuffle(val.f, lane, width);
  buff.i = shuffle(val.i, lane, width);
  return buff;
}

///////////////////////////////////////////////////////////////////////////////
//
// SHUFFLE DOWN
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ T shuffle_down(T val, int offset,
                                          int width = warpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(0xfffffff, val, offset, width);
#else  // !(CUDA_VERSION >= 9000)
  return __shfl_down(val, offset, width);
#endif // CUDA_VERSION >= 9000
}

#if CUDA_VERSION < 8000
template <>
__forceinline__ __device__ half shuffle_down<half>(half val, int offset,
                                                   int width) {
  unsigned int val_ =
      val.x; // Use uint32 because there is no overload for uint16.
  return half{(unsigned short)(__shfl_down(val_, offset, width))};
}
#endif // CUDA_VERSION < 8000

template <>
__forceinline__ __device__ HalfCuda shuffle_down<HalfCuda>(HalfCuda val,
                                                           int offset,
                                                           int width) {
#if CUDA_VERSION >= 9000
  return __shfl_down_sync(0xfffffff, val.h, offset, width);
#else // !(CUDA_VERSION >= 9000)
#if CUDA_VERSION >= 8000
  return __shfl_down(val.h, offset, width);
#else // CUDA_VERSION >= 8000
  unsigned int val_ = val.h.x;
  return half{(unsigned short)(__shfl_down(val_, offset, width))};
#endif
#endif // CUDA_VERSION >= 9000
}

template <>
__forceinline__ __device__ float2 shuffle_down(float2 val, int offset,
                                               int width) {
  return make_float2(shuffle_down(val.x, offset, width),
                     shuffle_down(val.y, offset, width));
}

template <>
__forceinline__ __device__ float3 shuffle_down(float3 val, int offset,
                                               int width) {
  return make_float3(shuffle_down(val.x, offset, width),
                     shuffle_down(val.y, offset, width),
                     shuffle_down(val.z, offset, width));
}

template <>
__forceinline__ __device__ float4 shuffle_down(float4 val, int offset,
                                               int width) {
  return make_float4(
      shuffle_down(val.x, offset, width), shuffle_down(val.y, offset, width),
      shuffle_down(val.z, offset, width), shuffle_down(val.w, offset, width));
}

template <>
__forceinline__ __device__ floatint shuffle_down(floatint val, int offset,
                                                 int width) {
  floatint buff;
  buff.f = shuffle_down(val.f, offset, width);
  buff.i = shuffle_down(val.i, offset, width);
  return buff;
}

///////////////////////////////////////////////////////////////////////////////
//
// SHUFFLE UP
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ T shuffle_up(T val, int offset,
                                        int width = warpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(0xfffffff, val, offset, width);
#else  // !(CUDA_VERSION >= 9000)
  return __shfl_up(val, offset, width);
#endif // CUDA_VERSION >= 9000
}

#if CUDA_VERSION < 8000
template <>
__forceinline__ __device__ half shuffle_up<half>(half val, int offset,
                                                 int width) {
  unsigned int val_ =
      val.x; // Use uint32 because there is no overload for uint16.
  return half{(unsigned short)(__shfl_up(val_, offset, width))};
}
#endif // CUDA_VERSION < 8000

template <>
__forceinline__ __device__ HalfCuda shuffle_up<HalfCuda>(HalfCuda val,
                                                         int offset,
                                                         int width) {
#if CUDA_VERSION >= 9000
  return __shfl_up_sync(0xfffffff, val.h, offset, width);
#else // !(CUDA_VERSION >= 9000)
#if CUDA_VERSION >= 8000
  return __shfl_up(val.h, offset, width);
#else // CUDA_VERSION >= 8000
  unsigned int val_ = val.h.x;
  return half{(unsigned short)(__shfl_up(val_, offset, width))};
#endif
#endif // CUDA_VERSION >= 9000
}

template <>
__forceinline__ __device__ float2 shuffle_up(float2 val, int offset,
                                             int width) {
  return make_float2(shuffle_up(val.x, offset, width),
                     shuffle_up(val.y, offset, width));
}

template <>
__forceinline__ __device__ float3 shuffle_up(float3 val, int offset,
                                             int width) {
  return make_float3(shuffle_up(val.x, offset, width),
                     shuffle_up(val.y, offset, width),
                     shuffle_up(val.z, offset, width));
}

template <>
__forceinline__ __device__ float4 shuffle_up(float4 val, int offset,
                                             int width) {
  return make_float4(
      shuffle_up(val.x, offset, width), shuffle_up(val.y, offset, width),
      shuffle_up(val.z, offset, width), shuffle_up(val.w, offset, width));
}

template <>
__forceinline__ __device__ floatint shuffle_up(floatint val, int offset,
                                               int width) {
  floatint buff;
  buff.f = shuffle_up(val.f, offset, width);
  buff.i = shuffle_up(val.i, offset, width);
  return buff;
}

///////////////////////////////////////////////////////////////////////////////
//
// SHUFFLE XOR
//
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ T shuffle_xor(T val, int mask, int width = 32) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xfffffff, val, mask, width);
#else  // !(CUDA_VERSION >= 9000)
  return __shfl_xor(val, mask, width);
#endif // CUDA_VERSION >= 9000
}

#if CUDA_VERSION < 8000
template <>
__forceinline__ __device__ half shuffle_xor<half>(half val, int mask,
                                                  int width) {
  unsigned int val_ =
      val.x; // Use uint32 because there is no overload for uint16.
  return half{(unsigned short)(__shfl_xor(val_, mask, width))};
}
#endif // CUDA_VERSION < 8000

template <>
__forceinline__ __device__ HalfCuda shuffle_xor<HalfCuda>(HalfCuda val,
                                                          int mask, int width) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xfffffff, val.h, mask, width);
#else // !(CUDA_VERSION >= 9000)
#if CUDA_VERSION >= 8000
  return __shfl_xor(val.h, mask, width);
#else  // CUDA_VERSION >= 8000
  unsigned int val_ = val.h.x;
  return half{(unsigned short)(__shfl_xor(val_, mask, width))};
#endif // CUDA_VERSION >= 8000
#endif // CUDA_VERSION >= 9000
}

template <>
__forceinline__ __device__ float2 shuffle_xor(float2 val, int mask, int width) {
  return make_float2(shuffle_xor(val.x, mask, width),
                     shuffle_xor(val.y, mask, width));
}

template <>
__forceinline__ __device__ float3 shuffle_xor(float3 val, int mask, int width) {
  return make_float3(shuffle_xor(val.x, mask, width),
                     shuffle_xor(val.y, mask, width),
                     shuffle_xor(val.z, mask, width));
}

template <>
__forceinline__ __device__ float4 shuffle_xor(float4 val, int mask, int width) {
  return make_float4(
      shuffle_xor(val.x, mask, width), shuffle_xor(val.y, mask, width),
      shuffle_xor(val.z, mask, width), shuffle_xor(val.w, mask, width));
}

template <>
__forceinline__ __device__ floatint shuffle_xor(floatint val, int mask,
                                                int width) {
  floatint buff;
  buff.f = shuffle_xor(val.f, mask, width);
  buff.i = shuffle_xor(val.i, mask, width);
  return buff;
}

} // namespace warp
} // namespace nbla

#endif

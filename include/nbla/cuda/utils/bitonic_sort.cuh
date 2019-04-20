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

#ifndef __NBLA_CUDA_UTILS_BITONIC_SORT_CUH__
#define __NBLA_CUDA_UTILS_BITONIC_SORT_CUH__

namespace nbla {
namespace bitonic_sort_details {
// bfe(val, pos) returns the one bit of `val` located at `pos`,
// e.g. bfe(0x01020304, 17) => 0x00000001
__device__ __forceinline__ unsigned int bfe(unsigned int val,
                                            unsigned int pos) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, 1;" : "=r"(ret) : "r"(val), "r"(pos));
  return ret;
}

// bfe(val, pos, len) returns `len` bits of `val` starting at `pos`,
// e.g. bfe(0x01020304, 16, 2) => 0x00000002
__device__ __forceinline__ unsigned int bfe(unsigned int val, unsigned int pos,
                                            unsigned int len) {
  unsigned int ret;
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return ret;
}

// rotate_left(val, pos, len, cnt) returns a copy of `val` where
// `len` bits starting at `pos` are rotated to the left by `cnt`
// positions, e.g. rotate_left(0x01020304, 8, 4, 3) => 0x01020904
__device__ __forceinline__ unsigned int rotate_left(unsigned int val,
                                                    unsigned int pos,
                                                    unsigned int len,
                                                    unsigned int cnt) {
  unsigned int tmp = (val >> pos) << (32 - len);
#if __CUDA_ARCH__ >= 320
  tmp = (tmp >> (32 - len - cnt)) | __funnelshift_l(tmp, 0, cnt);
#else
  tmp = (tmp >> (32 - len - cnt)) | (tmp >> (32 - cnt));
#endif
  tmp = ((tmp << (32 - len)) >> (32 - len)) << pos;
  return tmp | (val & ~(((1 << len) - 1) << pos));
}

template <typename T>
__device__ __forceinline__ T cmp_swap(const T a, int mask, int dir) {
  T b = T::shuffle(a, mask);
  return T::compare(a, b) == dir ? b : a;
}

template <typename T>
__device__ __forceinline__ T warp_bitonic_split(T val, unsigned int tid) {
  val = cmp_swap(val, 0x01, bfe(tid, 1) ^ bfe(tid, 0));
  val = cmp_swap(val, 0x02, bfe(tid, 2) ^ bfe(tid, 1));
  val = cmp_swap(val, 0x01, bfe(tid, 2) ^ bfe(tid, 0));
  val = cmp_swap(val, 0x04, bfe(tid, 3) ^ bfe(tid, 2));
  val = cmp_swap(val, 0x02, bfe(tid, 3) ^ bfe(tid, 1));
  val = cmp_swap(val, 0x01, bfe(tid, 3) ^ bfe(tid, 0));
  val = cmp_swap(val, 0x08, bfe(tid, 4) ^ bfe(tid, 3));
  val = cmp_swap(val, 0x04, bfe(tid, 4) ^ bfe(tid, 2));
  val = cmp_swap(val, 0x02, bfe(tid, 4) ^ bfe(tid, 1));
  val = cmp_swap(val, 0x01, bfe(tid, 4) ^ bfe(tid, 0));
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_bitonic_merge(T val, unsigned int tid,
                                                unsigned int dir) {
  val = cmp_swap(val, 0x10, dir ^ bfe(tid, 4));
  val = cmp_swap(val, 0x08, dir ^ bfe(tid, 3));
  val = cmp_swap(val, 0x04, dir ^ bfe(tid, 2));
  val = cmp_swap(val, 0x02, dir ^ bfe(tid, 1));
  val = cmp_swap(val, 0x01, dir ^ bfe(tid, 0));
  return val;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T *data, unsigned int tid) {
  T val = warp_bitonic_split<T>(data[tid], tid);
  data[tid] = warp_bitonic_merge<T>(val, tid, bfe(tid, 5));
  __syncthreads();
}

template <typename T>
__device__ __forceinline__ void bitonic_merge(T *data, unsigned int tid,
                                              unsigned int dir) {
  data[tid] = warp_bitonic_merge<T>(data[tid], tid, dir);
  __syncthreads();
}

template <typename T, unsigned int N>
__device__ __forceinline__ void bitonic_merge(T *data, unsigned int tid,
                                              unsigned int dir) {
  unsigned int idx = rotate_left(tid, 4, N + 1, N);
  data[idx] = cmp_swap(data[idx], 0x10, dir ^ bfe(tid, 4));
  __syncthreads();
}
} // namespace details

template <typename T, unsigned N = 1024>
__global__ void bitonic_sort(T *data, const int size) {
  using namespace bitonic_sort_details;
  static_assert(N == 32 || N == 64 || N == 128 || N == 256 || N == 512 ||
                    N == 1024,
                "bitonic_sort supports only N=2^n with n=5..10");

  static __shared__ T shared[1024];
  const unsigned int tid = threadIdx.x;

  shared[tid] = tid < size & 1023 ? data[tid] : T::extrema();
  warp_bitonic_sort<T>(shared, tid);

  if (N > 32) {
    bitonic_merge<T, 1>(shared, tid, bfe(tid, 6));
    bitonic_merge<T>(shared, tid, bfe(tid, 6));
  }
  if (N > 64) {
    bitonic_merge<T, 2>(shared, tid, bfe(tid, 7));
    bitonic_merge<T, 1>(shared, tid, bfe(tid, 7));
    bitonic_merge<T>(shared, tid, bfe(tid, 7));
  }
  if (N > 128) {
    bitonic_merge<T, 3>(shared, tid, bfe(tid, 8));
    bitonic_merge<T, 2>(shared, tid, bfe(tid, 8));
    bitonic_merge<T, 1>(shared, tid, bfe(tid, 8));
    bitonic_merge<T>(shared, tid, bfe(tid, 8));
  }
  if (N > 256) {
    bitonic_merge<T, 4>(shared, tid, bfe(tid, 9));
    bitonic_merge<T, 3>(shared, tid, bfe(tid, 9));
    bitonic_merge<T, 2>(shared, tid, bfe(tid, 9));
    bitonic_merge<T, 1>(shared, tid, bfe(tid, 9));
    bitonic_merge<T>(shared, tid, bfe(tid, 9));
  }
  if (N > 512) {
    bitonic_merge<T, 5>(shared, tid, bfe(tid, 10));
    bitonic_merge<T, 4>(shared, tid, bfe(tid, 10));
    bitonic_merge<T, 3>(shared, tid, bfe(tid, 10));
    bitonic_merge<T, 2>(shared, tid, bfe(tid, 10));
    bitonic_merge<T, 1>(shared, tid, bfe(tid, 10));
    bitonic_merge<T>(shared, tid, bfe(tid, 10));
  }
  data[tid] = shared[tid];
}
}

#endif

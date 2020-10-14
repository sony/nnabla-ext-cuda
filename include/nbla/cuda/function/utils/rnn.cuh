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

#ifndef NBLA_CUDA_FUNCTION_UTILS_RNN_CUH
#define NBLA_CUDA_FUNCTION_UTILS_RNN_CUH

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/nd_index.cuh>

namespace nbla {
namespace cuda {
namespace function {
namespace utils {
namespace rnn {

template <typename U, bool accum = false>
__global__ void kernel_sequential_add(int size, const U *x, U *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    y[idx] = accum ? y[idx] + x[idx] : x[idx];
  }
}

// Parallelize over the unpacked array.
// Parallelizing over the packed array is slower since we have to iterate over
// the time dimension
// which is generally very long.
template <typename U, bool accum = false>
__global__ void kernel_pack_parallel_add0(int size, const U *padded_sequence,
                                          const int *batch_sizes,
                                          U *packed_sequence, int T, int B,
                                          int D) {
  auto stride0 = B * D;
  auto stride1 = D;
  NBLA_CUDA_KERNEL_LOOP(iidx, size) {
    auto nd_iidx = device_flat_to_3d(iidx, make_int2(stride0, stride1));
    auto t = nd_iidx.x;
    auto b = nd_iidx.y;
    auto d = nd_iidx.z;
    auto batch_size = batch_sizes[t];
    if (b < batch_size) {
      int s = 0;
      for (int i = 0; i < t; i++) {
        s += batch_sizes[i];
      }
      auto packed_sequence_oidx = packed_sequence + (s + b) * D + d;
      *packed_sequence_oidx =
          accum ? *packed_sequence_oidx + padded_sequence[iidx]
                : padded_sequence[iidx];
    }
  }
}

template <typename U, bool accum = false>
__global__ void kernel_unpack_parallel_add0(int size, const U *packed_sequence,
                                            const int *batch_sizes,
                                            U *padded_sequence, int T, int B,
                                            int D) {
  auto stride0 = B * D;
  auto stride1 = D;
  NBLA_CUDA_KERNEL_LOOP(iidx, size) {
    auto nd_iidx = device_flat_to_3d(iidx, make_int2(stride0, stride1));
    auto t = nd_iidx.x;
    auto b = nd_iidx.y;
    auto d = nd_iidx.z;
    auto batch_size = batch_sizes[t];
    if (b < batch_size) {
      int s = 0;
      for (int i = 0; i < t; i++) {
        s += batch_sizes[i];
      }
      auto packed_sequence_oidx = packed_sequence + (s + b) * D + d;
      padded_sequence[iidx] =
          accum ? padded_sequence[iidx] + *packed_sequence_oidx
                : *packed_sequence_oidx;
    }
  }
}

template <typename U, bool accum = false>
__global__ void kernel_zeros(int size, U *padded_sequence) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { padded_sequence[idx] = U(0); }
}

template <typename U, bool accum = false>
inline void pack(const Context &ctx, const U *padded_sequence,
                 const int *batch_sizes, U *packed_sequence, int T, int B,
                 int D, int N) {
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
  // > The following device operations are asynchronous with respect to the
  // host:
  // >   - Memory copies from host to device of a memory block of 64 KB or less;
  if (N > (64 * 1024 / sizeof(int))) {
    auto s = 0;
    for (int t = 0; t < T; t++) {
      auto batch_size = batch_sizes[t];
      auto isize = s * D;
      auto padded_sequence_t = padded_sequence + t * (B * D);
      auto packed_sequence_t = packed_sequence + isize;
      auto kernel = accum ? kernel_sequential_add<U, true>
                          : kernel_sequential_add<U, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, batch_size * D, padded_sequence_t,
                                     packed_sequence_t);
      s += batch_size;
    }
  } else {
    auto bytes = T * sizeof(int);
    shared_ptr<CudaCachedArray> arr_buff =
        make_shared<CudaCachedArray>(T, get_dtype<int>(), ctx);
    int *buff = arr_buff->pointer<int>();
    NBLA_CUDA_CHECK(
        cudaMemcpy(buff, batch_sizes, bytes, cudaMemcpyHostToDevice));

    auto kernel = accum ? kernel_pack_parallel_add0<U, true>
                        : kernel_pack_parallel_add0<U, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, T * B * D, padded_sequence, buff,
                                   packed_sequence, T, B, D);
  }
}

template <typename U, bool accum = false>
inline void unpack(const Context &ctx, const U *packed_sequence,
                   const int *batch_sizes, U *padded_sequence, int T, int B,
                   int D, int N, int TL = -1) {
  // Zeroing
  auto size = (TL > T) ? (TL * B * D) : (T * B * D);
  if (!accum)
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_zeros, size, padded_sequence);
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-host-device
  // > The following device operations are asynchronous with respect to the
  // host:
  // >   - Memory copies from host to device of a memory block of 64 KB or less;
  // Unapck
  if (N > (64 * 1024 / sizeof(int))) {
    auto s = 0;
    for (int t = 0; t < T; t++) {
      auto batch_size = batch_sizes[t];
      auto isize = s * D;
      auto padded_sequence_t = padded_sequence + t * (B * D);
      auto packed_sequence_t = packed_sequence + isize;
      auto kernel = accum ? kernel_sequential_add<U, true>
                          : kernel_sequential_add<U, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, batch_size * D, packed_sequence_t,
                                     padded_sequence_t);
      s += batch_size;
    }
  } else {
    auto bytes = T * sizeof(int);
    shared_ptr<CudaCachedArray> arr_buff =
        make_shared<CudaCachedArray>(T, get_dtype<int>(), ctx);
    auto buff = arr_buff->pointer<int>();
    NBLA_CUDA_CHECK(
        cudaMemcpy(buff, batch_sizes, bytes, cudaMemcpyHostToDevice));
    auto kernel = accum ? kernel_unpack_parallel_add0<U, true>
                        : kernel_unpack_parallel_add0<U, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, T * B * D, packed_sequence, buff,
                                   padded_sequence, T, B, D);
  }
}

} // rnn
} // utils
} // function
} // cuda
} // nbla
#endif
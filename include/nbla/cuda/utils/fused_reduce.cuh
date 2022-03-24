// Copyright 2022 Sony Group Corporation.
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

#ifndef __NBLA_CUDA_UTILS_FUSED_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_FUSED_REDUCE_CUH__
#include <nbla/cuda/half.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>

namespace nbla {

template <typename T, typename Scalar> struct Element {
  T *x;
  Scalar *n;
  __device__ float2 operator[](int i) const { return make_float2(x[i], n[i]); }
};

template <typename T> struct Id {
  __device__ T operator()(const T &val) const { return val; }
};

template <typename T> struct Square {
  __device__ T operator()(const T &x) const { return x * x; }
};
template <> struct Square<float2> {
  __device__ float2 operator()(const float2 &x) const {
    return make_float2(x.x * x.x, x.y * x.y);
  }
};

template <typename T> struct ReduceTarget {
  using ScalarType = typename CudaTypeForceFloat<T>::type;
  const T *input;
  ScalarType *buff;
  ScalarType *output;
};

namespace internal {

template <typename F>
__device__ void device_fused_reduce_per_block(int N, const F &f) {
  // Base case
}
template <typename F, typename T, typename... Ts>
__device__ void device_fused_reduce_per_block(int N, const F &f,
                                              ReduceTarget<T> target,
                                              ReduceTarget<Ts>... targets) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += f((AccT)target.input[i]); }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    target.output[blockIdx.x] = thread_data;
  }
  device_fused_reduce_per_block(N, f, targets...);
}
template <typename F, typename... Ts>
__global__ void kernel_fused_reduce_per_block(int N, const F &f,
                                              ReduceTarget<Ts>... targets) {
  device_fused_reduce_per_block(N, f, targets...);
}

template <typename F>
__device__ void device_fused_reduce_per_block_from_buff(int N, const F &f) {
  // Base case
}
template <typename F, typename T, typename... Ts>
__device__ void device_fused_reduce_per_block_from_buff(
    int N, const F &f, ReduceTarget<T> target, ReduceTarget<Ts>... targets) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += f((AccT)target.buff[i]); }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    target.output[blockIdx.x] = thread_data;
  }
  device_fused_reduce_per_block_from_buff(N, f, targets...);
}
template <typename F, typename... Ts>
__global__ void
kernel_fused_reduce_per_block_from_buff(int N, const F &f,
                                        ReduceTarget<Ts>... targets) {
  device_fused_reduce_per_block_from_buff(N, f, targets...);
}

template <typename F>
__device__ void device_fused_reduce_per_block_to_buff(int N, const F &f) {
  // Base case
}
template <typename F, typename T, typename... Ts>
__device__ void
device_fused_reduce_per_block_to_buff(int N, const F &f, ReduceTarget<T> target,
                                      ReduceTarget<Ts>... targets) {
  typedef typename CudaTypeForceFloat<T>::type AccT;
  AccT thread_data = 0;
  NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += f((AccT)target.input[i]); }
  thread_data = blockReduceSum(thread_data);
  if (threadIdx.x == 0) {
    target.buff[blockIdx.x] = thread_data;
  }
  device_fused_reduce_per_block_to_buff(N, f, targets...);
}
template <typename F, typename... Ts>
__global__ void
kernel_fused_reduce_per_block_to_buff(int N, const F &f,
                                      ReduceTarget<Ts>... targets) {
  device_fused_reduce_per_block_to_buff(N, f, targets...);
}
}

template <typename F, typename... T>
void fused_reduce(cudaStream_t stream, int num, ReduceTarget<T>... targets) {
  if (num >= 1024) {
    int blocks = min(NBLA_CUDA_GET_BLOCKS(num), /*max blocks*/ 1024);
    internal::kernel_fused_reduce_per_block_to_buff<
        F, T...><<<blocks, NBLA_CUDA_NUM_THREADS, 0, stream>>>(num, F(),
                                                               targets...);
    internal::kernel_fused_reduce_per_block_from_buff<
        Id<float>, T...><<<1, 1024, 0, stream>>>(blocks, Id<float>(),
                                                 targets...);
  } else {
    internal::kernel_fused_reduce_per_block<F, T...><<<1, 1024, 0, stream>>>(
        num, F(), targets...);
  }
}

namespace layer {
template <typename V = float, typename Scalar = float> struct Argument {
  size_t num;
  V input;
  Scalar *buff;
  Scalar *output;
};

template <typename Scalar> struct Zero {
  Scalar operator()() const { return Scalar(); }
};
template <> struct Zero<float2> {
  float2 operator()() const { return make_float2(0.0, 0.0); }
};

template <typename F, typename V, typename Scalar>
__global__ void
kernel_fused_reduce_per_block_to_buff(const F &f, Scalar zero, size_t num,
                                      Argument<V, Scalar> *args) {
  for (int i = 0; i < num; ++i) {
    const auto &arg = args[i];
    Scalar thread_data = zero;
    NBLA_CUDA_KERNEL_LOOP(i, arg.num) {
      thread_data += f((Scalar)arg.input[i]);
    }
    thread_data = blockReduceSum(thread_data);
    if (threadIdx.x == 0) {
      arg.buff[blockIdx.x] = thread_data;
    }
  }
}
template <typename F, typename V, typename Scalar>
__global__ void
kernel_fused_reduce_per_block_from_buff(int N, const F &f, Scalar zero,
                                        size_t num, Argument<V, Scalar> *args) {
  for (int i = 0; i < num; ++i) {
    const auto &arg = args[i];
    Scalar thread_data = zero;
    NBLA_CUDA_KERNEL_LOOP(i, N) { thread_data += f((Scalar)arg.buff[i]); }
    thread_data = blockReduceSum(thread_data);
    if (threadIdx.x == 0) {
      arg.output[blockIdx.x] = thread_data;
    }
  }
}
template <typename F, typename V, typename Scalar>
__global__ void kernel_fused_reduce_per_block(const F &f, Scalar zero,
                                              size_t num,
                                              Argument<V, Scalar> *args) {
  for (int i = 0; i < num; ++i) {
    const auto &arg = args[i];
    Scalar thread_data = zero;
    NBLA_CUDA_KERNEL_LOOP(i, arg.num) {
      thread_data += f((Scalar)arg.input[i]);
    }
    thread_data = blockReduceSum(thread_data);
    if (threadIdx.x == 0) {
      arg.output[blockIdx.x] = thread_data;
    }
  }
}

template <typename F, typename V, typename Scalar>
void fused_reduce(Context ctx, cudaStream_t stream,
                  const std::vector<Argument<V, Scalar>> &args) {
  size_t max_n = 0;
  for (const auto &arg : args) {
    max_n = std::max<size_t>(max_n, arg.num);
  }

  // Send argument to GPU
  dtypes dtype = get_dtype<float>();
  auto buff_arr =
      make_shared<NdArray>(Shape_t{sizeof(Argument<V, Scalar>) * args.size()});
  void *buff = buff_arr->cast(dtype, ctx, true)->template pointer<void>();
  NBLA_CUDA_CHECK(cudaMemcpyAsync(buff, args.data(),
                                  args.size() * sizeof(Argument<V, Scalar>),
                                  cudaMemcpyHostToDevice, stream));

  if (max_n >= 1024) {
    int blocks = min(NBLA_CUDA_GET_BLOCKS(max_n), /*max blocks*/ 1024);
    kernel_fused_reduce_per_block_to_buff<
        F, V, Scalar><<<blocks, NBLA_CUDA_NUM_THREADS, 0, stream>>>(
        F(), Zero<Scalar>()(), args.size(), (Argument<V, Scalar> *)buff);
    kernel_fused_reduce_per_block_from_buff<Id<Scalar>, V,
                                            Scalar><<<1, 1024, 0, stream>>>(
        blocks, Id<Scalar>(), Zero<Scalar>()(), args.size(),
        (Argument<V, Scalar> *)buff);
  } else {
    kernel_fused_reduce_per_block<F, V, Scalar><<<1, 1024, 0, stream>>>(
        F(), Zero<Scalar>()(), args.size(), (Argument<V, Scalar> *)buff);
  }
}
}
}
#endif
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

#ifndef NBLA_CUDA_FUNCTION_GENERIC_MIXED_PRECISION_TRAINING_CUH
#define NBLA_CUDA_FUNCTION_GENERIC_MIXED_PRECISION_TRAINING_CUH

#include <nbla/cuda/common.hpp>
#include <nbla/variable.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>

namespace nbla {

template <typename T> struct check_inf {
  __host__ __device__ bool operator()(const T a) const { return isinf(a); }
};

template <typename T> struct check_nan {
  __host__ __device__ bool operator()(const T a) const { return isnan(a); }
};

template <typename T> struct check_inf_or_nan {
  __host__ __device__ bool operator()(const T a) const {
    return isinf(a) || isnan(a);
  }
};

template <typename T>
__global__ void kernel_scale_grad_impl(const int num, float scale, T *grad) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] *= scale; }
}

template <typename T>
bool check_inf_grad_cuda(const Context &ctx, const shared_ptr<Variable> param) {
  cuda_set_device(std::stoi(ctx.device_id));
  Size_t size = param->size();
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  bool flag = false;
  thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(grad);
  flag = thrust::transform_reduce(dev_ptr, dev_ptr + size, check_inf<T>(), 0,
                                  thrust::plus<bool>());
  return flag;
}

template <typename T>
bool check_nan_grad_cuda(const Context &ctx, const shared_ptr<Variable> param) {
  cuda_set_device(std::stoi(ctx.device_id));
  Size_t size = param->size();
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  bool flag = false;
  thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(grad);
  flag = thrust::transform_reduce(dev_ptr, dev_ptr + size, check_nan<T>(), 0,
                                  thrust::plus<bool>());
  return flag;
}

template <typename T>
bool check_inf_or_nan_grad_cuda(const Context &ctx,
                                const shared_ptr<Variable> param) {
  cuda_set_device(std::stoi(ctx.device_id));
  Size_t size = param->size();
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  bool flag = false;
  thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(grad);
  flag = thrust::transform_reduce(
      dev_ptr, dev_ptr + size, check_inf_or_nan<T>(), 0, thrust::plus<bool>());
  return flag;
}

template <typename T>
void scale_grad_impl_cuda(const Context &ctx, const shared_ptr<Variable> param,
                          float scale) {
  cuda_set_device(std::stoi(ctx.device_id));
  Size_t size = param->size();
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_scale_grad_impl, size, scale, grad);
}
}
#endif

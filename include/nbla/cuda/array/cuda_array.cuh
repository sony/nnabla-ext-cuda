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

#ifndef __NBLA_CUDA_H_ARRAY_HPP__
#define __NBLA_CUDA_H_ARRAY_HPP__

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <nbla/cuda/half.hpp>

#include <vector>

namespace nbla {

using std::vector;

namespace {
template <typename T>
__global__ void kernel_fill(const int num, T *y, float value) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = (T)value; }
}
template <typename Ta, typename Tb>
__global__ void kernel_copy(const int num, Ta *y, const Tb *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { y[idx] = (Ta)x[idx]; }
}
}

template <typename T> void cuda_fill(Array *self, float value) {
  typedef typename CudaType<T>::type TT;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_fill, self->size(), self->pointer<TT>(),
                                 value);
}

template <typename Ta, typename Tb>
void thrust_copy(const Array *src, Array *dst) {
  typedef typename CudaType<Ta>::type type_a;
  typedef typename CudaType<Tb>::type type_b;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_copy, src->size(),
                                 dst->pointer<type_b>(),
                                 src->const_pointer<type_a>());
}

template <typename Ta, typename Tb>
__global__ void kernel_copy_half(int size, const Ta *src, Tb *dst) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { dst[idx] = src[idx]; }
}

template <typename Ta, typename Tb>
void cuda_array_copy(const Array *src, Array *dst) {
  int src_device = std::stoi(src->context().device_id);
  int dst_device = std::stoi(dst->context().device_id);
  if (src_device == dst_device) {
    // In-device copy.
    cuda_set_device(src_device);
    thrust_copy<Ta, Tb>(src, dst);
    return;
  }
  // Inter-device copy.
  std::unique_ptr<Array> src_tmp;

  // At first convert dtype on source device if necessary.
  if (src->dtype() != dst->dtype()) {
    cuda_set_device(src_device);
    // Create array with dest dtype at source device
    src_tmp.reset(
        new CudaCachedArray(src->size(), dst->dtype(), src->context()));
    thrust_copy<Ta, Tb>(src, src_tmp.get());
    src = src_tmp.get(); // Replace src pointer with the created memory with the
                         // same dtype
  }
  // Copy over devices.
  cuda_set_device(dst_device);
  NBLA_CUDA_CHECK(cudaMemcpyPeer(dst->pointer<void>(), dst_device,
                                 src->const_pointer<void>(), src_device,
                                 dst->size() * sizeof(Tb)));
}

NBLA_DEFINE_COPY_WRAPPER(cuda_array_copy);
NBLA_DISABLE_TYPE(cuda_array_copy, cuda_fill, long long);
NBLA_DISABLE_TYPE(cuda_array_copy, cuda_fill, long double);
NBLA_DISABLE_TYPE(cuda_array_copy, cuda_fill, bool);
NBLA_DEFINE_FUNC_COPY_FROM(CudaArray, cuda_array_copy, cuda);
NBLA_DEFINE_FUNC_FILL(CudaArray, cuda_fill, cuda);
}
#endif
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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <vector>

namespace nbla {

using std::vector;

template <typename T> void cuda_fill(Array *self, float value) {
  thrust::device_ptr<T> dev_ptr(self->pointer<T>());
  thrust::fill(dev_ptr, dev_ptr + self->size(), (T)value);
}

template <typename Ta, typename Tb>
void cuda_array_copy(const Array *src, Array *dst) {
  int src_device = std::stoi(src->context().device_id);
  int dst_device = std::stoi(dst->context().device_id);
  if (src_device == dst_device) {
    // In-device copy.
    cuda_set_device(src_device);
    thrust::device_ptr<const Ta> a(src->const_pointer<Ta>());
    thrust::device_ptr<Tb> b(dst->pointer<Tb>());
    thrust::copy(a, a + src->size(), b);
    return;
  }
  // Inter-devcie copy.
  std::unique_ptr<Array> src_tmp;

  // At first convert dtype on source device if necessary.
  if (src->dtype() != dst->dtype()) {
    cuda_set_device(src_device);
    // Create array with dest dtype at source device
    src_tmp.reset(
        new CudaCachedArray(src->size(), dst->dtype(), src->context()));
    thrust::device_ptr<const Ta> a(src->const_pointer<Ta>());
    thrust::device_ptr<Tb> b(src_tmp->pointer<Tb>());
    thrust::copy(a, a + src->size(), b);
    src = src_tmp.get(); // Replace src pointer with the created memory with the
                         // same dtype
  }
  // Copy over devices.
  cuda_set_device(dst_device);
  NBLA_CUDA_CHECK(cudaMemcpyPeer(dst->pointer<Tb>(), dst_device,
                                 src->const_pointer<Tb>(), src_device,
                                 dst->size() * sizeof(Tb)));
}

NBLA_DEFINE_TYPE_DISABLER(cuda);
NBLA_DISABLE_TYPE(cuda, long long);
NBLA_DISABLE_TYPE(cuda, unsigned long long);
NBLA_DISABLE_TYPE(cuda, long double);
NBLA_DISABLE_TYPE(cuda, bool);
NBLA_DEFINE_FUNC_COPY_FROM(CudaArray, cuda_array_copy, cuda);
NBLA_DEFINE_FUNC_FILL(CudaArray, cuda_fill, cuda);
}

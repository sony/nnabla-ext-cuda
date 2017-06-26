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

void CudaArray::fill(float value) {
  cuda_set_device(this->device_);
  switch (this->dtype()) {
    NBLA_CASE_ARRAY_FILL(cuda_fill, UBYTE, unsigned char);
    NBLA_CASE_ARRAY_FILL(cuda_fill, BYTE, char);
    NBLA_CASE_ARRAY_FILL(cuda_fill, USHORT, unsigned short);
    NBLA_CASE_ARRAY_FILL(cuda_fill, SHORT, short);
    NBLA_CASE_ARRAY_FILL(cuda_fill, UINT, unsigned int);
    NBLA_CASE_ARRAY_FILL(cuda_fill, INT, int);
    NBLA_CASE_ARRAY_FILL(cuda_fill, ULONG, unsigned long);
    NBLA_CASE_ARRAY_FILL(cuda_fill, LONG, long);
    // NBLA_CASE_ARRAY_FILL(cuda_fill, ULONGLONG, unsigned long long);
    // NBLA_CASE_ARRAY_FILL(cuda_fill, LONGLONG, long long);
    NBLA_CASE_ARRAY_FILL(cuda_fill, FLOAT, float);
    NBLA_CASE_ARRAY_FILL(cuda_fill, DOUBLE, double);
  // NBLA_CASE_ARRAY_FILL(cuda_fill, BOOL, bool);
  // NBLA_CASE_ARRAY_FILL(cuda_fill, LONGDOUBLE, long double);
  default:
    NBLA_ERROR(error_code::type, "Unknown dtype.")
  }
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

#define NBLA_CUDA_ARRAY_COPY_FROM(copy_func, type)                             \
  switch (this->dtype()) {                                                     \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UBYTE, type, unsigned char);       \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, BYTE, type, char);                 \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, USHORT, type, unsigned short);     \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, SHORT, type, short);               \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, UINT, type, unsigned int);         \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, INT, type, int);                   \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, ULONG, type, unsigned long);       \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, LONG, type, long);                 \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, FLOAT, type, float);               \
    NBLA_CASE_ARRAY_COPY_FROM_TO(copy_func, DOUBLE, type, double);             \
  default:                                                                     \
    NBLA_ERROR(error_code::unclassified, "Unknown dtype.");                    \
  }

#define NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, type, src_type)              \
  case dtypes::type:                                                           \
    NBLA_CUDA_ARRAY_COPY_FROM(copy_func, src_type);                            \
    break;

#define NBLA_DEFINE_FUNC_CUDA_COPY_FROM(array_class, copy_func)                \
  void array_class::copy_from(const Array *src_array) {                        \
    if (src_array->size() != this->size_) {                                    \
      NBLA_ERROR(error_code::unclassified, "Size mismatch.");                  \
    }                                                                          \
    switch (src_array->dtype()) {                                              \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, UBYTE, unsigned char);         \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, BYTE, char);                   \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, USHORT, unsigned short);       \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, SHORT, short);                 \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, UINT, unsigned int);           \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, INT, int);                     \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, ULONG, unsigned long);         \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, LONG, long);                   \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, FLOAT, float);                 \
      NBLA_CASE_CUDA_ARRAY_COPY_FROM(copy_func, DOUBLE, double);               \
    default:                                                                   \
      NBLA_ERROR(error_code::unclassified, "Unknown dtype.");                  \
    }                                                                          \
  }

NBLA_DEFINE_FUNC_CUDA_COPY_FROM(CudaArray, cuda_array_copy);
}

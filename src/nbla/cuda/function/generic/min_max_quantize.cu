// Copyright 2019,2020,2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/min_max_quantize.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void MinMaxQuantizeCuda<T>::setup_impl(const Variables &inputs,
                                       const Variables &outputs) {
  MinMaxQuantize<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
__global__ void kernel_nudge_range(const int size, float eps,
                                   const T *qr_min_data, T *qr_max_data) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    if (qr_max_data[idx] - qr_min_data[idx] < eps) {
      qr_max_data[idx] = qr_min_data[idx] + eps;
    }
  }
}

template <typename T>
__global__ void
kernel_nudge_qr_min_max(const int size, const T *qr_min_data,
                        const T *qr_max_data, const T *ql_min_data,
                        const T *ql_max_data, const T *scale_data,
                        T *qr_min_nudged_data, T *qr_max_nudged_data) {
  T zero_point_nudged = T(0.0);
  T zero_point_from_min = T(0.0);
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    zero_point_from_min = ql_min_data[idx] - qr_min_data[idx] / scale_data[idx];
    if (zero_point_from_min <= ql_min_data[idx]) {
      zero_point_nudged = ql_min_data[idx];
    } else if (zero_point_from_min >= ql_max_data[idx]) {
      zero_point_nudged = ql_max_data[idx];
    } else {
      zero_point_nudged = round(zero_point_from_min);
    }
    qr_min_nudged_data[idx] =
        (ql_min_data[idx] - zero_point_nudged) * scale_data[idx];
    qr_max_nudged_data[idx] =
        (ql_max_data[idx] - zero_point_nudged) * scale_data[idx];
  }
}

template <typename T>
void MinMaxQuantizeCuda<T>::nudge_range(Variable *qr_min, Variable *qr_max) {
  const Tcu *qr_min_data = qr_min->get_data_pointer<Tcu>(this->ctx_);
  Tcu *qr_max_data = qr_max->cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nudge_range<Tcu>, qr_min->size(),
                                 this->eps_, qr_min_data, qr_max_data);
}

template <typename T>
void MinMaxQuantizeCuda<T>::nudge_qr_min_max(Variable *qr_min, Variable *qr_max,
                                             Variable *ql_min, Variable *ql_max,
                                             Variable *scale,
                                             Variable *qr_min_nudged,
                                             Variable *qr_max_nudged) {
  const Tcu *qr_min_data = qr_min->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *qr_max_data = qr_max->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *ql_min_data = ql_min->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *ql_max_data = ql_max->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *scale_data = scale->get_data_pointer<Tcu>(this->ctx_);
  Tcu *qr_min_nudged_data =
      qr_min_nudged->cast_data_and_get_pointer<Tcu>(this->ctx_);
  Tcu *qr_max_nudged_data =
      qr_max_nudged->cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nudge_qr_min_max<Tcu>, qr_min->size(),
                                 qr_min_data, qr_max_data, ql_min_data,
                                 ql_max_data, scale_data, qr_min_nudged_data,
                                 qr_max_nudged_data);
}
}

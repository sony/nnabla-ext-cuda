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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/rand.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void RandCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  Rand<T>::setup_impl(inputs, outputs);
  output_data_for_recomp_.reshape(outputs[0]->shape(), true);
}

template <typename T>
void RandCuda<T>::setup_recompute_impl(const Variables &inputs,
                                       const Variables &outputs) {
  save_output_data_ = true;
}

template <typename T>
void RandCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  // In any type, this uses float32 type.
  typedef typename CudaTypeForceFloat<T>::type Tc;
  cuda_set_device(device_);
  curandGenerator_t &gen =
      this->seed_ == -1 ? SingletonManager::get<Cuda>()->curand_generator()
                        : curand_generator_;
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  curand_generate_rand<float>(gen, this->low_, this->high_, y,
                              outputs[0]->size());

  // Save output data for recomputation.
  if (save_output_data_) {
    save_output_data<Tc>(this->ctx_, outputs[0], output_data_for_recomp_);
  }
}

template <typename T>
void RandCuda<T>::recompute_impl(const Variables &inputs,
                                 const Variables &outputs) {
  // Restore output data of previous forward execution.
  typedef typename CudaTypeForceFloat<T>::type Tc;
  restore_output_data<Tc>(this->ctx_, output_data_for_recomp_, outputs[0]);
  save_output_data_ = false;
}

template <typename T>
void RandCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  // Pass
}
}

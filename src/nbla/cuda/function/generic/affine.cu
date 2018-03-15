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
#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/affine.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <class T>
void AffineCuda<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  // y = x * w.
  cuda_gemm<Tc>(device_, y, true, x, this->i_col_, this->i_row_, true, w,
                this->w_col_, this->w_row_, true, 1,
                0); // Note that arrays are row-major.
  if (inputs.size() == 3) {
    // With bias
    const Tc *b = inputs[2]->get_data_pointer<Tc>(this->ctx_);
    const Tc *ones =
        static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
            this->o_row_, get_dtype<Tc>(), this->ctx_));
    // y = 1s * b^T + y
    cuda_gemm<Tc>(device_, y, true, ones, this->o_row_, 1, false, b, 1,
                  this->o_col_, false, 1, 1);
  }
}

template <class T>
void AffineCuda<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  if (propagate_down[0]) {
    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);
    const Tc *w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
    // dx += dy * w^t
    cuda_gemm<Tc>(device_, dx, true, dy, this->o_col_, this->o_row_, true, w,
                  this->w_col_, this->w_row_, false, 1,
                  (accum[0] ? 1 : 0)); // Note that arrays are row-major.
  }
  if (propagate_down[1]) {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    Tc *dw = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[1]);
    // dw += x^t * dy;
    cuda_gemm<Tc>(device_, dw, true, x, this->i_col_, this->i_row_, false, dy,
                  this->o_col_, this->o_row_, true, 1,
                  (accum[1] ? 1 : 0)); // Note that arrays are row-major.
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    // With bias.
    Tc *db = inputs[2]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[2]);
    const Tc *ones =
        static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
            this->o_row_, get_dtype<Tc>(), this->ctx_));
    // db += dy^T * 1;
    cuda_gemv<Tc>(device_, db, dy, this->o_col_, this->o_row_, false, ones,
                  this->o_row_, 1, (accum[2] ? 1 : 0));
  }
}
}

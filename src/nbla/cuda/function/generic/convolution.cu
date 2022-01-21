// Copyright 2018,2019,2020,2021 Sony Corporation.
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

// convolution.cu

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/convolution.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/col2im.hpp>
#include <nbla/cuda/utils/im2col.hpp>

#include <algorithm>

namespace nbla {

template <typename T>
void ConvolutionCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  Convolution<T>::setup_impl(inputs, outputs);
}

template <class T>
void ConvolutionCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  NBLA_CHECK(!this->channel_last_, error_code::value,
             "The passed argument channel_last_=true is not supported in CUDA "
             "Convolution.");

  cuda_set_device(std::stoi(this->ctx_.device_id));
  // Getting variable pointers
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  Variable *vcol = &this->col_;
  Tc *col = vcol->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  const Tc *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<Tc>(this->ctx_);
  }
  // Sample loop
  for (int n = 0; n < this->outer_size_; ++n) {
    // Im2col
    if (this->spatial_dims_ == 2) {
      im2col_cuda<Tc>(x + n * this->inner_size_i_, this->channels_i_,
                      this->spatial_shape_i_.data(), this->kernel_.data(),
                      this->pad_.data(), this->stride_.data(),
                      this->dilation_.data(), col);
    } else {
      im2col_nd_cuda<Tc>(x + n * this->inner_size_i_, this->channels_i_,
                         this->spatial_dims_, this->spatial_shape_i_.data(),
                         this->kernel_.data(), this->pad_.data(),
                         this->stride_.data(), this->dilation_.data(), col);
    }
    // Convolution by matrix multiplication
    Tc *y_n = y + n * this->inner_size_o_;
    for (int g = 0; g < this->group_; ++g) {
      // y = x * w
      cuda_gemm<Tc>(device_, y_n + g * this->row_y_ * this->col_y_, false,
                    col + g * this->row_col_ * this->col_col_, this->col_col_,
                    this->row_col_, false, w + g * this->row_w_ * this->col_w_,
                    this->col_w_, this->row_w_, false, 1, 0);
    }
    // Adding bias
    if (inputs.size() == 3) {
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              this->col_y_, get_dtype<Tc>(), this->ctx_));
      // y = 1s * b^T + y
      cuda_gemm<Tc>(device_, y_n, false, ones, 1, this->col_y_, true, b,
                    this->channels_o_, 1, true, 1, 1);
    }
  }
}

template <class T>
void ConvolutionCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }

  NBLA_CHECK(!this->channel_last_, error_code::value,
             "The passed argument channel_last_=true is not supported in CUDA "
             "Convolution.");

  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *x;
  const Tc *w;
  Tc *dx, *dw, *db, *col;
  Variable *temp_col = &this->col_;
  if (propagate_down[0] || propagate_down[1]) {
    col = temp_col->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  }
  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero();
    w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
    dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();
    x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      inputs[2]->grad()->zero();
    db = inputs[2]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
  }
  // Sample loop
  for (int n = 0; n < this->outer_size_; ++n) {
    const Tc *dy_n = dy + n * this->inner_size_o_;
    if (propagate_down[0]) {
      // Backprop to image
      Tc *dx_n = dx + n * this->inner_size_i_;
      for (int g = 0; g < this->group_; ++g) {
        // dx = w^T * dy
        cuda_gemm<Tc>(device_, col + this->row_col_ * this->col_col_ * g, true,
                      w + this->row_w_ * this->col_w_ * g, this->col_w_,
                      this->row_w_, false,
                      dy_n + this->row_y_ * this->col_y_ * g, this->col_y_,
                      this->row_y_, true, 1, 0);
      }
      // col2im
      if (this->spatial_dims_ == 2) {
        col2im_cuda<Tc>(col, this->channels_i_, this->spatial_shape_i_.data(),
                        this->kernel_.data(), this->pad_.data(),
                        this->stride_.data(), this->dilation_.data(), dx_n);
      } else {
        col2im_nd_cuda<Tc>(col, this->channels_i_, this->spatial_dims_,
                           this->spatial_shape_i_.data(), this->kernel_.data(),
                           this->pad_.data(), this->stride_.data(),
                           this->dilation_.data(), dx_n);
      }
    }
    if (propagate_down[1]) {
      // Backprop to weights
      // im2col
      if (this->spatial_dims_ == 2) {
        im2col_cuda<Tc>(x + n * this->inner_size_i_, this->channels_i_,
                        this->spatial_shape_i_.data(), this->kernel_.data(),
                        this->pad_.data(), this->stride_.data(),
                        this->dilation_.data(), col);
      } else {
        im2col_nd_cuda<Tc>(x + n * this->inner_size_i_, this->channels_i_,
                           this->spatial_dims_, this->spatial_shape_i_.data(),
                           this->kernel_.data(), this->pad_.data(),
                           this->stride_.data(), this->dilation_.data(), col);
      }
      // Weight convolution by matrix multiplication
      for (int g = 0; g < this->group_; ++g) {
        // dw += dy * col^T
        cuda_gemm<Tc>(device_, dw + g * this->row_w_ * this->col_w_, true,
                      dy_n + g * this->row_y_ * this->col_y_, this->col_y_,
                      this->row_y_, true,
                      col + g * this->row_col_ * this->col_col_, this->col_col_,
                      this->row_col_, false, 1, 1);
      }
    }
    if (inputs.size() == 3 && propagate_down[2]) {
      // Backprop to bias
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              this->col_y_, get_dtype<Tc>(), this->ctx_));
      cuda_gemv<Tc>(device_, db, dy_n, this->col_y_, this->channels_o_, true,
                    ones, this->col_y_, 1, 1);
    }
  }
}
}

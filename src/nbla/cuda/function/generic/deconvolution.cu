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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/deconvolution.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/col2im.hpp>
#include <nbla/cuda/utils/im2col.hpp>

namespace nbla {

template <typename T>
void DeconvolutionCuda<T>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
  Deconvolution<T>::setup_impl(inputs, outputs);
}

template <typename T>
void DeconvolutionCuda<T>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  // Getting variable pointers
  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *y = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  const Tc *w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
  CudaCachedArray col_array(this->row_col_ * this->col_col_ * this->group_,
                            get_dtype<Tc>(), this->ctx_);
  Tc *col = col_array.pointer<Tc>();
  outputs[0]->data()->zero();
  Tc *x = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);
  const Tc *b;
  if (inputs.size() == 3) {
    b = inputs[2]->get_data_pointer<Tc>(this->ctx_);
  }

  // Sample loop
  for (int n = 0; n < this->outer_size_; ++n) {

    // matrix multiplication
    const Tc *y_n = y + n * this->inner_size_o_;
    for (int g = 0; g < this->group_; ++g) {
      cuda_gemm<Tc>(device_, col + this->row_col_ * this->col_col_ * g, true,
                    w + this->row_w_ * this->col_w_ * g, this->col_w_,
                    this->row_w_, false, y_n + this->row_y_ * this->col_y_ * g,
                    this->col_y_, this->row_y_, true, 1, 0);
    }

    // col2im for w * x
    Tc *x_n = x + n * this->inner_size_i_;
    if (this->spatial_dims_ == 2) {
      col2im_cuda<Tc>(col, this->channels_i_, this->spatial_shape_i_.data(),
                      this->kernel_.data(), this->pad_.data(),
                      this->stride_.data(), this->dilation_.data(), x_n);
    } else {
      col2im_nd_cuda<Tc>(col, this->channels_i_, this->spatial_dims_,
                         this->spatial_shape_i_.data(), this->kernel_.data(),
                         this->pad_.data(), this->stride_.data(),
                         this->dilation_.data(), x_n);
    }

    // adding bias
    if (inputs.size() == 3) {
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              this->inner_size_i_ / this->channels_i_, get_dtype<Tc>(),
              this->ctx_));
      // y = 1s * b^T + y
      cuda_gemm<Tc>(device_, x_n, false, ones, 1,
                    this->inner_size_i_ / this->channels_i_, true, b,
                    this->channels_i_, 1, true, 1, 1);
    }
  }
}

template <typename T>
void DeconvolutionCuda<T>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));
  const Tc *dx = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  const Tc *y;
  const Tc *w;
  Tc *dy, *dw, *db, *col;
  shared_ptr<CudaCachedArray> col_array;

  if (propagate_down[0] || propagate_down[1]) {
    col_array = make_shared<CudaCachedArray>(this->row_col_ * this->col_col_ *
                                                 this->group_,
                                             get_dtype<Tc>(), this->ctx_);
    col = col_array->pointer<Tc>();
  }
  if (propagate_down[0]) {
    w = inputs[1]->get_data_pointer<Tc>(this->ctx_);
    dy = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }
  if (propagate_down[1]) {
    if (!accum[1])
      inputs[1]->grad()->zero();
    y = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    dw = inputs[1]->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      inputs[2]->grad()->zero();
    db = inputs[2]->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }

  // Sample loop
  for (int n = 0; n < this->outer_size_; ++n) {
    const Tc *dx_n = dx + n * this->inner_size_i_;

    if (propagate_down[0] || propagate_down[1]) {
      // im2col
      if (this->spatial_dims_ == 2) {
        im2col_cuda<Tc>(dx_n, this->channels_i_, this->spatial_shape_i_.data(),
                        this->kernel_.data(), this->pad_.data(),
                        this->stride_.data(), this->dilation_.data(), col);
      } else {
        im2col_nd_cuda<Tc>(dx_n, this->channels_i_, this->spatial_dims_,
                           this->spatial_shape_i_.data(), this->kernel_.data(),
                           this->pad_.data(), this->stride_.data(),
                           this->dilation_.data(), col);
      }
    }

    if (propagate_down[0]) {
      // Backprop to image
      Tc *dy_n = dy + n * this->inner_size_o_;
      for (int g = 0; g < this->group_; ++g) {
        // y = x * w
        cuda_gemm<Tc>(device_, dy_n + g * this->row_y_ * this->col_y_, false,
                      col + g * this->row_col_ * this->col_col_, this->col_col_,
                      this->row_col_, false,
                      w + g * this->row_w_ * this->col_w_, this->col_w_,
                      this->row_w_, false, 1, (accum[0] ? 1 : 0));
      }
    }

    if (propagate_down[1]) {
      // Backprop to weights
      const Tc *y_n = y + n * this->inner_size_o_;
      for (int g = 0; g < this->group_; ++g) {
        cuda_gemm<Tc>(device_, dw + g * this->row_w_ * this->col_w_, true,
                      y_n + g * this->row_y_ * this->col_y_, this->col_y_,
                      this->row_y_, true,
                      col + g * this->row_col_ * this->col_col_, this->col_col_,
                      this->row_col_, false, 1, 1);
      }
    }

    if (inputs.size() == 3 && propagate_down[2]) {
      // Backprop to bias
      const int spatial_size = this->inner_size_i_ / this->channels_i_;
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              spatial_size, get_dtype<Tc>(), this->ctx_));
      cuda_gemv<Tc>(device_, db, dx_n, spatial_size, this->channels_i_, true,
                    ones, spatial_size, 1, 1);
    }
  }
}
}

// Copyright 2020,2021 Sony Corporation.
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
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/affine_grid.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_affine_grid_add(const int size, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] += x[idx]; }
}

template <typename T>
void AffineGridCudaCudnn<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  AffineGridCuda<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();
  if (!affine_grid::cudnn_condition(this->size_.size(), this->align_corners_)) {
    return;
  }
  int B = oshape[0];
  int H = oshape[1];
  int W = oshape[2];
  int V = oshape[3];
  vector<int> dimA{
      B, 1, H,
      W}; // cudnn requires the redandant argument, C. This is for free.
  NBLA_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(
      theta_desc_, CUDNN_SAMPLER_BILINEAR, cudnn_data_type<T>::type(), 4,
      dimA.data()));
}

template <typename T>
void AffineGridCudaCudnn<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();
  if (!affine_grid::cudnn_condition(this->size_.size(), this->align_corners_)) {
    AffineGridCuda<T>::forward_impl(inputs, outputs);
    return;
  }

  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);
  const auto theta = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto grid = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  NBLA_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(cudnn_handle, theta_desc_,
                                                      theta, grid));
}

template <typename T>
void AffineGridCudaCudnn<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();
  if (!affine_grid::cudnn_condition(this->size_.size(), this->align_corners_)) {
    AffineGridCuda<T>::backward_impl(inputs, outputs, propagate_down, accum);
    return;
  }

  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);
  const auto g_grid = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  if (propagate_down[0]) {
    if (accum[0]) {
      auto g_theta_tmp = make_shared<Variable>(inputs[0]->shape());
      auto g_theta_tmp_ptr0 =
          g_theta_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto size = inputs[0]->size();
      auto g_theta_tmp_ptr1 = g_theta_tmp->get_grad_pointer<Tcu>(this->ctx_);
      auto g_theta =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
      NBLA_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
          cudnn_handle, theta_desc_, g_grid, g_theta_tmp_ptr0));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_affine_grid_add, size, g_theta,
                                     g_theta_tmp_ptr1);
    } else {
      auto g_theta =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      NBLA_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(
          cudnn_handle, theta_desc_, g_grid, g_theta));
    }
  }
}
} // namespace nbla

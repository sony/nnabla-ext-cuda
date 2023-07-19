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
#include <nbla/cuda/cudnn/function/warp_by_grid.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
__global__ void kernel_warp_by_grid_add(const int size, T *y, const T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] += x[idx]; }
}

template <typename T>
void WarpByGridCudaCudnn<T>::setup_impl(const Variables &inputs,
                                        const Variables &outputs) {
  WarpByGridCuda<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();
  if (!warp_by_grid::cudnn_condition(
          outputs[0]->shape().size(), this->mode_, this->padding_mode_t_,
          this->align_corners_, this->channel_last_)) {
    return;
  }
  int B = oshape[0];
  int C = oshape[1];
  int Ho = oshape[2];
  int Wo = oshape[3];
  vector<int> dimA{B, C, Ho, Wo};
  NBLA_CUDNN_CHECK(cudnnSetSpatialTransformerNdDescriptor(
      spatial_tf_desc_, CUDNN_SAMPLER_BILINEAR, cudnn_data_type<T>::type(), 4,
      dimA.data()));
  auto ishape = inputs[0]->shape();
  int Hi = ishape[2];
  int Wi = ishape[3];
  vector<int> dimX{B, C, Hi, Wi};
  cudnn_set_tensor_nd_descriptor_force_dim(x_desc_, cudnn_data_type<T>::type(),
                                           dimX, dimX.size(),
                                           this->channel_last_, false);
  vector<int> dimY{B, C, Ho, Wo};
  cudnn_set_tensor_nd_descriptor_force_dim(y_desc_, cudnn_data_type<T>::type(),
                                           dimY, dimY.size(),
                                           this->channel_last_, false);
}

template <typename T>
void WarpByGridCudaCudnn<T>::forward_impl(const Variables &inputs,
                                          const Variables &outputs) {
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();
  if (!warp_by_grid::cudnn_condition(
          outputs[0]->shape().size(), this->mode_, this->padding_mode_t_,
          this->align_corners_, this->channel_last_)) {
    WarpByGridCuda<T>::forward_impl(inputs, outputs);
    return;
  }

  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);
  auto x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerForward(cudnn_handle, spatial_tf_desc_,
                                                &alpha, x_desc_, x, grid, &beta,
                                                y_desc_, y));
}

template <typename T>
void WarpByGridCudaCudnn<T>::backward_impl(const Variables &inputs,
                                           const Variables &outputs,
                                           const vector<bool> &propagate_down,
                                           const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  auto oshape = outputs[0]->shape();

#if defined(_WIN32) || defined(_WIN64)
  WarpByGridCuda<T>::backward_impl(inputs, outputs, propagate_down, accum);
  return;
#endif
  if (!warp_by_grid::cudnn_condition(
          outputs[0]->shape().size(), this->mode_, this->padding_mode_t_,
          this->align_corners_, this->channel_last_)) {
    WarpByGridCuda<T>::backward_impl(inputs, outputs, propagate_down, accum);
    return;
  }

  cudnnHandle_t cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);
  const auto x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const auto y = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  auto g_alpha = get_cudnn_scalar_arg<T>(1);
  auto g_beta = get_cudnn_scalar_arg<T>(0);
  auto x_size = inputs[0]->size();
  auto g_size = inputs[1]->size();

  // beta 1 seems not working
  if (propagate_down[0] && propagate_down[1]) {
    if (!accum[0] && !accum[1]) {
      auto g_x =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
      auto g_grid =
          inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x, &g_alpha, y_desc_, g_y, grid, &g_beta, g_grid));
    } else if (!accum[0] && accum[1]) {
      auto g_x =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
      auto g_grid =
          inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
      auto g_grid_tmp = make_shared<Variable>(inputs[1]->shape());
      auto g_grid_tmp_ptr0 =
          g_grid_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_grid_tmp_ptr1 = g_grid_tmp->get_grad_pointer<Tcu>(this->ctx_);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x, &g_alpha, y_desc_, g_y, grid, &g_beta, g_grid_tmp_ptr0));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, g_size, g_grid,
                                     g_grid_tmp_ptr1);
    } else if (accum[0] && !accum[1]) {
      auto g_x =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
      auto g_grid =
          inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
      auto g_x_tmp = make_shared<Variable>(inputs[0]->shape());
      auto g_x_tmp_ptr0 =
          g_x_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_x_tmp_ptr1 = g_x_tmp->get_grad_pointer<Tcu>(this->ctx_);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x_tmp_ptr0, &g_alpha, y_desc_, g_y, grid, &g_beta, g_grid));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, x_size, g_x,
                                     g_x_tmp_ptr1);
    } else if (accum[0] && accum[1]) {
      auto g_x =
          inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
      auto g_grid =
          inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
      auto g_x_tmp = make_shared<Variable>(inputs[0]->shape());
      auto g_x_tmp_ptr0 =
          g_x_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_x_tmp_ptr1 = g_x_tmp->get_grad_pointer<Tcu>(this->ctx_);
      auto g_grid_tmp = make_shared<Variable>(inputs[1]->shape());
      auto g_grid_tmp_ptr0 =
          g_grid_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_grid_tmp_ptr1 = g_grid_tmp->get_grad_pointer<Tcu>(this->ctx_);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x_tmp_ptr0, &g_alpha, y_desc_, g_y, grid, &g_beta,
          g_grid_tmp_ptr0));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, g_size, g_grid,
                                     g_grid_tmp_ptr1);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, x_size, g_x,
                                     g_x_tmp_ptr1);
    }
  } else if (propagate_down[0] && !propagate_down[1]) {
    auto g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
    auto g_grid_tmp = make_shared<Variable>(inputs[1]->shape());
    auto g_grid_tmp_ptr0 =
        g_grid_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
    if (!accum[0]) {
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x, &g_alpha, y_desc_, g_y, grid, &g_beta, g_grid_tmp_ptr0));
    } else {
      auto g_x_tmp = make_shared<Variable>(inputs[0]->shape());
      auto g_x_tmp_ptr0 =
          g_x_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_x_tmp_ptr1 = g_x_tmp->get_grad_pointer<Tcu>(this->ctx_);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x_tmp_ptr0, &g_alpha, y_desc_, g_y, grid, &g_beta,
          g_grid_tmp_ptr0));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, x_size, g_x,
                                     g_x_tmp_ptr1);
    }

  } else if (!propagate_down[0] && propagate_down[1]) {
    auto g_x_tmp = make_shared<Variable>(inputs[0]->shape());
    auto g_x_tmp_ptr0 =
        g_x_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
    auto g_grid =
        inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
    if (!accum[1]) {
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x_tmp_ptr0, &g_alpha, y_desc_, g_y, grid, &g_beta, g_grid));
    } else {
      auto g_grid_tmp = make_shared<Variable>(inputs[0]->shape());
      auto g_grid_tmp_ptr0 =
          g_grid_tmp->cast_grad_and_get_pointer<Tcu>(this->ctx_, true);
      auto g_grid_tmp_ptr1 = g_grid_tmp->get_grad_pointer<Tcu>(this->ctx_);
      NBLA_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
          cudnn_handle, spatial_tf_desc_, &alpha, x_desc_, x, &beta, x_desc_,
          g_x_tmp_ptr0, &g_alpha, y_desc_, g_y, grid, &g_beta,
          g_grid_tmp_ptr0));
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_warp_by_grid_add, g_size, g_grid,
                                     g_grid_tmp_ptr1);
    }
  }
}
} // namespace nbla

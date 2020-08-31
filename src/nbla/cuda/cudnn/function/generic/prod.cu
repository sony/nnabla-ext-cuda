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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/prod.hpp>
#include <nbla/variable.hpp>

#if CUDNN_VERSION >= 6000
namespace nbla {

template <typename T>
void ProdCudaCudnn<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  ProdCuda<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  NBLA_CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
      this->reduce_desc_, CUDNN_REDUCE_TENSOR_MUL, CUDNN_DATA_FLOAT,
      CUDNN_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_8BIT_INDICES));

  std::vector<int> x_shape, y_shape;
  x_shape.reserve(CUDNN_DIM_MAX);
  y_shape.reserve(CUDNN_DIM_MAX);

  for (auto dim : inputs[0]->shape()) {
    x_shape.push_back(static_cast<int>(dim));
    y_shape.push_back(static_cast<int>(dim));
  }
  for (auto axis : this->axes_)
    y_shape.at(axis) = 1;

  this->same_in_out_shape_ = (y_shape == x_shape) ? true : false;

  if (!this->same_in_out_shape_) {
    cudnn_set_tensor_descriptor<T>(this->x_desc_, x_shape);
    cudnn_set_tensor_descriptor<T>(this->y_desc_, y_shape);

    auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
    auto cudnn_handle = cudnn_handle_manager->handle(this->device_);
    NBLA_CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        cudnn_handle, this->reduce_desc_, this->x_desc_, this->y_desc_,
        &this->workspace_size_));
  }
}

template <typename T>
void ProdCudaCudnn<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  if ((!this->f_transpose_) || (inputs[0]->shape().size() > CUDNN_DIM_MAX)) {
    ProdCuda<T>::forward_impl(inputs, outputs);
    return;
  }
  if (this->same_in_out_shape_) {
    const Array *x = inputs[0]->data()->get(get_dtype<Tcu>(), this->ctx_);
    Array *y = outputs[0]->data()->cast(get_dtype<Tcu>(), this->ctx_, true);
    y->copy_from(x);
    return;
  }
  cuda_set_device(this->device_);
  auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
  auto cudnn_handle = cudnn_handle_manager->handle(this->device_);

  unique_ptr<CudaCachedArray> workspace_arr;
  void *workspace{nullptr};
  if (this->workspace_size_) {
    workspace_arr.reset(
        new CudaCachedArray(this->workspace_size_, dtypes::BYTE, this->ctx_));
    workspace = workspace_arr->pointer<void>();
  }

  auto x_data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto y_data = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  float alpha = 1.0f, beta = 0.0f;

  NBLA_CUDNN_CHECK(cudnnReduceTensor(cudnn_handle, this->reduce_desc_, nullptr,
                                     0UL, workspace, this->workspace_size_,
                                     &alpha, this->x_desc_, x_data, &beta,
                                     this->y_desc_, y_data));
}

} // namespace nbla
#endif // CUDNN_VERSION >= 6000

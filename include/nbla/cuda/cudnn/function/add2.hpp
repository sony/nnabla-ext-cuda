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

#ifndef __NBLA_CUDA_CUDNN_FUNCTION_ADD2_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_ADD2_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/add2.hpp>

namespace nbla {

/** @copydoc Add2
*/
template <typename T> class Add2CudaCudnn : public Add2Cuda<T> {
public:
  typedef typename CudaType<T>::type Tw;

  explicit Add2CudaCudnn(const Context &ctx, bool inplace)
      : Add2Cuda<T>(ctx, inplace), device_(std::stoi(ctx.device_id)) {
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
  }
  virtual ~Add2CudaCudnn() {
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
  }
  virtual string name() { return "Add2CudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

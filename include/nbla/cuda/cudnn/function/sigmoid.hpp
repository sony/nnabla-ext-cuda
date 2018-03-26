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

/** Sigmoid
 */
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_SIGMOID_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_SIGMOID_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/function/sigmoid.hpp>

namespace nbla {

using std::string;

template <typename T> class SigmoidCudaCudnn : public Sigmoid<T> {
protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc_;
  cudnnActivationDescriptor_t activation_desc_;

public:
  typedef typename CudaType<T>::type Tw;

  SigmoidCudaCudnn(const Context &ctx)
      : Sigmoid<T>(ctx), device_(std::stoi(ctx.device_id)) {
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
    NBLA_CUDNN_CHECK(cudnnSetActivationDescriptor(
        activation_desc_, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, T(0)));
  }
  virtual ~SigmoidCudaCudnn() {}
  virtual string name() { return "SigmoidCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

#ifndef __NBLA_CUDA_CUDNN_FUNCTION_RELU_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_RELU_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/relu.hpp>

namespace nbla {

// This implementation can be activated when the relu in
// build-tools/code_generator/function_types.yaml is activated.
// See also src/nbla/cuda/cudnn/function/generic/relu.cu
//
// ReLUCudaCudnn requres inputs[0] (x) and outputs[0] (y), but ReLUCuda only
// requires y. Therefore ReLUCuda has an advantage of memory usage over
// ReLUCudaCudnn. That is why ReLUCudaCudnn was removed.
#if 0
/** @copydoc ReLU
*/
template <typename T> class ReLUCudaCudnn : public ReLUCuda<T> {
public:
  typedef typename CudaType<T>::type Tw;

  explicit ReLUCudaCudnn(const Context &ctx, bool inplace)
      : ReLUCuda<T>(ctx, inplace), device_(std::stoi(ctx.device_id)) {
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc_));
    NBLA_CUDNN_CHECK(cudnnSetActivationDescriptor(
        activation_desc_, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, T(0)));
  }
  virtual ~ReLUCudaCudnn() {
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
  }
  virtual string name() { return "ReLUCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnActivationDescriptor_t activation_desc_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
#endif
} // namespace nbla
#endif

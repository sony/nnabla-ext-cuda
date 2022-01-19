// Copyright 2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_INSTANCE_NORMALIZATION_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_INSTANCE_NORMALIZATION_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <nbla/cuda/function/instance_normalization.hpp>

namespace nbla {

// Currently, original CUDA C implementation is faster than cuDNN.
// However, the cuDNN implementation is left here in terms of future
// optimization of cuDNN.
// We can switch both implementations by `IN_USE_CUDNN` (0: not use cuDNN, 1:
// use cuDNN).
#define IN_USE_CUDNN 0

template <typename T>
class InstanceNormalizationCudaCudnn : public InstanceNormalizationCuda<T> {
public:
  typedef typename CudaType<T>::type Tc;

  explicit InstanceNormalizationCudaCudnn(const Context &ctx, int channel_axis,
                                          const vector<int> &batch_axis,
                                          float eps, bool no_scale,
                                          bool no_bias)
      : InstanceNormalizationCuda<T>(ctx, channel_axis, batch_axis, eps,
                                     no_scale, no_bias),
        device_(std::stoi(ctx.device_id)) {
#if IN_USE_CUDNN
#if CUDNN_VERSION < 5000
    std::cout << "Falling back to InstanceNormalizationCuda since BN does not "
                 "exist in CUDNN_VERSION < 5000."
              << std::endl; // TODO: warn.
    this->fall_back_func_.reset(new InstanceNormalizationCuda<T>(
        ctx, channel_axis, batch_axis, eps, no_scale, no_bias));
#else
    NBLA_CHECK(eps >= (float)CUDNN_BN_MIN_EPSILON, error_code::value,
               "eps must be greater than or equal to CUDNN_BN_MIN_EPSILON. "
               "eps=%g, CUDNN_BN_MIN_EPSILON=%g",
               eps, CUDNN_BN_MIN_EPSILON);
#endif
#else
    // Currently, the CUDA C implementation is faster than one using cuDNN
    // BatchNormalization.
    this->fall_back_func_ = make_shared<InstanceNormalizationCuda<T>>(
        ctx, channel_axis, batch_axis, eps, no_scale, no_bias);
#endif
  }
  virtual ~InstanceNormalizationCudaCudnn() {}
  virtual string name() { return "InstanceNormalizationCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;

#if IN_USE_CUDNN
#if CUDNN_VERSION < 5000
// No need to define members since InstanceNormalizationCudaCudnn must be fallen
// back into InstanceNormalizationCuda in this cuDNN version.
#else
  // Members for cuDNN implementation
  Variable mean_, var_;
  Variable beta_dummy_, gamma_dummy_;
  bool channel_last_;
  int b_idx_, g_idx_;
  Size_t reduction_size_, outer_size_;

  cudnnHandle_t cudnn_handle_;
  CudnnTensorDescriptor input_desc_, output_desc_;
  CudnnTensorDescriptor bn_scale_bias_mean_var_desc_;
  cudnnDataType_t derived_bn_dtype_;
  cudnnBatchNormMode_t mode_;
#endif
#endif

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_channel_first(const Variables &inputs,
                                     const Variables &outputs);
  virtual void backward_channel_first(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
};
}
#endif

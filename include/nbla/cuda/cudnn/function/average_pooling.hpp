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

/** AveragePooling
*/
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_AVERAGEPOOLING_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_AVERAGEPOOLING_HPP__

#include <nbla/function/average_pooling.hpp>
#include <nbla/function_registry.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <algorithm>

namespace nbla {

template <typename T> class AveragePoolingCudaCudnn : public AveragePooling<T> {
protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnPoolingDescriptor_t pooling_desc_;
  cudnnPoolingMode_t mode_;

public:
  AveragePoolingCudaCudnn(const Context &ctx, const vector<int> &kernel,
                          const vector<int> &stride, bool ignore_border,
                          const vector<int> &pad, bool including_pad)
      : AveragePooling<T>(ctx, kernel, stride, ignore_border, pad,
                          including_pad),
        device_(std::stoi(ctx.device_id)) {
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc_));
    NBLA_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
    if (this->including_pad_) {
      mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else {
      mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }
    int h = this->kernel_[0];
    int w = this->kernel_[1];
    int pad_h = this->pad_[0];
    int pad_w = this->pad_[1];
    int stride_h = this->stride_[0];
    int stride_w = this->stride_[1];
#if CUDNN_VERSION >= 5000
    NBLA_CUDNN_CHECK(
        cudnnSetPooling2dDescriptor(pooling_desc_, mode_, CUDNN_PROPAGATE_NAN,
                                    h, w, pad_h, pad_w, stride_h, stride_w));
#else
    NBLA_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pooling_desc_, mode_, h, w, pad_h, pad_w, stride_h, stride_w));
#endif
  }

  virtual ~AveragePoolingCudaCudnn() {
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc_));
  }
  virtual string name() { return "AveragePoolingCudaCudnn"; }
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

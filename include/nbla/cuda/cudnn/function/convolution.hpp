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

#ifndef __NBLA_CUDA_CUDNN_FUNCTION_CONVOLUTION_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_CONVOLUTION_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/convolution.hpp>
#include <nbla/function/convolution.hpp>

#include <iostream>

namespace nbla {

template <typename T> class ConvolutionCudaCudnn : public Convolution<T> {
public:
  typedef typename CudaType<T>::type Tw;

  explicit ConvolutionCudaCudnn(const Context &ctx, int base_axis,
                                const vector<int> &pad,
                                const vector<int> &stride,
                                const vector<int> &dilation, int group,
                                bool channel_last)
      : Convolution<T>(ctx, base_axis, pad, stride, dilation, group,
                       channel_last),
        device_(std::stoi(ctx.device_id)) {
#if CUDNN_VERSION < 6000
    // NOTE: dilation > 1 is not supported by cudnn. (2016.10.19)
    for (int i = 0; i < dilation.size(); ++i) {
      if (dilation[i] > 1) {
        // Fall back to original CUDA implementation if dilation > 1.
        // Setting fall_back_func_ overwrites behaviors of setup, forward and
        // backward functions by the specified function class instance.
        std::cout << "Falling back to ConvolutionCuda since dilation > 1 is "
                     "not supported in cuDNN."
                  << std::endl; // TODO: warn.
        this->fall_back_func_.reset(new ConvolutionCuda<T>(
            ctx, base_axis, pad, stride, dilation, group, channel_last));
        return;
      }
    }
#endif
  }
  virtual ~ConvolutionCudaCudnn() {
    if (this->fall_back_func_)
      return;
  }
  virtual string name() { return "ConvolutionCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }
  /**
  void set_cudnn_convolution_forward_algorithm(std::string algorithm);
  void set_cudnn_convolution_backward_filter_algorithm(std::string algorithm);
  void set_cudnn_convolution_backward_data_algorithm(std::string algorithm);
  void set_cudnn_convolution_mode(std::string mode);
  */

protected:
  int device_;
  cudnnHandle_t cudnn_handle_;
  cudnnHandle_t dgrad_handle_;
  shared_ptr<cudaEvent_t> default_event_;
  shared_ptr<cudaEvent_t> dgrad_event_;
  shared_ptr<cudaStream_t> dgrad_stream_;
#if CUDNN_VERSION < 7000
  int x_offset_;
  int w_offset_;
  int b_offset_;
  int y_offset_;
#endif
  shared_ptr<CudnnConvResource> rsc_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);

  void wait_default_on_dgrad();
  void wait_dgrad_on_default();
};
}
#endif

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

#ifndef __NBLA_CUDA_FUNCTION_DEPTHWISEDECONVOLUTION_HPP__
#define __NBLA_CUDA_FUNCTION_DEPTHWISEDECONVOLUTION_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/depthwise_deconvolution.hpp>

namespace nbla {

template <typename T>
class DepthwiseDeconvolutionCuda : public DepthwiseDeconvolution<T> {
public:
  typedef typename CudaType<T>::type Tc;
  DepthwiseDeconvolutionCuda(const Context &ctx, int base_axis,
                             const vector<int> &pad, const vector<int> &stride,
                             const vector<int> &dilation, int divisor)
      : DepthwiseDeconvolution<T>(ctx, base_axis, pad, stride, dilation,
                                  divisor),
        device_(std::stoi(ctx.device_id)) {}

  virtual ~DepthwiseDeconvolutionCuda() {}

  virtual string name() { return "DepthwiseDeconvolutionCuda"; }

  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;

  int warp_size_, max_threads_per_block_;

  int input_data_size_;
  int output_data_size_;

  int kernel_1d_, stride_1d_, padding_1d_, dilation_1d_;
  int2 sample_1d_, outmap_1d_;

  int2 kernel_2d_, stride_2d_, padding_2d_, dilation_2d_;
  int3 sample_2d_, outmap_2d_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);

  virtual void forward_impl(const Variables &inputs, const Variables &outputs);

  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

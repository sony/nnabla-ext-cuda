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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_SUM_POOLING_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_SUM_POOLING_HPP

#include <nbla/function.hpp>

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <nbla/cuda/cudnn/function/average_pooling.hpp>
#include <nbla/function/sum_pooling.hpp>

namespace nbla {

template <typename T> class SumPoolingCudaCudnn : public SumPooling<T> {
public:
  typedef typename CudaType<T>::type Tcu;
  // Current implementation depends on AveragePoolingCudaCudn since cudnn does
  // not support the sumpooling.
  // shared_ptr<Function> average_pooling_;

  explicit SumPoolingCudaCudnn(const Context &ctx, const vector<int> &kernel,
                               const vector<int> &stride, bool ignore_border,
                               const vector<int> &pad, bool channel_last)
      : SumPooling<T>(ctx, kernel, stride, ignore_border, pad, channel_last),
        device_(std::stoi(ctx.device_id)),
        average_pooling_(ctx, kernel, stride, ignore_border, pad, channel_last,
                         true) {}
  virtual string name() { return "SumPoolingCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  int pool_size_;
  AveragePoolingCudaCudnn<T> average_pooling_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

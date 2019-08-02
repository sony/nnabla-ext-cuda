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

#ifndef __NBLA_CUDA_CUDNN_FUNCTION_LOG_SOFTMAX_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_LOG_SOFTMAX_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/function/log_softmax.hpp>

namespace nbla {

/** @copydoc LogSoftmax

@note The default algorithm is set as ACCURATE. TODO: Set an algorithm by
      context.
*/
template <typename T> class LogSoftmaxCudaCudnn : public LogSoftmax<T> {
public:
  typedef typename CudaType<T>::type Tw;

  explicit LogSoftmaxCudaCudnn(const Context &ctx, int axis)
      : LogSoftmax<T>(ctx, axis), device_(std::stoi(ctx.device_id)) {}
  virtual string name() { return "LogSoftmaxCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  CudnnSoftmax::Ptr cudnn_softmax_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

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

#ifndef NBLA_CUDA_FUNCTION_ONE_HOT_HPP
#define NBLA_CUDA_FUNCTION_ONE_HOT_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/one_hot.hpp>

namespace nbla {

template <typename TI, typename T> class OneHotCuda : public OneHot<TI, T> {
public:
  typedef typename CudaType<TI>::type TIcu;
  typedef typename CudaType<T>::type Tcu;

  explicit OneHotCuda(const Context &ctx, const vector<int> &shape)
      : OneHot<TI, T>(ctx, shape), device_(std::stoi(ctx.device_id)) {}
  virtual ~OneHotCuda() {}
  virtual string name() { return "OneHotCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  NdArray stride_info_buf_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

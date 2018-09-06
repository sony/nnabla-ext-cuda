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

#ifndef __NBLA_CUDA_FUNCTION_LEAKY_RELU_HPP__
#define __NBLA_CUDA_FUNCTION_LEAKY_RELU_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/leaky_relu.hpp>

namespace nbla {

/** @copydoc LeakyReLU
*/
template <typename T> class LeakyReLUCuda : public LeakyReLU<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit LeakyReLUCuda(const Context &ctx, float alpha, bool inplace)
      : LeakyReLU<T>(ctx, alpha, inplace) {}
  virtual ~LeakyReLUCuda() {}
  virtual string name() { return "LeakyReLUCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

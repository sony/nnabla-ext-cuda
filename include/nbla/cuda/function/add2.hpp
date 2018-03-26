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

#ifndef __NBLA_CUDA_FUNCTION_ADD2_HPP__
#define __NBLA_CUDA_FUNCTION_ADD2_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/add2.hpp>

namespace nbla {

template <typename T> class Add2Cuda : public Add2<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit Add2Cuda(const Context &ctx, bool inplace) : Add2<T>(ctx, inplace) {}
  virtual ~Add2Cuda() {}
  virtual string name() { return "Add2Cuda"; }
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

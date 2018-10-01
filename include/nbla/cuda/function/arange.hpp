// Copyright (c) 2018 Sony Corporation. All Rights Reserved.
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

#ifndef NBLA_CUDA_FUNCTION_ARANGE_HPP
#define NBLA_CUDA_FUNCTION_ARANGE_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/arange.hpp>

namespace nbla {

template <typename T> class ArangeCuda : public Arange<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit ArangeCuda(const Context &ctx, float start, float stop, float step)
      : Arange<T>(ctx, start, stop, step), device_(std::stoi(ctx.device_id)) {}
  virtual ~ArangeCuda() {}
  virtual string name() { return "ArangeCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
};
}
#endif

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

#ifndef NBLA_CUDA_FUNCTION_BOOL_FILL_HPP
#define NBLA_CUDA_FUNCTION_BOOL_FILL_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/bool_fill.hpp>

namespace nbla {

template <typename T> class BoolFillCuda : public BoolFill<T> {
public:
  /* TODO: remove this help message.
  Typedef of CUDA scalar types used in source file.
  This template function class might be instantiated for each CPU scalar types
  (double, float, nbla::Half), however, for Half, CUDA kernel functions
  must use nbla::HalfCuda in which a bunch of device operator functions are
  overloaded. nbla::CudaType<T>::type will translate nbla::Half
  to nbla::HalfCuda. For other types, it will keep it as-is.
  See nbla/cuda/half.hpp for other template utilities.
  */
  typedef typename CudaType<T>::type Tcu;

  explicit BoolFillCuda(const Context &ctx, float value)
      : BoolFill<T>(ctx, value), device_(std::stoi(ctx.device_id)) {}
  virtual ~BoolFillCuda() {}
  virtual string name() { return "BoolFillCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

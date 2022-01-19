// Copyright 2020,2021 Sony Corporation.
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

#ifndef NBLA_CUDA_FUNCTION_SCATTER_ADD_HPP
#define NBLA_CUDA_FUNCTION_SCATTER_ADD_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/scatter_add.hpp>

namespace nbla {

template <typename T> class ScatterAddCuda : public ScatterAdd<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit ScatterAddCuda(const Context &ctx, int axis)
      : ScatterAdd<T>(ctx, axis), device_(std::stoi(ctx.device_id)) {}
  virtual ~ScatterAddCuda() {}
  virtual string name() { return "ScatterAddCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  Variable x0_meta_;
  Variable indices_meta_;
  Variable x1_meta_;
  virtual void copy_meta(const Variable *const src, Variable &dst,
                         Context &cpu_ctx) {
    auto ptr = dst.cast_data_and_get_pointer<int>(cpu_ctx, true);
    for (auto s : src->shape()) {
      *ptr++ = static_cast<int>(s);
    }
    for (auto s : src->strides()) {
      *ptr++ = static_cast<int>(s);
    }
  }
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

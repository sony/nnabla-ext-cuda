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

#ifndef NBLA_CUDA_FUNCTION_WEIGHT_NORMALIZATION_HPP
#define NBLA_CUDA_FUNCTION_WEIGHT_NORMALIZATION_HPP

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla/function/weight_normalization.hpp>
#include <nbla/imperative.hpp>

namespace nbla {

template <typename T>
class WeightNormalizationCuda : public WeightNormalization<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  FunctionPtr f_sum_{nullptr};

  explicit WeightNormalizationCuda(const Context &ctx, int dim, float eps)
      : WeightNormalization<T>(ctx, dim, eps),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~WeightNormalizationCuda() {}
  virtual string name() { return "WeightNormalizationCuda"; }
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

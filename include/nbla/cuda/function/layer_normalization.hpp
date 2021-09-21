// Copyright 2021 Sony Corporation.
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

#ifndef NBLA_CUDA_FUNCTION_LAYER_NORMALIZATION_HPP
#define NBLA_CUDA_FUNCTION_LAYER_NORMALIZATION_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/layer_normalization.hpp>

namespace nbla {

template <typename T>
class LayerNormalizationCuda : public LayerNormalization<T> {
public:
  typedef typename CudaType<T>::type Tc;

  explicit LayerNormalizationCuda(const Context &ctx,
                                  const vector<int> &batch_axis, float eps,
                                  bool no_scale, bool no_bias)
      : LayerNormalization<T>(ctx, batch_axis, eps, no_scale, no_bias),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~LayerNormalizationCuda() {}
  virtual string name() { return "LayerNormalizationCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  Size_t reduce_size_, batch_size_;
  Variable mean_, var_;

  // Internal buffres for backward
  Variable sum_dygamma_, sum_dyxgamma_;
  Variable factor_a_, factor_b_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

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

#ifndef NBLA_CUDA_FUNCTION_QUANTIZE_LINEAR_HPP
#define NBLA_CUDA_FUNCTION_QUANTIZE_LINEAR_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/quantize_linear.hpp>

namespace nbla {

template <typename T> class QuantizeLinearCuda : public QuantizeLinear<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit QuantizeLinearCuda(const Context &ctx, const string &round_mode,
                              bool narrow_range, int dtype)
      : QuantizeLinear<T>(ctx, round_mode, narrow_range, dtype),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~QuantizeLinearCuda() {}
  virtual string name() { return "QuantizeLinearCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void round(Variable *inp, std::string round_mode);
  NBLA_API virtual void saturate(Variable *inp, int min_range, int max_range);
};
}
#endif

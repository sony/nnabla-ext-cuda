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

#ifndef NBLA_CUDA_FUNCTION_MIN_MAX_QUANTIZE_HPP
#define NBLA_CUDA_FUNCTION_MIN_MAX_QUANTIZE_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/min_max_quantize.hpp>

namespace nbla {

template <typename T> class MinMaxQuantizeCuda : public MinMaxQuantize<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit MinMaxQuantizeCuda(const Context &ctx, float decay, bool train,
                              bool ema, bool ste_fine_grained, float eps)
      : MinMaxQuantize<T>(ctx, decay, train, ema, ste_fine_grained, eps),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~MinMaxQuantizeCuda() {}
  virtual string name() { return "MinMaxQuantizeCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  NBLA_API virtual void nudge_range(Variable *qr_min, Variable *qr_max);
  NBLA_API virtual void nudge_qr_min_max(Variable *qr_min, Variable *qr_max,
                                         Variable *ql_min, Variable *ql_max,
                                         Variable *scale,
                                         Variable *qr_min_nudged,
                                         Variable *qr_max_nudged);
};
}
#endif

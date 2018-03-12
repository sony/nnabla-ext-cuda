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

// fixed_point_quantize.hpp
#ifndef __NBLA_CUDA_FUNCTION_QUANTIZE_HPP__
#define __NBLA_CUDA_FUNCTION_QUANTIZE_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/fixed_point_quantize.hpp>

namespace nbla {

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename T>
class FixedPointQuantizeCuda : public FixedPointQuantize<T> {

protected:
  int device_;

public:
  typedef typename CudaType<T>::type Tc;
  explicit FixedPointQuantizeCuda(const Context &ctx, bool sign, int bw,
                                  float delta, bool ste_fine_grained)
      : FixedPointQuantize<T>(ctx, sign, bw, delta, ste_fine_grained),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~FixedPointQuantizeCuda() {}

  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "FixedPointQuantizeCuda"; }
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

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

// pow2_point_quantize.hpp
#ifndef __NBLA_CUDA_FUNCTION_POW2QUANTIZE_HPP__
#define __NBLA_CUDA_FUNCTION_POW2QUANTIZE_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/pow2_quantize.hpp>

namespace nbla {

/**
    @todo PLACE HERE FUNCTION DOCUMENTATION.
 */
template <typename T> class Pow2QuantizeCuda : public Pow2Quantize<T> {
protected:
  int device_;

public:
  typedef typename CudaType<T>::type Tc;

  explicit Pow2QuantizeCuda(const Context &ctx, bool sign, bool with_zero,
                            int n, int m, bool ste_fine_grained)
      : Pow2Quantize<T>(ctx, sign, with_zero, n, m, ste_fine_grained),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~Pow2QuantizeCuda() {}

  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "Pow2QuantizeCuda"; }
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

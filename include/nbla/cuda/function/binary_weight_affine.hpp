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

/** BinaryWeightAffine
 */
#ifndef __NBLA_CUDA_FUNCTION_BINARYWEIGHTAFFINE_HPP__
#define __NBLA_CUDA_FUNCTION_BINARYWEIGHTAFFINE_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/binary_weight_affine.hpp>

namespace nbla {

template <typename T>
class BinaryWeightAffineCuda : public BinaryWeightAffine<T> {
public:
  typedef typename CudaType<T>::type Tc;

  BinaryWeightAffineCuda(const Context &ctx, int base_axis,
                         float quantize_zero_to)
      : BinaryWeightAffine<T>(ctx, base_axis, quantize_zero_to),
        device_(std::stoi(ctx.device_id)) {}

  virtual ~BinaryWeightAffineCuda() {}

  virtual string name() { return "BinaryWeightAffineCuda"; }

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

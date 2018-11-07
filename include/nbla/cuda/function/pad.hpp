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

#ifndef NBLA_CUDA_FUNCTION_PAD_HPP
#define NBLA_CUDA_FUNCTION_PAD_HPP

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/function/pad.hpp>

namespace nbla {

template <typename T> class PadCuda : public Pad<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit PadCuda(const Context &ctx, const vector<int> &pad_width,
                   const string &mode, float constant_value)
      : Pad<T>(ctx, pad_width, mode, constant_value),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~PadCuda() {}
  virtual string name() { return "PadCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  std::unique_ptr<CudaCachedArray> parameter_memory_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

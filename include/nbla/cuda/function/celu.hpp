// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

/** CELU
 */
#ifndef __NBLA_CUDA_FUNCTION_CELU_HPP__
#define __NBLA_CUDA_FUNCTION_CELU_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/celu.hpp>
namespace nbla {
/** @copydoc CELU
*/

template <typename T> class CELUCuda : public CELU<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit CELUCuda(const Context &ctx, double alpha, int axis)
      : CELU<T>(ctx, alpha, axis), device_(std::stoi(ctx.device_id)) {}
  virtual ~CELUCuda() {}
  virtual string name() { return "CELUCuda"; }
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

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

/** Inverse
 */
#ifndef __NBLA_CUDA_FUNCTION_INVERSE_HPP__
#define __NBLA_CUDA_FUNCTION_INVERSE_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/function/inverse.hpp>
namespace nbla {
/** @copydoc Inverse
*/

template <typename T> class InverseCuda : public Inverse<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit InverseCuda(const Context &ctx)
      : Inverse<T>(ctx),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~InverseCuda() {}
  virtual string name() { return "InverseCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  int dim_;
  // for backward
  shared_ptr<Function> f_batch_matmul1_, f_batch_matmul2_;
  VariablePtr inv_x_, matmul1_out_, gx_, gy_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}

#endif

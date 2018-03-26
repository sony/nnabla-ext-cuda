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

/** MeanSubtraction
 */
#ifndef __NBLA_CUDA_FUNCTION_MEANSUBTRACTION_HPP__
#define __NBLA_CUDA_FUNCTION_MEANSUBTRACTION_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/mean_subtraction.hpp>
namespace nbla {
/** @copydoc MeanSubtraction
*/

template <typename T> class MeanSubtractionCuda : public MeanSubtraction<T> {
public:
  typedef typename CudaType<T>::type Tc;

  explicit MeanSubtractionCuda(const Context &ctx, int base_axis,
                               bool update_running_mean)
      : MeanSubtraction<T>(ctx, base_axis, update_running_mean),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~MeanSubtractionCuda() {}
  virtual string name() { return "MeanSubtractionCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual void forward_impl_batch(const Variables &inputs,
                                  const Variables &outputs);
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs);
  virtual void backward_impl_batch(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum);
  virtual void backward_impl_global(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum);
};
}

#endif

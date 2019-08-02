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

/** Min
 */
#ifndef __NBLA_CUDA_FUNCTION_MIN_HPP__
#define __NBLA_CUDA_FUNCTION_MIN_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/min.hpp>

namespace nbla {
/** @copydoc Min
*/

template <typename T> class MinCuda : public Min<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit MinCuda(const Context &ctx, const vector<int> &axes, bool keep_dims,
                   bool with_index, bool only_index)
      : Min<T>(ctx, axes, keep_dims, with_index, only_index),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~MinCuda() {}
  virtual string name() { return "MinCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_reduce(const T *x, T *y, int outer_size,
                                   int reduction_size);
  virtual void backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                    int reduction_size, bool accum);
};
}

#endif

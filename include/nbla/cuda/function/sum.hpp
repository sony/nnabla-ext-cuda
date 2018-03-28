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

/** Sum
 */
#ifndef __NBLA_CUDA_FUNCTION_SUM_HPP__
#define __NBLA_CUDA_FUNCTION_SUM_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/sum.hpp>

namespace nbla {
/** @copydoc Sum
*/

template <typename T> class SumCuda : public Sum<T> {
public:
  typedef typename CudaType<T>::type Tc;

  explicit SumCuda(const Context &ctx, const vector<int> &axes, bool keep_dims)
      : Sum<T>(ctx, axes, keep_dims), device_(std::stoi(ctx.device_id)) {}
  virtual ~SumCuda() {}
  virtual string name() { return "SumCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void forward_impl_reduce(const T *x, T *y, int outer_size,
                                   int reduction_size);
  virtual void backward_impl_reduce(const T *dy, T *dx, int outer_size,
                                    int reduction_size, bool accum);
};
}

#endif

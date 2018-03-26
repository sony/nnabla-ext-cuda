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

/** Randint
 */
#ifndef __NBLA_CUDA_FUNCTION_RANDINT_HPP__
#define __NBLA_CUDA_FUNCTION_RANDINT_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/randint.hpp>

namespace nbla {
/** @copydoc Randint
*/

template <typename T> class RandintCuda : public Randint<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit RandintCuda(const Context &ctx, int low, int high,
                       const vector<int> &shape, int seed)
      : Randint<T>(ctx, low, high, shape, seed),
        device_(std::stoi(ctx.device_id)) {
    cuda_set_device(device_);
    if (this->seed_ != -1) {
      curand_generator_ = curand_create_generator(this->seed_);
    } else {
      curand_generator_ = SingletonManager::get<Cuda>()->curand_generator();
    }
  }
  virtual ~RandintCuda() {
    if (this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "RandintCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  curandGenerator_t curand_generator_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}

#endif

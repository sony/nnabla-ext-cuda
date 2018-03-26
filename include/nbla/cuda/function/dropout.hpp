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

/** Dropout
 */
#ifndef __NBLA_CUDA_FUNCTION_DROPOUT_HPP__
#define __NBLA_CUDA_FUNCTION_DROPOUT_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/dropout.hpp>

namespace nbla {

template <typename T> class DropoutCuda : public Dropout<T> {
public:
  typedef typename CudaType<T>::type Tc;

  explicit DropoutCuda(const Context &ctx, double p, int seed = -1)
      : Dropout<T>(ctx, T(p), seed) {
    cuda_set_device(std::stoi(ctx.device_id));
    NBLA_CHECK(this->p_ > 0., error_code::value,
               "p must be between 0.0 and 1.0");
    NBLA_CHECK(this->p_ < 1., error_code::value,
               "p must be between 0.0 and 1.0");
    this->scale_ = 1. / (1. - this->p_);
    // if seed is set, create local curand generator.
    if (this->seed_ != -1) {
      // CURAND_RNG_PSEUDO_DEFAULT is CURAND_RNG_PSEUDO_XORWOW.
      curand_generator_ = curand_create_generator(this->seed_);
    } else {
      curand_generator_ = SingletonManager::get<Cuda>()->curand_generator();
    }
  }
  virtual ~DropoutCuda() {
    if (this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "DropoutCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  curandGenerator_t curand_generator_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

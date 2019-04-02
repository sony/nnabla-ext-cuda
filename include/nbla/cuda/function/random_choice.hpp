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

#ifndef NBLA_CUDA_FUNCTION_RANDOM_CHOICE_HPP
#define NBLA_CUDA_FUNCTION_RANDOM_CHOICE_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/random_choice.hpp>

namespace nbla {

template <typename T> class RandomChoiceCuda : public RandomChoice<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit RandomChoiceCuda(const Context &ctx, const vector<int> &shape,
                            bool replace, int seed)
      : RandomChoice<T>(ctx, shape, replace, seed),
        device_(std::stoi(ctx.device_id)) {
    cuda_set_device(device_);
    if (this->seed_ != -1) {
      curand_generator_ = curand_create_generator(this->seed_);
    } else {
      curand_generator_ = SingletonManager::get<Cuda>()->curand_generator();
    }
  }
  virtual ~RandomChoiceCuda() {
    if (this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "RandomChoiceCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  curandGenerator_t curand_generator_;
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual void sample_with_replacement(const Variables &inputs,
                                       const Variables &outputs);
  virtual void sample_without_replace(const Variables &inputs,
                                      const Variables &outputs);
};
}
#endif

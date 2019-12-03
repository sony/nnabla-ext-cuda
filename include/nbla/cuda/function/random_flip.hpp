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

#ifndef NBLA_CUDA_FUNCTION_RANDOM_FLIP_HPP
#define NBLA_CUDA_FUNCTION_RANDOM_FLIP_HPP

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/random_flip.hpp>

namespace nbla {

template <typename T> class RandomFlipCuda : public RandomFlip<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit RandomFlipCuda(const Context &ctx, const vector<int> &axes,
                          int base_axis, int seed)
      : RandomFlip<T>(ctx, axes, base_axis, seed),
        device_(std::stoi(ctx.device_id)) {

    cuda_set_device(std::stoi(ctx.device_id));
    if (this->seed_ != -1) {
      curand_generator_ = curand_create_generator(this->seed_);
    } else {
      curand_generator_ = SingletonManager::get<Cuda>()->curand_generator();
    }
  }
  virtual ~RandomFlipCuda() {}
  virtual string name() { return "RandomFlipCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  curandGenerator_t curand_generator_;
  int device_;
  shared_ptr<CudaCachedArray> flip_flags_;
  NdArray shape_info_buf_;
  NdArray onehot_axses_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

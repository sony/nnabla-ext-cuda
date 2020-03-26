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

#ifndef NBLA_CUDA_FUNCTION_RANDOM_ERASE_HPP
#define NBLA_CUDA_FUNCTION_RANDOM_ERASE_HPP

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/random_erase.hpp>

namespace nbla {

template <typename T> class RandomEraseCuda : public RandomErase<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit RandomEraseCuda(const Context &ctx, float prob,
                           const vector<float> &area_ratios,
                           const vector<float> &aspect_ratios,
                           const vector<float> &replacements, int n, bool share,
                           bool inplace, int base_axis, int seed,
                           bool channel_last, bool ste_fine_grained)
      : RandomErase<T>(ctx, prob, area_ratios, aspect_ratios, replacements, n,
                       share, inplace, base_axis, seed, channel_last,
                       ste_fine_grained),
        device_(std::stoi(ctx.device_id)) {
    cuda_set_device(device_);
    if (this->seed_ != -1) {
      curand_generator_ = curand_create_generator(this->seed_);
    } else {
      curand_generator_ = SingletonManager::get<Cuda>()->curand_generator();
    }
  }
  virtual ~RandomEraseCuda() {
    if (this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "RandomEraseCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  NdArrayPtr state_;
  curandGenerator_t curand_generator_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

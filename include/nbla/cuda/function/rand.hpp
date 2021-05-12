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

/** Rand
 */
#ifndef __NBLA_CUDA_FUNCTION_RAND_HPP__
#define __NBLA_CUDA_FUNCTION_RAND_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/rand.hpp>

namespace nbla {
/** @copydoc Rand
*/

template <typename T> class RandCuda : public Rand<T> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit RandCuda(const Context &ctx, float low, float high,
                    const vector<int> &shape, int seed)
      : Rand<T>(ctx, low, high, shape, seed),
        device_(std::stoi(ctx.device_id)) {
    cuda_set_device(device_);
    if (this->seed_ != -1) {
      curand_generator_ = curand_create_generator(this->seed_);
    }
  }
  virtual ~RandCuda() {
    if (this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "RandCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  curandGenerator_t curand_generator_;
  bool save_output_data_ = false;
  NdArray output_data_for_recomp_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual void setup_recompute_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &need_recompute);
  virtual void recompute_impl(const Variables &inputs, const Variables &outputs,
                              const vector<bool> &need_recompute);
};
}

#endif

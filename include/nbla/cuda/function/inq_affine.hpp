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

/** INQAffine
 */
#ifndef __NBLA_CUDA_FUNCTION_INQAFFINE_HPP__
#define __NBLA_CUDA_FUNCTION_INQAFFINE_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/function/inq_affine.hpp>
namespace nbla {
/** @copydoc INQAffine
*/

template <typename T, typename T1>
class INQAffineCuda : public INQAffine<T, T1> {

public:
  typedef typename CudaType<T>::type Tc;
  explicit INQAffineCuda(const Context &ctx, int base_axis, int num_bits,
                         const vector<int> &inq_iterations,
                         const string &selection_algorithm, int seed)
      : INQAffine<T, T1>(ctx, base_axis, num_bits, inq_iterations,
                         selection_algorithm, seed),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~INQAffineCuda() {
    if (this->selection_algorithm_ == "random" && this->seed_ != -1) {
      curand_destroy_generator(curand_generator_);
    }
  }
  virtual string name() { return "INQAffineCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  curandGenerator_t curand_generator_;
  Variable indices_;
  Variable cumulative_count_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}

#endif

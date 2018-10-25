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

#ifndef NBLA_CUDA_FUNCTION_PRUNE_HPP
#define NBLA_CUDA_FUNCTION_PRUNE_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/prune.hpp>

#include <nbla/array.hpp>
#include <nbla/communicator/multi_process_data_parallel_communicator.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/context.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nbla {

template <typename T> class PruneCuda : public Prune<T> {

public:
  typedef typename CudaType<T>::type Tc;

  explicit PruneCuda(const Context &ctx, float rate)
      : Prune<T>(ctx, rate), device_(std::stoi(ctx.device_id)) {}
  virtual ~PruneCuda() {}
  virtual string name() { return "PruneCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

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

#ifndef NBLA_CUDA_FUNCTION_GROUP_NORMALIZATION_HPP
#define NBLA_CUDA_FUNCTION_GROUP_NORMALIZATION_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/group_normalization.hpp>

namespace nbla {

template <typename T>
class GroupNormalizationCuda : public GroupNormalization<T> {
public:
  explicit GroupNormalizationCuda(const Context &ctx, int num_groups,
                                  int channel_axis,
                                  const vector<int> &batch_axis, float eps,
                                  bool no_scale, bool no_bias)
      : GroupNormalization<T>(ctx, num_groups, channel_axis, batch_axis, eps,
                              no_scale, no_bias),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~GroupNormalizationCuda() {}
  virtual string name() { return "GroupNormalizationCuda"; }
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

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

#ifndef NBLA_CUDA_FUNCTION_WARP_BY_GRID_HPP
#define NBLA_CUDA_FUNCTION_WARP_BY_GRID_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/warp_by_grid.hpp>

namespace nbla {

template <typename T> class WarpByGridCuda : public WarpByGrid<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit WarpByGridCuda(const Context &ctx, const string &mode,
                          const string &padding_mode, bool align_corners,
                          bool channel_last)
      : WarpByGrid<T>(ctx, mode, padding_mode, align_corners, channel_last),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~WarpByGridCuda() {}
  virtual string name() { return "WarpByGridCuda"; }
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

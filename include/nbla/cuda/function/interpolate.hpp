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

#ifndef NBLA_CUDA_FUNCTION_INTERPOLATE_HPP
#define NBLA_CUDA_FUNCTION_INTERPOLATE_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/interpolate.hpp>

namespace nbla {

template <typename T> class InterpolateCuda : public Interpolate<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit InterpolateCuda(const Context &ctx, const vector<int> &output_size,
                           const string &mode, bool align_corners,
                           bool half_pixel, bool half_pixel_for_nn,
                           bool channel_last)
      : Interpolate<T>(ctx, output_size, mode, align_corners, half_pixel,
                       half_pixel_for_nn, channel_last),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~InterpolateCuda() {}
  virtual string name() { return "InterpolateCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

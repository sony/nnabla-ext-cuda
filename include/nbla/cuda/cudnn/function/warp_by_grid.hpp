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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_WARP_BY_GRID_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_WARP_BY_GRID_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <nbla/cuda/function/warp_by_grid.hpp>

namespace nbla {

namespace warp_by_grid {
inline bool cudnn_condition(const int size, const std::string mode,
                            const PADDING_MODE padding_mode_t,
                            const bool align_corners, const bool channel_last) {
  return (size == 4 && mode == "linear" &&
          padding_mode_t == PADDING_MODE::zero && align_corners == true &&
          !channel_last)
             ? true
             : false;
}
}

template <typename T> class WarpByGridCudaCudnn : public WarpByGridCuda<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit WarpByGridCudaCudnn(const Context &ctx, const string &mode,
                               const string &padding_mode, bool align_corners,
                               bool channel_last)
      : WarpByGridCuda<T>(ctx, mode, padding_mode, align_corners, channel_last),
        device_(std::stoi(ctx.device_id)) {
    NBLA_CUDNN_CHECK(
        cudnnCreateSpatialTransformerDescriptor(&spatial_tf_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~WarpByGridCudaCudnn() {
    NBLA_CUDNN_CHECK(
        cudnnDestroySpatialTransformerDescriptor(spatial_tf_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
  }
  virtual string name() { return "WarpByGridCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;

  cudnnSpatialTransformerDescriptor_t spatial_tf_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

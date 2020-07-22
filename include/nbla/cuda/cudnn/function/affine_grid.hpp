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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_AFFINE_GRID_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_AFFINE_GRID_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <nbla/cuda/function/affine_grid.hpp>

namespace nbla {

namespace affine_grid {
inline bool cudnn_condition(int size, bool align_corners) {
  return (size == 2 && align_corners == true) ? true : false;
}
}

template <typename T> class AffineGridCudaCudnn : public AffineGridCuda<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit AffineGridCudaCudnn(const Context &ctx, const vector<int> &size,
                               bool align_corners)
      : AffineGridCuda<T>(ctx, size, align_corners),
        device_(std::stoi(ctx.device_id)) {
    if (affine_grid::cudnn_condition(this->size_.size(),
                                     this->align_corners_)) {
      NBLA_CUDNN_CHECK(cudnnCreateSpatialTransformerDescriptor(&theta_desc_));
    }
  }
  virtual ~AffineGridCudaCudnn() {
    if (affine_grid::cudnn_condition(this->size_.size(),
                                     this->align_corners_)) {
      NBLA_CUDNN_CHECK(cudnnDestroySpatialTransformerDescriptor(theta_desc_));
    }
  }
  virtual string name() { return "AffineGridCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;

  cudnnSpatialTransformerDescriptor_t theta_desc_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif

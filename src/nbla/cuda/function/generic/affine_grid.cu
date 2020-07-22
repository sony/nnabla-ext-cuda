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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/affine_grid.hpp>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, bool align_corners>
__global__ void kernel_generate_target_grid_2d(const Size_t isize, T *grid,
                                               int3 shape, int2 stride, int B) {

  NBLA_CUDA_KERNEL_LOOP(idx, isize) {
    auto H = shape.x;
    auto W = shape.y;
    auto nd_index = device_flat_to_3d(idx, stride);
    auto h = nd_index.x;
    auto w = nd_index.y;
    auto v = nd_index.z;

    for (auto b = 0; b < B; b++) {
      auto bidx = idx + b * isize;
      // [-1, 1] <--> [0, S - 1] if align_corner
      // [-1, 1] <--> [-0.5, S - 0.5] = [0 - 0.5, S - 1 + 0.5] if not
      // align_corner
      // Num. of v = 3, corresponding to (x, y, 1)
      if (v == 0) {
        auto x = T(2.0) * w / (W - 1) - T(1.0);
        x = align_corners ? x : x * (T(W - 1) / T(W));
        grid[bidx] = x;
      } else if (v == 1) {
        auto y = T(2.0) * h / (H - 1) - T(1.0);
        y = align_corners ? y : y * (T(H - 1) / T(H));
        grid[bidx] = y;
      } else {
        grid[bidx] = T(1);
      }
    }
  }
}

template <typename T, bool align_corners>
__global__ void kernel_generate_target_grid_3d(const Size_t isize, T *grid,
                                               int4 shape, int3 stride, int B) {

  NBLA_CUDA_KERNEL_LOOP(idx, isize) {
    auto D = shape.x;
    auto H = shape.y;
    auto W = shape.z;
    auto nd_index = device_flat_to_4d(idx, stride);
    auto d = nd_index.x;
    auto h = nd_index.y;
    auto w = nd_index.z;
    auto v = nd_index.w;

    for (auto b = 0; b < B; b++) {
      auto bidx = idx + b * isize;
      // [-1, 1] <--> [0, S - 1] if align_corner
      // [-1, 1] <--> [-0.5, S - 0.5] = [0 - 0.5, S - 1 + 0.5] if not
      // align_corner
      // Num. of v = 3, corresponding to (x, y, 1)
      if (v == 0) {
        auto x = T(2.0) * w / (W - 1) - T(1.0);
        x = align_corners ? x : x * (T(W - 1) / T(W));
        grid[bidx] = x;
      } else if (v == 1) {
        auto y = T(2.0) * h / (H - 1) - T(1.0);
        y = align_corners ? y : y * (T(H - 1) / T(H));
        grid[bidx] = y;
      } else if (v == 2) {
        auto z = T(2.0) * d / (D - 1) - T(1.0);
        z = align_corners ? z : z * (T(D - 1) / T(D));
        grid[bidx] = z;
      } else {
        grid[bidx] = T(1);
      }
    }
  }
}

template <typename T>
void AffineGridCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  AffineGrid<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void AffineGridCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);

  auto affine = inputs[0];
  auto grid_s = outputs[0];

  if (this->size_.size() == 2) {
    // Target grid (with 1 for the translation)
    auto B = affine->shape()[0];
    auto H = this->size_[0];
    auto W = this->size_[1];
    Variable grid_t(Shape_t{B, H, W, 3});

    auto isize = H * W * 3;
    auto shape = make_int3(H, W, 3);
    auto stride = make_int2(W * 3, 3);
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto generate_target_grid =
        this->align_corners_ ? kernel_generate_target_grid_2d<Tcu, true>
                             : kernel_generate_target_grid_2d<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(generate_target_grid, isize, grid_t_ptr,
                                   shape, stride, B);

    // Transform: (B, H, W, 3) @ (B, 2, 3) --> (B, H, W, 2)
    grid_t.reshape(Shape_t{B, H * W, 3}, false);
    grid_s->reshape(Shape_t{B, H * W, 2}, false);
    execute(this->batch_matmul_, Variables{&grid_t, affine}, Variables{grid_s});
    grid_s->reshape(Shape_t{B, H, W, 2}, false);
  } else if (this->size_.size() == 3) {
    // Target grid (with 1 for the translation)
    auto B = affine->shape()[0];
    auto D = this->size_[0];
    auto H = this->size_[1];
    auto W = this->size_[2];
    Variable grid_t(Shape_t{B, D, H, W, 4});

    auto isize = D * H * W * 4;
    auto shape = make_int4(D, H, W, 4);
    auto stride = make_int3(H * W * 4, W * 4, 4);
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto generate_target_grid =
        this->align_corners_ ? kernel_generate_target_grid_3d<Tcu, true>
                             : kernel_generate_target_grid_3d<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(generate_target_grid, isize, grid_t_ptr,
                                   shape, stride, B);

    /// Transform: (B, D, H, W, 4) @ (B, 3, 4) --> (B, D, H, W, 3)
    grid_t.reshape(Shape_t{B, D * H * W, 4}, false);
    grid_s->reshape(Shape_t{B, D * H * W, 3}, false);
    execute(this->batch_matmul_, Variables{&grid_t, affine}, Variables{grid_s});
    grid_s->reshape(Shape_t{B, D, H, W, 3}, false);
  }
}

template <typename T>
void AffineGridCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  // Gradient of outputs
  auto affine = inputs[0];
  auto B = affine->shape()[0];
  auto grid_s = outputs[0];
  if (this->size_.size() == 2) {
    // Target grid with 1 for the translation
    auto B = affine->shape()[0];
    auto H = this->size_[0];
    auto W = this->size_[1];
    Variable grid_t(Shape_t{B, H, W, 3});

    auto isize = H * W * 3;
    auto shape = make_int3(H, W, 3);
    auto stride = make_int2(W * 3, 3);
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto generate_target_grid =
        this->align_corners_ ? kernel_generate_target_grid_2d<Tcu, true>
                             : kernel_generate_target_grid_2d<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(generate_target_grid, isize, grid_t_ptr,
                                   shape, stride, B);

    // Backward of the transformation: (B, H, W, 2) @ (B, 2, 3) --> (B, H, W, 2)
    grid_t.reshape(Shape_t{B, H * W, 3}, false);
    grid_s->reshape(Shape_t{B, H * W, 2}, false);
    nbla::backward(this->batch_matmul_, Variables{&grid_t, affine},
                   Variables{grid_s}, vector<bool>{false, propagate_down[0]},
                   vector<bool>{false, accum[0]});
    grid_s->reshape(Shape_t{B, H, W, 2}, false);
  } else if (this->size_.size() == 3) {
    auto B = affine->shape()[0];
    auto D = this->size_[0];
    auto H = this->size_[1];
    auto W = this->size_[2];
    Variable grid_t(Shape_t{B, D, H, W, 4});

    auto isize = D * H * W * 4;
    auto shape = make_int4(D, H, W, 4);
    auto stride = make_int3(H * W * 4, W * 4, 4);
    auto grid_t_ptr = grid_t.cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto generate_target_grid =
        this->align_corners_ ? kernel_generate_target_grid_3d<Tcu, true>
                             : kernel_generate_target_grid_3d<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(generate_target_grid, isize, grid_t_ptr,
                                   shape, stride, B);

    // Backward of the transformation: (B, D, H, W, 4) @ (B, 3, 4) --> (B, D, H,
    // W, 3)
    grid_t.reshape(Shape_t{B, D * H * W, 4}, false);
    grid_s->reshape(Shape_t{B, D * H * W, 3}, false);
    nbla::backward(this->batch_matmul_, Variables{&grid_t, affine},
                   Variables{grid_s}, vector<bool>{false, propagate_down[0]},
                   vector<bool>{false, accum[0]});
    grid_s->reshape(Shape_t{B, D, H, W, 3}, false);
  }
}
}

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
#include <nbla/cuda/function/random_erase.hpp>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/cuda/utils/random.cuh>
#include <nbla/variable.hpp>

#include <curand_kernel.h>

namespace nbla {

namespace random_erase {
template <typename T, bool accum = true>
__global__ void kernel_copy(const int size, T *gx, const T *gy) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    gx[idx] = accum ? gx[idx] + gy[idx] : gy[idx];
  }
}

__global__ void kernel_create_random_coordinates(const int size,
                                                 float *random_coords,
                                                 const int H, const int W,
                                                 const float2 area_ratios,
                                                 const float2 aspect_ratios) {
  auto S = H * W;
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto elm0 = random_coords[idx];
    auto elm1 = random_coords[idx + size * 1];
    auto elm2 = random_coords[idx + size * 2];
    auto elm3 = random_coords[idx + size * 3];
    auto elm4 = random_coords[idx + size * 4];

    auto eprob = elm0;
    auto Se = ((area_ratios.y - area_ratios.x) * elm1 + area_ratios.x) * S;
    auto re = (aspect_ratios.y - aspect_ratios.x) * elm2 + aspect_ratios.x;
    auto He = sqrt(Se * re);
    auto We = sqrt(Se / re);
    He = min(He, (float)H);
    We = min(We, (float)W);
    int ye_start = (H - He) * elm3;
    int xe_start = (W - We) * elm4;

    random_coords[idx] = eprob;
    random_coords[idx + size * 1] = ye_start;
    random_coords[idx + size * 2] = xe_start;
    random_coords[idx + size * 3] = ye_start + He;
    random_coords[idx + size * 4] = xe_start + We;
  }
}

template <typename T, bool channel_last = false, bool share = true,
          bool accum = false>
__global__ void
kernel_random_erase_backward(const int size, T *gx, const T *gy, int3 dstride,
                             int N, const float *random_coords, int3 estride,
                             float prob, float2 replacements) {
  // size: B x C x H x W
  // y, x: (B, C, H, W) or (B, H, W, C)
  // random_coords: (5, N, B) or (5, N, B, C)
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto nd_index = device_flat_to_4d(idx, dstride);
    auto b = nd_index.x;
    auto c = channel_last ? nd_index.w : nd_index.y;
    auto h = channel_last ? nd_index.y : nd_index.z;
    auto w = channel_last ? nd_index.z : nd_index.w;

    auto fall = false;
    for (int n = 0; n < N; n++) {
      auto offset =
          share ? (n * estride.y + b) : (n * estride.y + b * estride.z + c);
      float eprob = random_coords[0 * estride.x + offset];
      int ye_start = random_coords[1 * estride.x + offset];
      int xe_start = random_coords[2 * estride.x + offset];
      int ye_end = random_coords[3 * estride.x + offset];
      int xe_end = random_coords[4 * estride.x + offset];
      if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
          (xe_start <= w && w <= xe_end)) {
        fall = true;
        break;
      }
    }
    if (fall) {
      gx[idx] = accum ? (gx[idx] + T(0)) : T(0);
    } else {
      gx[idx] = accum ? (gx[idx] + gy[idx]) : gy[idx];
    }
  }
}

template <typename T, bool channel_last = false, bool share = true>
__global__ void
kernel_random_erase(const int size, T *y, const T *x, int3 dstride, int N,
                    int4 shape, const float *random_coords, int3 estride,
                    float prob, curandState *func_state, float2 replacements) {
  // size: H x W
  // y, x: (B, C, H, W) or (B, H, W, C)
  // random_coords: (5, N, B) or (5, N, B, C)

  // Minimize size of func_states since it cannot be reused over layers
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto B = shape.x;
    auto C = channel_last ? shape.w : shape.y;
    auto H = channel_last ? shape.y : shape.z;
    auto W = channel_last ? shape.z : shape.w;

    auto h = idx / W;
    auto w = idx - h * W;

    curandState local_state = func_state[idx];
    for (int n = 0; n < N; n++) {
      for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
          auto offset =
              share ? (n * estride.y + b) : (n * estride.y + b * estride.z + c);
          float eprob = random_coords[0 * estride.x + offset];
          int ye_start = random_coords[1 * estride.x + offset];
          int xe_start = random_coords[2 * estride.x + offset];
          int ye_end = random_coords[3 * estride.x + offset];
          int xe_end = random_coords[4 * estride.x + offset];

          auto nd_index =
              channel_last ? make_int4(b, h, w, c) : make_int4(b, c, h, w);
          auto idx_data = device_4d_to_flat(nd_index, dstride);
          if ((eprob <= prob) && (ye_start <= h && h <= ye_end) &&
              (xe_start <= w && w <= xe_end)) {
            y[idx_data] =
                curand_uniform(&local_state, replacements.x, replacements.y);
          }
        }
      }
    }
    func_state[idx] = local_state;
  }
}
}

template <typename T>
void RandomEraseCuda<T>::setup_impl(const Variables &inputs,
                                    const Variables &outputs) {
  RandomErase<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  auto shape = inputs[0]->shape();
  auto H = this->channel_last_ ? shape[this->base_axis_ + 1]
                               : shape[this->base_axis_ + 1];
  auto W = this->channel_last_ ? shape[this->base_axis_]
                               : shape[this->base_axis_ + 2];
  this->state_ = std::make_shared<NdArray>(
      Shape_t{static_cast<long>(H * W * sizeof(curandState))});
  curandState *func_state =
      this->state_->cast(get_dtype<char>(), this->ctx_)->pointer<curandState>();
  curand_initialize(H * W, this->seed_, 0, func_state);
}

template <typename T>
void RandomEraseCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  // Settings
  auto shape = inputs[0]->shape();
  auto N = this->n_;
  auto B =
      std::accumulate(shape.begin(), std::next(shape.begin(), this->base_axis_),
                      1, std::multiplies<size_t>());
  auto C = this->channel_last_ ? shape[this->base_axis_ + 2]
                               : shape[this->base_axis_];
  auto H = this->channel_last_ ? shape[this->base_axis_ + 1]
                               : shape[this->base_axis_ + 1];
  auto W = this->channel_last_ ? shape[this->base_axis_]
                               : shape[this->base_axis_ + 2];

  // Generate 5 x N x B (x C), 5 is {prob, Se, re, xe, ye}
  this->random_coordinates_ =
      this->share_ ? std::make_shared<NdArray>(Shape_t{5, N, B})
                   : std::make_shared<NdArray>(Shape_t{5, N, B, C});
  float *random_coords =
      this->random_coordinates_->cast(get_dtype<float>(), this->ctx_)
          ->template pointer<float>();

  curand_generate_rand<float>(this->curand_generator_, 0.0f, 1.0f,
                              random_coords, this->random_coordinates_->size());

  // Create 5 x N x B (x C), 5 is {prob, ye_start, xe_start, ye_end, xe_end}
  // inplace
  auto area_ratios = make_float2(this->area_ratios_[0], this->area_ratios_[1]);
  auto aspect_ratios =
      make_float2(this->aspect_ratios_[0], this->aspect_ratios_[1]);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (random_erase::kernel_create_random_coordinates),
      this->random_coordinates_->size() / 5, random_coords, H, W, area_ratios,
      aspect_ratios);

  // // Debug
  // auto ctx_cpu = Context({"cpu:float"}, "CpuCachedArray", "0");
  // auto random_coordinates_cpu = std::make_shared<NdArray>(Shape_t{5, N, B});
  // float *random_coords_cpu = random_coordinates_cpu->cast(get_dtype<Tcu>(),
  //                                                         ctx_cpu)->pointer<typename
  //                                                         force_float<T>::type>();

  // cudaMemcpy((void*)random_coords_cpu, (void*)random_coords,
  // 5*N*B*sizeof(float), cudaMemcpyDeviceToHost);

  // for (int n = 0; n < N; ++n) {
  //   for (int b = 0; b < B; ++b) {
  //     float prob = random_coords_cpu[0 * (N*B) + n * (B) + b];
  //     float ye_start = random_coords_cpu[1 * (N*B) + n * (B) + b];
  //     float xe_start = random_coords_cpu[2 * (N*B) + n * (B) + b];
  //     float ye_end = random_coords_cpu[3 * (N*B) + n * (B) + b];
  //     float xe_end = random_coords_cpu[4 * (N*B) + n * (B) + b];
  //     printf("prob = %f\n", prob);
  //     printf("ye_start = %f\n", ye_start);
  //     printf("ye_end = %f\n", ye_end);
  //     printf("xe_start = %f\n", xe_start);
  //     printf("xe_end = %f\n", xe_end);
  //   }
  // }

  // Copy once
  auto size = inputs[0]->size();
  Tcu *y =
      outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, !this->inplace_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((random_erase::kernel_copy<Tcu, false>), size,
                                 y, x);

  // Replace
  auto dstride = this->channel_last_ ? make_int3(H * W * C, W * C, C)
                                     : make_int3(C * H * W, H * W, W);
  auto dshape =
      this->channel_last_ ? make_int4(B, H, W, C) : make_int4(B, C, H, W);
  auto estride =
      this->share_ ? make_int3(N * B, B, 1) : make_int3(N * B * C, B * C, C);
  curandState *func_state =
      this->state_->cast(get_dtype<char>(), this->ctx_)->pointer<curandState>();
  auto replacements =
      make_float2(this->replacements_[0], this->replacements_[1]);
  using random_erase::kernel_random_erase;
  auto kernel = this->channel_last_
                    ? (this->share_ ? kernel_random_erase<Tcu, true, true>
                                    : kernel_random_erase<Tcu, true, false>)
                    : (this->share_ ? kernel_random_erase<Tcu, false, true>
                                    : kernel_random_erase<Tcu, false, false>);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, H * W, y, x, dstride, N, dshape,
                                 random_coords, estride, this->prob_,
                                 func_state, replacements);

  // Release memory
  if (!this->ste_fine_grained_) {
    this->random_coordinates_ = nullptr;
  }
}

template <typename T>
void RandomEraseCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  auto size = inputs[0]->size();
  const Tcu *gy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  Tcu *gx = inputs[0]->cast_grad_and_get_pointer<Tcu>(
      this->ctx_, !(this->inplace_ || accum[0]));

  // STE
  if (!this->ste_fine_grained_) {
    if (accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((random_erase::kernel_copy<Tcu, true>),
                                     size, gx, gy);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((random_erase::kernel_copy<Tcu, false>),
                                     size, gx, gy);
    }
    return;
  }

  // Correct backward
  auto shape = inputs[0]->shape();
  auto N = this->n_;
  auto B =
      std::accumulate(shape.begin(), std::next(shape.begin(), this->base_axis_),
                      1, std::multiplies<size_t>());
  auto C = this->channel_last_ ? shape[this->base_axis_ + 2]
                               : shape[this->base_axis_];
  auto H = this->channel_last_ ? shape[this->base_axis_ + 1]
                               : shape[this->base_axis_ + 1];
  auto W = this->channel_last_ ? shape[this->base_axis_]
                               : shape[this->base_axis_ + 2];
  auto dstride = this->channel_last_ ? make_int3(H * W * C, W * C, C)
                                     : make_int3(C * H * W, H * W, W);
  auto estride =
      this->share_ ? make_int3(N * B, B, 1) : make_int3(N * B * C, B * C, C);
  float *random_coords =
      this->random_coordinates_->cast(get_dtype<float>(), this->ctx_)
          ->template pointer<float>();
  auto replacements =
      make_float2(this->replacements_[0], this->replacements_[1]);
  using random_erase::kernel_random_erase_backward;

  if (accum[0]) {
    auto kernel =
        this->channel_last_
            ? (this->share_
                   ? kernel_random_erase_backward<Tcu, true, true, true>
                   : kernel_random_erase_backward<Tcu, true, false, true>)
            : (this->share_
                   ? kernel_random_erase_backward<Tcu, false, true, true>
                   : kernel_random_erase_backward<Tcu, false, false, true>);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, gx, gy, dstride, N,
                                   random_coords, estride, this->prob_,
                                   replacements);
  } else {
    auto kernel =
        this->channel_last_
            ? (this->share_
                   ? kernel_random_erase_backward<Tcu, true, true, false>
                   : kernel_random_erase_backward<Tcu, true, false, false>)
            : (this->share_
                   ? kernel_random_erase_backward<Tcu, false, true, false>
                   : kernel_random_erase_backward<Tcu, false, false, false>);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, gx, gy, dstride, N,
                                   random_coords, estride, this->prob_,
                                   replacements);
  }

  // Release memory
  this->random_coordinates_ = nullptr;
}
}

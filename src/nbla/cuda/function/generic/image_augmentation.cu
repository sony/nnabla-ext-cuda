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
#include <nbla/cuda/function/image_augmentation.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

#include <curand_kernel.h>

namespace nbla {

__global__ void kernel_prepare_curand(const int num, curandStateXORWOW_t *state,
                                      const int seed) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) { curand_init(seed, idx, 0, &state[idx]); }
}

template <typename T>
__global__ void IAKernel(const T *x, const int w_in, const int h_in,
                         const float x0_in, const float y0_in, T *y,
                         const int w_out, const int h_out, const float x_ax,
                         const float y_ax, const float x_ay, const float y_ay,
                         const float distortion, const float brightness,
                         const float contrast, const float contrast_center,
                         curandStateXORWOW_t *state, const float noise) {
  const int x_out = blockDim.x * blockIdx.x + threadIdx.x;
  const int y_out = blockDim.y * blockIdx.y + threadIdx.y;
  if (x_out < w_out && y_out < h_out) {
    const int out_offset = w_out * y_out + x_out;

    const float w_out_half = w_out * 0.5f;
    const float h_out_half = h_out * 0.5f;

    float dist_x = (x_out - w_out_half) / w_out_half;
    float dist_y = (y_out - h_out_half) / h_out_half;
    const float r = sqrt(dist_x * dist_x + dist_y * dist_y);
    const float r2 = r * r;
    const float dist_scale = 1.0f / (1.0f + distortion);
    dist_x = (dist_x + dist_x * distortion * r2) * w_out_half * dist_scale +
             w_out_half;
    dist_y = (dist_y + dist_y * distortion * r2) * h_out_half * dist_scale +
             h_out_half;

    float x_in = x0_in + dist_x * x_ax + dist_y * y_ax;
    float y_in = y0_in + dist_x * x_ay + dist_y * y_ay;

    if (x_in < 0) {
      x_in = 0.0;
    } else if (x_in > w_in - 1) {
      x_in = w_in - 1;
    }
    if (y_in < 0) {
      y_in = 0.0;
    } else if (y_in > h_in - 1) {
      y_in = h_in - 1;
    }

    // Prepare linear interpolation
    const int intx = (int)x_in;
    const int inty = (int)y_in;
    const float fmodx = x_in - intx;
    const float fmody = y_in - inty;
    const int intx_plus1 = intx < w_in - 1 ? intx + 1 : intx;
    const int inty_plus1 = inty < h_in - 1 ? inty + 1 : inty;
    // Top left
    const int pos0 = intx + inty * w_in;
    const T pos0_gain = (1 - fmodx) * (1 - fmody);
    // Top right
    const int pos1 = intx_plus1 + inty * w_in;
    const T pos1_gain = fmodx * (1 - fmody);
    // Bottom left
    const int pos2 = intx + inty_plus1 * w_in;
    const T pos2_gain = (1 - fmodx) * fmody;
    // Bottom right
    const int pos3 = intx_plus1 + inty_plus1 * w_in;
    const T pos3_gain = fmodx * fmody;

    // Linear interpolation
    T result = x[pos0] * pos0_gain + x[pos1] * pos1_gain + x[pos2] * pos2_gain +
               x[pos3] * pos3_gain;
    result = (result + brightness) * contrast + contrast_center;
    if (state) {
      result += curand_normal(&state[out_offset]) * noise;
    }
    y[out_offset] = result;
  }
}

template <typename T>
void ImageAugmentationCuda<T>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  ImageAugmentation<T>::setup_impl(inputs, outputs);

  Shape_t shape_in = inputs[0]->shape();
  const int w_in = shape_in[shape_in.size() - 1];
  const int h_in = shape_in[shape_in.size() - 2];
  const int num_ch = shape_in.size() >= 3 ? shape_in[shape_in.size() - 3] : 1;
  const int num_image = inputs[0]->size() / (w_in * h_in * num_ch);

  int curand_state_len = 0;
  if (this->noise_ > 0.0) {
    const int data_size = w_in * h_in;

    if (data_size > curand_state_len) {
      curand_state_len = data_size;
    }
  }
  if (curand_state_len) {
    int curand_state_size =
        (sizeof(curandStateXORWOW_t) - 1) / sizeof(T) + sizeof(int);

    // prepare curand state
    Shape_t state_shape;
    state_shape.push_back(curand_state_len * curand_state_size);
    curand_state_.reshape(state_shape, true);
    int *state = curand_state_.cast_data_and_get_pointer<int>(this->ctx_, true);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_prepare_curand, curand_state_len,
                                   (curandStateXORWOW_t *)state, this->seed_);
  }
}

template <typename T>
void ImageAugmentationCuda<T>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  Shape_t shape_in = inputs[0]->shape();
  const int w_in = shape_in[shape_in.size() - 1];
  const int h_in = shape_in[shape_in.size() - 2];
  const int w_in_pad = w_in + this->pad_[1] * 2;
  const int h_in_pad = h_in + this->pad_[0] * 2;
  const int num_ch = shape_in.size() >= 3 ? shape_in[shape_in.size() - 3] : 1;
  const int num_image = inputs[0]->size() / (w_in * h_in * num_ch);
  // std::cout << "shape_in : w=" << w_in << ", h=" << h_in << ", ch=" << num_ch
  // << ", num=" << num_image << "\n";

  Shape_t shape_out = outputs[0]->shape();
  const int w_out = shape_out[shape_out.size() - 1];
  const int h_out = shape_out[shape_out.size() - 2];
  // std::cout << "shape_out : w=" << w_out << ", h=" << h_out << "\n";

  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);

  const int ch_size_in = h_in * w_in;
  const int ch_size_out = h_out * w_out;

  vector<float> channel_brightness(num_ch);
  vector<float> channel_contrast(num_ch);

  int *state =
      this->noise_ > 0.0
          ? curand_state_.cast_data_and_get_pointer<int>(this->ctx_, false)
          : nullptr;
  for (int iim = 0; iim < num_image; ++iim) {
    // Define augmentation settings
    // std::cout << "* image " << iim << "\n";

    const int im_offset_in = iim * w_in * h_in * num_ch;
    const Tc *x_im = x + im_offset_in;
    int im_offset_out = iim * w_out * h_out * num_ch;
    Tc *y_im = y + im_offset_out;
    // std::cout << "offset : in=" << im_offset_in << ", out=" << im_offset_out
    // << "\n";

    const float scale =
        this->min_scale_ *
        std::exp(
            (this->rgen_() % 1001) * 0.001f *
            std::log(this->max_scale_ /
                     this->min_scale_)); // [this->min_scale_, this->max_scale_]
    const float scale_x = std::exp(-std::log(this->aspect_ratio_) * 0.5 +
                                   (this->rgen_() % 1001) * 0.001f *
                                       std::log(this->aspect_ratio_));
    const float scale_y = 1.0 / scale_x;
    const float i_scale_x = 1.0f / (scale * scale_x);
    const float i_scale_y = 1.0f / (scale * scale_y);
    // std::cout << "scale : min=" << min_scale_ << ", max=" << max_scale_ << ",
    // v=" << scale << ", inv=" << i_scale << "\n";

    const float angle = -this->angle_ +
                        ((this->rgen_() % 1001) * 0.001f) * this->angle_ *
                            2; // [-angle_, angle_]
    // std::cout << "angle : " << angle << "\n";

    // Preparation
    const float w_scaled = w_in_pad * scale * scale_x;
    const float h_scaled = h_in_pad * scale * scale_y;
    // std::cout << "shape_scaled : w=" << w_scaled << ", h=" << h_scaled <<
    // "\n";

    const float cx = (w_out - 1) * 0.5f;
    const float cy = (h_out - 1) * 0.5f;
    // std::cout << "center : x=" << cx << ", y=" << cy << "\n";

    const float cx_scaled =
        ((this->rgen_() % 1001) * 0.001f) * (w_scaled - w_out) + cx;
    const float cy_scaled =
        ((this->rgen_() % 1001) * 0.001f) * (h_scaled - h_out) + cy;
    // std::cout << "center_scaled : x=" << cx_scaled << ", y=" << cy_scaled <<
    // "\n";

    const bool flip_lr = this->flip_lr_ & (this->rgen_() % 2);
    const bool flip_ud = this->flip_ud_ & (this->rgen_() % 2);
    const float global_brightness =
        ((this->rgen_() % 1001) * 0.001f * this->brightness_ * 2.0f) -
        this->brightness_;
    // std::cout << "global_brightness : " << global_brightness << "\n";
    const float global_contrast = std::exp((this->rgen_() % 1001) * 0.001f *
                                           std::log(this->contrast_) * 2.0f) /
                                  this->contrast_;
    // std::cout << "global_contrast : " << global_contrast << "\n";

    for (int ic = 0; ic < num_ch; ++ic) {
      const float ch_brightness =
          this->brightness_each_
              ? ((this->rgen_() % 1001) * 0.001f * this->brightness_ * 2.0f) -
                    this->brightness_
              : global_brightness;
      channel_brightness[ic] = ch_brightness - this->contrast_center_;
      // std::cout << "channel_brightness - 0.5 : " << channel_brightness[ic] <<
      // "\n";

      const float ch_contrast =
          this->contrast_each_
              ? std::exp((this->rgen_() % 1001) * 0.001f *
                         std::log(this->contrast_) * 2.0f) /
                    this->contrast_
              : global_contrast;
      channel_contrast[ic] = ch_contrast;
      // std::cout << "channel_contrast : " << channel_contrast[ic] << "\n";
    }

    const float distortion =
        std::exp(((this->rgen_() % 1001) * 0.001f * 2.0f * this->distortion_) -
                 this->distortion_) -
        1.0f;
    // std::cout << "distortion : " << distortion << "\n";
    const float noise = (this->rgen_() % 1001) * 0.001f * this->noise_;
    // std::cout << "noise : " << noise << "\n";

    // Pixel loop
    const float cos_theta = std::cos(angle);
    const float sin_theta = std::sin(angle);
    const float x_ax = (flip_lr ? -cos_theta : cos_theta) * i_scale_x;
    const float y_ax = (flip_lr ? sin_theta : -sin_theta) * i_scale_y;
    const float x_ay = (flip_ud ? -sin_theta : sin_theta) * i_scale_x;
    const float y_ay = (flip_ud ? -cos_theta : cos_theta) * i_scale_y;
    float x0_in =
        (cx_scaled * i_scale_x) - (x_ax * cx + y_ax * cy) - this->pad_[1];
    float y0_in =
        (cy_scaled * i_scale_y) - (x_ay * cx + y_ay * cy) - this->pad_[0];

    dim3 threads(32, 16);
    dim3 blocks((w_out - 1) / threads.x + 1, (h_out - 1) / threads.y + 1);
    for (int ic = 0; ic < num_ch; ++ic) {
      IAKernel<<<blocks, threads>>>(
          x_im + ch_size_in * ic, w_in, h_in, x0_in, y0_in,
          y_im + ch_size_out * ic, w_out, h_out, x_ax, y_ax, x_ay, y_ay,
          distortion, channel_brightness[ic], channel_contrast[ic],
          this->contrast_center_, (curandStateXORWOW_t *)state, noise);
      NBLA_CUDA_KERNEL_CHECK();
    }
  }
}

template <typename T>
void ImageAugmentationCuda<T>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  // Not supported
}
}

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
#include <nbla/cuda/function/interpolate.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

inline float compute_scale(int isize, int osize, bool align_corners) {
  return (osize <= 1) ? 0.0f : (align_corners ? float(isize - 1) / (osize - 1)
                                              : float(isize) / osize);
}

__device__ __forceinline__ float get_src_index(float scale, int dst_index,
                                               bool align_corners) {
  return align_corners ? scale * dst_index
                       : fmaxf(0.0f, scale * (float(dst_index) + 0.5f) - 0.5f);
}

template <typename T>
__global__ void kernel_linear_interpolate_2d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int iw, const int ih, const int ow, const int oh,
    const float sx, const float sy, const bool align_corners) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const int oy = index / ow;
    const int ox = index % ow;

    const auto fy = get_src_index(sy, oy, align_corners);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, align_corners);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
#define _I(y, x) ((y)*iw + (x))
      const T val0 = lx0 * src[_I(y1, x1)];
      const T val1 = lx1 * src[_I(y1, x2)];
      const T val2 = lx0 * src[_I(y2, x1)];
      const T val3 = lx1 * src[_I(y2, x2)];
#undef _I
      dst[oy * ow + ox] = ly0 * (val0 + val1) + ly1 * (val2 + val3);
    }
  }
}

template <typename T>
__global__ void kernel_linear_interpolate_3d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int iw, const int ih, const int id, const int ow,
    const int oh, const int od, const float sx, const float sy, const float sz,
    const bool align_corners) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const int oz = index / (ow * oh);
    const int oy = (index - oz * ow * oh) / ow;
    const int ox = (index - oz * ow * oh) % ow;

    const auto fz = get_src_index(sz, oz, align_corners);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = min(z1 + 1, id - 1);
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    const auto fy = get_src_index(sy, oy, align_corners);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, align_corners);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
#define _I(z, y, x) ((z)*ih * iw + (y)*iw + (x))
      const T val0 = lx0 * src[_I(z1, y1, x1)];
      const T val1 = lx1 * src[_I(z1, y1, x2)];
      const T val2 = lx0 * src[_I(z1, y2, x1)];
      const T val3 = lx1 * src[_I(z1, y2, x2)];
      const T val4 = lx0 * src[_I(z2, y1, x1)];
      const T val5 = lx1 * src[_I(z2, y1, x2)];
      const T val6 = lx0 * src[_I(z2, y2, x1)];
      const T val7 = lx1 * src[_I(z2, y2, x2)];
      const T val8 = ly0 * (val0 + val1) + ly1 * (val2 + val3);
      const T val9 = ly0 * (val4 + val5) + ly1 * (val6 + val7);
#undef _I
      dst[oz * oh * ow + oy * ow + ox] = lz0 * val8 + lz1 * val9;
    }
  }
}

template <typename T>
__global__ void kernel_linear_interpolate_2d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int iw, const int ih, const int ow, const int oh,
    const float sx, const float sy, const bool align_corners) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const int oy = index / ow;
    const int ox = index % ow;

    const auto fy = get_src_index(sy, oy, align_corners);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = (y1 < ih - 1) ? (y1 + 1) : y1;
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, align_corners);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = (x1 < iw - 1) ? (x1 + 1) : x1;
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T g = g_y[oy * ow + ox];
#define _I(y, x) ((y)*iw + (x))
      atomic_add(g_x + _I(y1, x1), ly0 * lx0 * g);
      atomic_add(g_x + _I(y1, x2), ly0 * lx1 * g);
      atomic_add(g_x + _I(y2, x1), ly1 * lx0 * g);
      atomic_add(g_x + _I(y2, x2), ly1 * lx1 * g);
#undef _I
    }
  }
}

template <typename T>
__global__ void kernel_linear_interpolate_3d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int iw, const int ih, const int id, const int ow,
    const int oh, const int od, const float sx, const float sy, const float sz,
    const bool align_corners) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const int oz = index / (ow * oh);
    const int oy = (index - oz * ow * oh) / ow;
    const int ox = (index - oz * ow * oh) % ow;

    const auto fz = get_src_index(sz, oz, align_corners);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = min(z1 + 1, id - 1);
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    const auto fy = get_src_index(sy, oy, align_corners);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = (y1 < ih - 1) ? (y1 + 1) : y1;
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, align_corners);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = (x1 < iw - 1) ? (x1 + 1) : x1;
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T g = g_y[oz * oh * ow + oy * ow + ox];
#define _I(z, y, x) ((z)*ih * iw + (y)*iw + (x))
      atomic_add(g_x + _I(z1, y1, x1), lz0 * ly0 * lx0 * g);
      atomic_add(g_x + _I(z1, y1, x2), lz0 * ly0 * lx1 * g);
      atomic_add(g_x + _I(z1, y2, x1), lz0 * ly1 * lx0 * g);
      atomic_add(g_x + _I(z1, y2, x2), lz0 * ly1 * lx1 * g);
      atomic_add(g_x + _I(z2, y1, x1), lz1 * ly0 * lx0 * g);
      atomic_add(g_x + _I(z2, y1, x2), lz1 * ly0 * lx1 * g);
      atomic_add(g_x + _I(z2, y2, x1), lz1 * ly1 * lx0 * g);
      atomic_add(g_x + _I(z2, y2, x2), lz1 * ly1 * lx1 * g);
#undef _I
    }
  }
}

template <typename T>
__global__ void kernel_nearest_interpolate_2d(const int dst_inner_size, T *dst,
                                              const int src_inner_size,
                                              const T *src, int outer_size,
                                              const int iw, const int ih,
                                              const int ow, const int oh,
                                              const float sx, const float sy) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const int oy = index / ow;
    const int ox = index % ow;

    const auto iy = min(static_cast<int>(sy * (oy + 0.5f)), ih - 1);
    const auto ix = min(static_cast<int>(sx * (ox + 0.5f)), iw - 1);

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      dst[oy * ow + ox] = src[iy * iw + ix];
    }
  }
}

template <typename T>
__global__ void kernel_nearest_interpolate_3d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int iw, const int ih, const int id, const int ow,
    const int oh, const int od, const float sx, const float sy,
    const float sz) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const int oz = index / (ow * oh);
    const int oy = (index - oz * ow * oh) / ow;
    const int ox = (index - oz * ow * oh) % ow;

    const auto iz = min(static_cast<int>(sz * (oz + 0.5f)), id - 1);
    const auto iy = min(static_cast<int>(sy * (oy + 0.5f)), ih - 1);
    const auto ix = min(static_cast<int>(sx * (ox + 0.5f)), iw - 1);

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      dst[oz * oh * ow + oy * ow + ox] = src[iz * ih * iw + iy * iw + ix];
    }
  }
}

template <typename T>
__global__ void kernel_nearest_interpolate_2d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int iw, const int ih, const int ow, const int oh,
    const float sx, const float sy) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const int oy = index / ow;
    const int ox = index % ow;

    const auto iy = min(static_cast<int>(sy * (oy + 0.5f)), ih - 1);
    const auto ix = min(static_cast<int>(sx * (ox + 0.5f)), iw - 1);

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T grad = g_y[oy * ow + ox];
      atomic_add(g_x + iy * iw + ix, grad);
    }
  }
}

template <typename T>
__global__ void kernel_nearest_interpolate_3d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int iw, const int ih, const int id, const int ow,
    const int oh, const int od, const float sx, const float sy,
    const float sz) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const int oz = index / (ow * oh);
    const int oy = (index - oz * ow * oh) / ow;
    const int ox = (index - oz * ow * oh) % ow;

    const auto iz = min(static_cast<int>(sz * (oz + 0.5f)), id - 1);
    const auto iy = min(static_cast<int>(sy * (oy + 0.5f)), ih - 1);
    const auto ix = min(static_cast<int>(sx * (ox + 0.5f)), iw - 1);

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T grad = g_y[oz * oh * ow + oy * ow + ox];
      atomic_add(g_x + iz * ih * iw + iy * iw + ix, grad);
    }
  }
}

template <typename T>
void InterpolateCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  auto src = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto dst = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];

  if (this->output_size_.size() == 2) {
    const int src_inner_size = iw * ih;
    const int dst_inner_size = ow * oh;
    const int outer_size = inputs[0]->size() / src_inner_size;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel_linear_interpolate_2d, dst_inner_size, dst, src_inner_size,
          src, outer_size, iw, ih, ow, oh, sx, sy, this->align_corners_);
    } else if (this->mode_ == "nearest") {
      const float sx = iw / static_cast<float>(ow);
      const float sy = ih / static_cast<float>(oh);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nearest_interpolate_2d,
                                     dst_inner_size, dst, src_inner_size, src,
                                     outer_size, iw, ih, ow, oh, sx, sy);
    }
  }

  else if (this->output_size_.size() == 3) {
    const int id = inputs[0]->shape()[ndim - 3];
    const int od = outputs[0]->shape()[ndim - 3];
    const int src_inner_size = iw * ih * id;
    const int dst_inner_size = ow * oh * od;
    const int outer_size = inputs[0]->size() / src_inner_size;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      const float sz = compute_scale(id, od, this->align_corners_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_linear_interpolate_3d,
                                     dst_inner_size, dst, src_inner_size, src,
                                     outer_size, iw, ih, id, ow, oh, od, sx, sy,
                                     sz, this->align_corners_);
    } else if (this->mode_ == "nearest") {
      const float sx = iw / static_cast<float>(ow);
      const float sy = ih / static_cast<float>(oh);
      const float sz = id / static_cast<float>(od);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel_nearest_interpolate_3d, dst_inner_size, dst, src_inner_size,
          src, outer_size, iw, ih, id, ow, oh, od, sx, sy, sz);
    }
  }
}

template <typename T>
void InterpolateCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  auto g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);

  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];

  if (this->output_size_.size() == 2) {
    const int g_x_inner_size = iw * ih;
    const int g_y_inner_size = ow * oh;
    const int outer_size = inputs[0]->size() / g_x_inner_size;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_linear_interpolate_2d_backward,
                                     g_y_inner_size, g_y, g_x_inner_size, g_x,
                                     outer_size, iw, ih, ow, oh, sx, sy,
                                     this->align_corners_);
    } else if (this->mode_ == "nearest") {
      const float sx = iw / static_cast<float>(ow);
      const float sy = ih / static_cast<float>(oh);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_nearest_interpolate_2d_backward,
                                     g_y_inner_size, g_y, g_x_inner_size, g_x,
                                     outer_size, iw, ih, ow, oh, sx, sy);
    }
  }

  else if (this->output_size_.size() == 3) {
    const int id = inputs[0]->shape()[ndim - 3];
    const int od = outputs[0]->shape()[ndim - 3];
    const int g_x_inner_size = iw * ih * id;
    const int g_y_inner_size = ow * oh * od;
    const int outer_size = inputs[0]->size() / g_x_inner_size;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      const float sz = compute_scale(id, od, this->align_corners_);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_linear_interpolate_3d_backward,
                                     g_y_inner_size, g_y, g_x_inner_size, g_x,
                                     outer_size, iw, ih, id, ow, oh, od, sx, sy,
                                     sz, this->align_corners_);
    } else if (this->mode_ == "nearest") {
      const float sx = iw / static_cast<float>(ow);
      const float sy = ih / static_cast<float>(oh);
      const float sz = id / static_cast<float>(od);
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel_nearest_interpolate_3d_backward, g_y_inner_size, g_y,
          g_x_inner_size, g_x, outer_size, iw, ih, id, ow, oh, od, sx, sy, sz);
    }
  }
}
}

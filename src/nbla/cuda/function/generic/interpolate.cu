// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

namespace nbla {

inline float compute_scale(int isize, int osize, bool align_corners) {
  return (osize <= 1) ? 0.0f : (align_corners ? float(isize - 1) / (osize - 1)
                                              : float(isize) / osize);
}

inline float compute_scale_for_nn(int isize, int osize, bool align_corners,
                                  bool half_pixel_for_nn) {
  return half_pixel_for_nn ? isize / static_cast<float>(osize)
                           : compute_scale(isize, osize, align_corners);
}

__device__ __forceinline__ float get_src_index(float scale, int dst_index,
                                               bool half_pixel) {
  return half_pixel ? fmaxf(0.0f, scale * (float(dst_index) + 0.5f) - 0.5f)
                    : scale * dst_index;
}

__device__ __forceinline__ float get_src_index_for_nn(float scale,
                                                      int dst_index,
                                                      bool half_pixel,
                                                      bool half_pixel_for_nn) {
  return half_pixel_for_nn ? scale * (dst_index + 0.5f)
                           : get_src_index(scale, dst_index, half_pixel);
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_1d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int ishape, const int istride, const int ostride,
    const float sx, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_2d(index, ostride);
    const auto oc = channel_last ? nd_index.y : 0;
    const auto ox = nd_index.x;

    const auto iw = ishape;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_x1 = make_int2(x1, oc);
    const auto nd_idx_x2 = make_int2(x2, oc);

    const auto idx_lx0 = device_2d_to_flat(nd_idx_x1, istride);
    const auto idx_lx1 = device_2d_to_flat(nd_idx_x2, istride);

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      const T val0 = lx0 * src[idx_lx0];
      const T val1 = lx1 * src[idx_lx1];
      dst[index] = val0 + val1;
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_2d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int2 ishape, const int2 istride, const int2 ostride,
    const float sx, const float sy, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_3d(index, ostride);
    const auto oc = channel_last ? nd_index.z : 0;
    const auto oy = nd_index.x;
    const auto ox = nd_index.y;

    const auto ih = ishape.x;
    const auto iw = ishape.y;

    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_y1x1 = make_int3(y1, x1, oc);
    const auto nd_idx_y1x2 = make_int3(y1, x2, oc);
    const auto nd_idx_y2x1 = make_int3(y2, x1, oc);
    const auto nd_idx_y2x2 = make_int3(y2, x2, oc);

    const auto idx_ly0x0 = device_3d_to_flat(nd_idx_y1x1, istride);
    const auto idx_ly0x1 = device_3d_to_flat(nd_idx_y1x2, istride);
    const auto idx_ly1x0 = device_3d_to_flat(nd_idx_y2x1, istride);
    const auto idx_ly1x1 = device_3d_to_flat(nd_idx_y2x2, istride);

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      const T val0 = lx0 * src[idx_ly0x0];
      const T val1 = lx1 * src[idx_ly0x1];
      const T val2 = lx0 * src[idx_ly1x0];
      const T val3 = lx1 * src[idx_ly1x1];
      dst[index] = ly0 * (val0 + val1) + ly1 * (val2 + val3);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_3d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int3 ishape, const int3 istride, const int3 ostride,
    const float sx, const float sy, const float sz, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_4d(index, ostride);
    const auto oc = channel_last ? nd_index.w : 0;
    const auto oz = nd_index.x;
    const auto oy = nd_index.y;
    const auto ox = nd_index.z;

    const auto id = ishape.x;
    const auto ih = ishape.y;
    const auto iw = ishape.z;

    const auto fz = get_src_index(sz, oz, half_pixel);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = min(z1 + 1, id - 1);
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_z1y1x1 = make_int4(z1, y1, x1, oc);
    const auto nd_idx_z1y1x2 = make_int4(z1, y1, x2, oc);
    const auto nd_idx_z1y2x1 = make_int4(z1, y2, x1, oc);
    const auto nd_idx_z1y2x2 = make_int4(z1, y2, x2, oc);
    const auto nd_idx_z2y1x1 = make_int4(z2, y1, x1, oc);
    const auto nd_idx_z2y1x2 = make_int4(z2, y1, x2, oc);
    const auto nd_idx_z2y2x1 = make_int4(z2, y2, x1, oc);
    const auto nd_idx_z2y2x2 = make_int4(z2, y2, x2, oc);

    const auto idx_lz0y0x0 = device_4d_to_flat(nd_idx_z1y1x1, istride);
    const auto idx_lz0y0x1 = device_4d_to_flat(nd_idx_z1y1x2, istride);
    const auto idx_lz0y1x0 = device_4d_to_flat(nd_idx_z1y2x1, istride);
    const auto idx_lz0y1x1 = device_4d_to_flat(nd_idx_z1y2x2, istride);
    const auto idx_lz1y0x0 = device_4d_to_flat(nd_idx_z2y1x1, istride);
    const auto idx_lz1y0x1 = device_4d_to_flat(nd_idx_z2y1x2, istride);
    const auto idx_lz1y1x0 = device_4d_to_flat(nd_idx_z2y2x1, istride);
    const auto idx_lz1y1x1 = device_4d_to_flat(nd_idx_z2y2x2, istride);

    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      const T val0 = lx0 * src[idx_lz0y0x0];
      const T val1 = lx1 * src[idx_lz0y0x1];
      const T val2 = lx0 * src[idx_lz0y1x0];
      const T val3 = lx1 * src[idx_lz0y1x1];
      const T val4 = lx0 * src[idx_lz1y0x0];
      const T val5 = lx1 * src[idx_lz1y0x1];
      const T val6 = lx0 * src[idx_lz1y1x0];
      const T val7 = lx1 * src[idx_lz1y1x1];
      const T val8 = ly0 * (val0 + val1) + ly1 * (val2 + val3);
      const T val9 = ly0 * (val4 + val5) + ly1 * (val6 + val7);
      dst[index] = lz0 * val8 + lz1 * val9;
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_1d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int ishape, const int istride, const int ostride,
    const float sx, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_2d(index, ostride);
    const auto oc = channel_last ? nd_index.y : 0;
    const auto ox = nd_index.x;

    const auto iw = ishape;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_x1 = make_int2(x1, oc);
    const auto nd_idx_x2 = make_int2(x2, oc);

    const auto idx_lx1 = device_2d_to_flat(nd_idx_x1, istride);
    const auto idx_lx2 = device_2d_to_flat(nd_idx_x2, istride);

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T g = g_y[index];
      atomic_add(g_x + idx_lx1, lx0 * g);
      atomic_add(g_x + idx_lx2, lx1 * g);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_2d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int2 ishape, const int2 istride, const int2 ostride,
    const float sx, const float sy, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_3d(index, ostride);
    const auto oc = channel_last ? nd_index.z : 0;
    const auto oy = nd_index.x;
    const auto ox = nd_index.y;

    const auto ih = ishape.x;
    const auto iw = ishape.y;

    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_y1x1 = make_int3(y1, x1, oc);
    const auto nd_idx_y1x2 = make_int3(y1, x2, oc);
    const auto nd_idx_y2x1 = make_int3(y2, x1, oc);
    const auto nd_idx_y2x2 = make_int3(y2, x2, oc);

    const auto idx_ly0x0 = device_3d_to_flat(nd_idx_y1x1, istride);
    const auto idx_ly0x1 = device_3d_to_flat(nd_idx_y1x2, istride);
    const auto idx_ly1x0 = device_3d_to_flat(nd_idx_y2x1, istride);
    const auto idx_ly1x1 = device_3d_to_flat(nd_idx_y2x2, istride);
    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T g = g_y[index];
      atomic_add(g_x + idx_ly0x0, ly0 * lx0 * g);
      atomic_add(g_x + idx_ly0x1, ly0 * lx1 * g);
      atomic_add(g_x + idx_ly1x0, ly1 * lx0 * g);
      atomic_add(g_x + idx_ly1x1, ly1 * lx1 * g);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_linear_interpolate_3d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int3 ishape, const int3 istride, const int3 ostride,
    const float sx, const float sy, const float sz, const bool half_pixel) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_4d(index, ostride);
    const auto oc = channel_last ? nd_index.w : 0;
    const auto oz = nd_index.x;
    const auto oy = nd_index.y;
    const auto ox = nd_index.z;

    const auto id = ishape.x;
    const auto ih = ishape.y;
    const auto iw = ishape.z;

    const auto fz = get_src_index(sz, oz, half_pixel);
    const auto z1 = static_cast<int>(fz);
    const auto z2 = min(z1 + 1, id - 1);
    const auto lz1 = static_cast<T>(fz - z1);
    const auto lz0 = static_cast<T>(1) - lz1;

    const auto fy = get_src_index(sy, oy, half_pixel);
    const auto y1 = static_cast<int>(fy);
    const auto y2 = min(y1 + 1, ih - 1);
    const auto ly1 = static_cast<T>(fy - y1);
    const auto ly0 = static_cast<T>(1) - ly1;

    const auto fx = get_src_index(sx, ox, half_pixel);
    const auto x1 = static_cast<int>(fx);
    const auto x2 = min(x1 + 1, iw - 1);
    const auto lx1 = static_cast<T>(fx - x1);
    const auto lx0 = static_cast<T>(1) - lx1;

    const auto nd_idx_z1y1x1 = make_int4(z1, y1, x1, oc);
    const auto nd_idx_z1y1x2 = make_int4(z1, y1, x2, oc);
    const auto nd_idx_z1y2x1 = make_int4(z1, y2, x1, oc);
    const auto nd_idx_z1y2x2 = make_int4(z1, y2, x2, oc);
    const auto nd_idx_z2y1x1 = make_int4(z2, y1, x1, oc);
    const auto nd_idx_z2y1x2 = make_int4(z2, y1, x2, oc);
    const auto nd_idx_z2y2x1 = make_int4(z2, y2, x1, oc);
    const auto nd_idx_z2y2x2 = make_int4(z2, y2, x2, oc);

    const auto idx_lz0y0x0 = device_4d_to_flat(nd_idx_z1y1x1, istride);
    const auto idx_lz0y0x1 = device_4d_to_flat(nd_idx_z1y1x2, istride);
    const auto idx_lz0y1x0 = device_4d_to_flat(nd_idx_z1y2x1, istride);
    const auto idx_lz0y1x1 = device_4d_to_flat(nd_idx_z1y2x2, istride);
    const auto idx_lz1y0x0 = device_4d_to_flat(nd_idx_z2y1x1, istride);
    const auto idx_lz1y0x1 = device_4d_to_flat(nd_idx_z2y1x2, istride);
    const auto idx_lz1y1x0 = device_4d_to_flat(nd_idx_z2y2x1, istride);
    const auto idx_lz1y1x1 = device_4d_to_flat(nd_idx_z2y2x2, istride);

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      const T g = g_y[index];
      atomic_add(g_x + idx_lz0y0x0, lz0 * ly0 * lx0 * g);
      atomic_add(g_x + idx_lz0y0x1, lz0 * ly0 * lx1 * g);
      atomic_add(g_x + idx_lz0y1x0, lz0 * ly1 * lx0 * g);
      atomic_add(g_x + idx_lz0y1x1, lz0 * ly1 * lx1 * g);
      atomic_add(g_x + idx_lz1y0x0, lz1 * ly0 * lx0 * g);
      atomic_add(g_x + idx_lz1y0x1, lz1 * ly0 * lx1 * g);
      atomic_add(g_x + idx_lz1y1x0, lz1 * ly1 * lx0 * g);
      atomic_add(g_x + idx_lz1y1x1, lz1 * ly1 * lx1 * g);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_1d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int ishape, const int istride, const int ostride,
    const float sx, const bool half_pixel, const bool half_pixel_for_nn) {

  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_2d(index, ostride);
    const auto oc = channel_last ? nd_index.y : 0;
    const auto ox = nd_index.x;

    const auto iw = ishape;
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_x = make_int2(ix, oc);
    const auto idx_x = device_2d_to_flat(nd_idx_x, istride);
    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      dst[index] = src[idx_x];
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_2d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int2 ishape, const int2 istride, const int2 ostride,
    const float sx, const float sy, const bool half_pixel,
    const bool half_pixel_for_nn) {
  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_3d(index, ostride);
    const auto oc = channel_last ? nd_index.z : 0;
    const auto oy = nd_index.x;
    const auto ox = nd_index.y;

    const auto ih = ishape.x;
    const auto iw = ishape.y;

    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto iy = min(static_cast<int>(fy), ih - 1);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_yx = make_int3(iy, ix, oc);
    const auto idx_yx = device_3d_to_flat(nd_idx_yx, istride);
    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      dst[index] = src[idx_yx];
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_3d(
    const int dst_inner_size, T *dst, const int src_inner_size, const T *src,
    int outer_size, const int3 ishape, const int3 istride, const int3 ostride,
    const float sx, const float sy, const float sz, const bool half_pixel,
    const bool half_pixel_for_nn) {
  NBLA_CUDA_KERNEL_LOOP(index, dst_inner_size) {
    const auto nd_index = device_flat_to_4d(index, ostride);
    const auto oc = channel_last ? nd_index.w : 0;
    const auto oz = nd_index.x;
    const auto oy = nd_index.y;
    const auto ox = nd_index.z;

    const auto id = ishape.x;
    const auto ih = ishape.y;
    const auto iw = ishape.z;

    const auto fz = get_src_index_for_nn(sz, oz, half_pixel, half_pixel_for_nn);
    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto iz = min(static_cast<int>(fz), id - 1);
    const auto iy = min(static_cast<int>(fy), ih - 1);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_zyx = make_int4(iz, iy, ix, oc);
    const auto idx_zyx = device_4d_to_flat(nd_idx_zyx, istride);
    for (; outer_size--; src += src_inner_size, dst += dst_inner_size) {
      dst[index] = src[idx_zyx];
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_1d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int ishape, const int istride, const int ostride,
    const float sx, const bool half_pixel, const bool half_pixel_for_nn) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_2d(index, ostride);
    const auto oc = channel_last ? nd_index.y : 0;
    const auto ox = nd_index.x;

    const auto iw = ishape;
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_x = make_int2(ix, oc);
    const auto idx_x = device_2d_to_flat(nd_idx_x, istride);
    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      atomic_add(g_x + idx_x, g_y[index]);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_2d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int2 ishape, const int2 istride, const int2 ostride,
    const float sx, const float sy, const bool half_pixel,
    const bool half_pixel_for_nn) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_3d(index, ostride);
    const auto oc = channel_last ? nd_index.z : 0;
    const auto oy = nd_index.x;
    const auto ox = nd_index.y;

    const auto ih = ishape.x;
    const auto iw = ishape.y;

    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto iy = min(static_cast<int>(fy), ih - 1);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_yx = make_int3(iy, ix, oc);
    const auto idx_yx = device_3d_to_flat(nd_idx_yx, istride);

    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      atomic_add(g_x + idx_yx, g_y[index]);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_nearest_interpolate_3d_backward(
    const int g_y_inner_size, const T *g_y, const int g_x_inner_size, T *g_x,
    int outer_size, const int3 ishape, const int3 istride, const int3 ostride,
    const float sx, const float sy, const float sz, const bool half_pixel,
    const bool half_pixel_for_nn) {

  NBLA_CUDA_KERNEL_LOOP(index, g_y_inner_size) {
    const auto nd_index = device_flat_to_4d(index, ostride);
    const auto oc = channel_last ? nd_index.w : 0;
    const auto oz = nd_index.x;
    const auto oy = nd_index.y;
    const auto ox = nd_index.z;

    const auto id = ishape.x;
    const auto ih = ishape.y;
    const auto iw = ishape.z;

    const auto fz = get_src_index_for_nn(sz, oz, half_pixel, half_pixel_for_nn);
    const auto fy = get_src_index_for_nn(sy, oy, half_pixel, half_pixel_for_nn);
    const auto fx = get_src_index_for_nn(sx, ox, half_pixel, half_pixel_for_nn);
    const auto iz = min(static_cast<int>(fz), id - 1);
    const auto iy = min(static_cast<int>(fy), ih - 1);
    const auto ix = min(static_cast<int>(fx), iw - 1);

    const auto nd_idx_zyx = make_int4(iz, iy, ix, oc);
    const auto idx_zyx = device_4d_to_flat(nd_idx_zyx, istride);
    for (; outer_size--; g_x += g_x_inner_size, g_y += g_y_inner_size) {
      atomic_add(g_x + idx_zyx, g_y[index]);
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

  if (this->output_size_.size() == 1) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];

    const int src_inner_size = this->channel_last_ ? ic * iw : iw;
    const int dst_inner_size = this->channel_last_ ? oc * ow : ow;
    const int outer_size = inputs[0]->size() / src_inner_size;
    const auto ishape = iw;
    const auto istride = this->channel_last_ ? ic : 1;
    const auto ostride = this->channel_last_ ? oc : 1;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_1d<Tcu, true>
                        : kernel_linear_interpolate_1d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, dst_inner_size, dst,
                                     src_inner_size, src, outer_size, ishape,
                                     istride, ostride, sx, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_1d<Tcu, true>
                        : kernel_nearest_interpolate_1d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, dst_inner_size, dst, src_inner_size, src, outer_size, ishape,
          istride, ostride, sx, this->half_pixel_, this->half_pixel_for_nn_);
    }
  }

  else if (this->output_size_.size() == 2) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 3];
    const int ih = this->channel_last_ ? inputs[0]->shape()[ndim - 3]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 3];
    const int oh = this->channel_last_ ? outputs[0]->shape()[ndim - 3]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];

    const int src_inner_size = this->channel_last_ ? ic * iw * ih : iw * ih;
    const int dst_inner_size = this->channel_last_ ? oc * ow * oh : ow * oh;
    const int outer_size = inputs[0]->size() / src_inner_size;
    const auto ishape = make_int2(ih, iw);
    const auto istride =
        this->channel_last_ ? make_int2(iw * ic, ic) : make_int2(iw, 1);
    const auto ostride =
        this->channel_last_ ? make_int2(ow * oc, oc) : make_int2(ow, 1);
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_2d<Tcu, true>
                        : kernel_linear_interpolate_2d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, dst_inner_size, dst, src_inner_size, src, outer_size, ishape,
          istride, ostride, sx, sy, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sy = compute_scale_for_nn(ih, oh, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_2d<Tcu, true>
                        : kernel_nearest_interpolate_2d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, dst_inner_size, dst, src_inner_size, src, outer_size, ishape,
          istride, ostride, sx, sy, this->half_pixel_,
          this->half_pixel_for_nn_);
    }
  }

  else if (this->output_size_.size() == 3) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 4];
    const int id = this->channel_last_ ? inputs[0]->shape()[ndim - 4]
                                       : inputs[0]->shape()[ndim - 3];
    const int ih = this->channel_last_ ? inputs[0]->shape()[ndim - 3]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 4];
    const int od = this->channel_last_ ? outputs[0]->shape()[ndim - 4]
                                       : outputs[0]->shape()[ndim - 3];
    const int oh = this->channel_last_ ? outputs[0]->shape()[ndim - 3]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];

    const int src_inner_size =
        this->channel_last_ ? ic * iw * ih * id : iw * ih * id;
    const int dst_inner_size =
        this->channel_last_ ? oc * ow * oh * od : ow * oh * od;
    const int outer_size = inputs[0]->size() / src_inner_size;
    const auto ishape = make_int3(id, ih, iw);
    const auto istride = this->channel_last_
                             ? make_int3(ih * iw * ic, iw * ic, ic)
                             : make_int3(ih * iw, iw, 1);
    const auto ostride = this->channel_last_
                             ? make_int3(oh * ow * oc, ow * oc, oc)
                             : make_int3(oh * ow, ow, 1);
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      const float sz = compute_scale(id, od, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_3d<Tcu, true>
                        : kernel_linear_interpolate_3d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, dst_inner_size, dst, src_inner_size, src, outer_size, ishape,
          istride, ostride, sx, sy, sz, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sy = compute_scale_for_nn(ih, oh, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sz = compute_scale_for_nn(id, od, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_3d<Tcu, true>
                        : kernel_nearest_interpolate_3d<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, dst_inner_size, dst, src_inner_size, src, outer_size, ishape,
          istride, ostride, sx, sy, sz, this->half_pixel_,
          this->half_pixel_for_nn_);
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

  if (this->output_size_.size() == 1) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];

    const int g_x_inner_size = this->channel_last_ ? ic * iw : iw;
    const int g_y_inner_size = this->channel_last_ ? oc * ow : ow;
    const int outer_size = inputs[0]->size() / g_x_inner_size;
    const auto ishape = iw;
    const auto istride = this->channel_last_ ? ic : 1;
    const auto ostride = this->channel_last_ ? oc : 1;
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_1d_backward<Tcu, true>
                        : kernel_linear_interpolate_1d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, g_y_inner_size, g_y,
                                     g_x_inner_size, g_x, outer_size, ishape,
                                     istride, ostride, sx, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_1d_backward<Tcu, true>
                        : kernel_nearest_interpolate_1d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, g_y_inner_size, g_y, g_x_inner_size, g_x, outer_size, ishape,
          istride, ostride, sx, this->half_pixel_, this->half_pixel_for_nn_);
    }
  }

  else if (this->output_size_.size() == 2) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 3];
    const int ih = this->channel_last_ ? inputs[0]->shape()[ndim - 3]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 3];
    const int oh = this->channel_last_ ? outputs[0]->shape()[ndim - 3]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];
    const int g_x_inner_size = this->channel_last_ ? ic * iw * ih : iw * ih;
    const int g_y_inner_size = this->channel_last_ ? oc * ow * oh : ow * oh;
    const int outer_size = inputs[0]->size() / g_x_inner_size;
    const auto ishape = make_int2(ih, iw);
    const auto istride =
        this->channel_last_ ? make_int2(iw * ic, ic) : make_int2(iw, 1);
    const auto ostride =
        this->channel_last_ ? make_int2(ow * oc, oc) : make_int2(ow, 1);
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_2d_backward<Tcu, true>
                        : kernel_linear_interpolate_2d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, g_y_inner_size, g_y, g_x_inner_size, g_x, outer_size, ishape,
          istride, ostride, sx, sy, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sy = compute_scale_for_nn(ih, oh, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_2d_backward<Tcu, true>
                        : kernel_nearest_interpolate_2d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, g_y_inner_size, g_y, g_x_inner_size, g_x, outer_size, ishape,
          istride, ostride, sx, sy, this->half_pixel_,
          this->half_pixel_for_nn_);
    }
  }

  else if (this->output_size_.size() == 3) {
    const int ic = this->channel_last_ ? inputs[0]->shape()[ndim - 1]
                                       : inputs[0]->shape()[ndim - 4];
    const int id = this->channel_last_ ? inputs[0]->shape()[ndim - 4]
                                       : inputs[0]->shape()[ndim - 3];
    const int ih = this->channel_last_ ? inputs[0]->shape()[ndim - 3]
                                       : inputs[0]->shape()[ndim - 2];
    const int iw = this->channel_last_ ? inputs[0]->shape()[ndim - 2]
                                       : inputs[0]->shape()[ndim - 1];
    const int oc = this->channel_last_ ? outputs[0]->shape()[ndim - 1]
                                       : outputs[0]->shape()[ndim - 4];
    const int od = this->channel_last_ ? outputs[0]->shape()[ndim - 4]
                                       : outputs[0]->shape()[ndim - 3];
    const int oh = this->channel_last_ ? outputs[0]->shape()[ndim - 3]
                                       : outputs[0]->shape()[ndim - 2];
    const int ow = this->channel_last_ ? outputs[0]->shape()[ndim - 2]
                                       : outputs[0]->shape()[ndim - 1];

    const int g_x_inner_size =
        this->channel_last_ ? ic * iw * ih * id : iw * ih * id;
    const int g_y_inner_size =
        this->channel_last_ ? oc * ow * oh * od : ow * oh * od;
    const int outer_size = inputs[0]->size() / g_x_inner_size;
    const auto ishape = make_int3(id, ih, iw);
    const auto istride = this->channel_last_
                             ? make_int3(ih * iw * ic, iw * ic, ic)
                             : make_int3(ih * iw, iw, 1);
    const auto ostride = this->channel_last_
                             ? make_int3(oh * ow * oc, ow * oc, oc)
                             : make_int3(oh * ow, ow, 1);
    if (this->mode_ == "linear") {
      const float sx = compute_scale(iw, ow, this->align_corners_);
      const float sy = compute_scale(ih, oh, this->align_corners_);
      const float sz = compute_scale(id, od, this->align_corners_);
      auto kernel = this->channel_last_
                        ? kernel_linear_interpolate_3d_backward<Tcu, true>
                        : kernel_linear_interpolate_3d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, g_y_inner_size, g_y, g_x_inner_size, g_x, outer_size, ishape,
          istride, ostride, sx, sy, sz, this->half_pixel_);
    } else if (this->mode_ == "nearest") {
      const float sx = compute_scale_for_nn(iw, ow, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sy = compute_scale_for_nn(ih, oh, this->align_corners_,
                                            this->half_pixel_for_nn_);
      const float sz = compute_scale_for_nn(id, od, this->align_corners_,
                                            this->half_pixel_for_nn_);
      auto kernel = this->channel_last_
                        ? kernel_nearest_interpolate_3d_backward<Tcu, true>
                        : kernel_nearest_interpolate_3d_backward<Tcu, false>;
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          kernel, g_y_inner_size, g_y, g_x_inner_size, g_x, outer_size, ishape,
          istride, ostride, sx, sy, sz, this->half_pixel_,
          this->half_pixel_for_nn_);
    }
  }
}
}

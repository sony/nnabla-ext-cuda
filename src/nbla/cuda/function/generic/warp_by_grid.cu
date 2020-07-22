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
#include <nbla/cuda/function/warp_by_grid.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, bool align_corners = false>
__forceinline__ __device__ T unnormalize_grid_with(T s, const int S) {
  if (align_corners) {
    // [-1, 1] <--> [0, S - 1]
    return (s + T(1)) * (S - T(1)) / T(2);
  } else {
    // [-1, 1] <--> [-0.5, S - 0.5] = [0 - 0.5, S - 1 + 0.5]
    return ((s + T(1)) * S - T(1)) / T(2);
  }
}

template <typename T>
__forceinline__ __device__ T get_src_findex_with_zero_pad(const T s,
                                                          const int S) {
  return s;
}

template <typename T>
__forceinline__ __device__ T get_src_findex_with_repeat_pad(const T s,
                                                            const int S) {
  if (s < 0) {
    return 0;
  } else if (s > S - 1) {
    return S - 1;
  } else {
    return s;
  }
}

template <typename T>
__forceinline__ __device__ T reflect(const T s, const int L, const int U) {
  auto len = (U - L);
  if (s < L) {
    auto d = L - s;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    auto r = d - n * len;
    if (n % 2 == 0) {
      return L + r;
    } else {
      return U - r;
    }
  } else if (s > U) {
    auto d = s - U;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    auto r = d - n * len;
    if (n % 2 == 0) {
      return U - r;
    } else {
      return L + r;
    }
  } else {
    return s;
  }
}

template <typename T, bool align_corners = false>
__forceinline__ __device__ T get_src_findex_with_reflect_pad(T s, const int S) {
  if (align_corners) {
    return reflect(s, T(0), T(S - 1));
  } else {
    // address the borders {-0.5, S - 0.5} condition by two multiplication
    auto sf = reflect(T(2) * s, T(-1), T(2) * T(S) - T(1));
    sf = sf * T(0.5);
    sf = get_src_findex_with_repeat_pad(sf, S);
    return sf;
  }
}

template <typename T>
__forceinline__ __device__ T get_grad_coef_with_zero_pad(const T s,
                                                         const int S) {
  return T(1);
}

template <typename T>
__forceinline__ __device__ T get_grad_coef_with_repeat_pad(const T s,
                                                           const int S) {
  if (s <= 0) {
    return 0;
  } else if (s >= S - 1) {
    return 0;
  } else {
    return 1;
  }
}

template <typename T>
__forceinline__ __device__ T reflect_coef(const T s, const int L, const int U) {
  auto len = (U - L);
  if (s < L) {
    auto d = L - s;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    if (n % 2 == 0) {
      return T(-1);
    } else {
      return T(1);
    }
  } else if (s > U) {
    auto d = s - U;
    auto nf = d / len;
    auto n = static_cast<int>(nf);
    if (n % 2 == 0) {
      return T(-1);
    } else {
      return T(1);
    }
  } else {
    return T(1);
  }
}

template <typename T, bool align_corners = false>
__forceinline__ __device__ T get_grad_coef_with_reflect_pad(T s, const int S) {
  if (align_corners) {
    return reflect_coef(s, T(0), T(S - 1));
  } else {
    // address the borders {-0.5, S - 0.5} condition by two multiplication
    auto coef = reflect_coef(T(2) * s, T(-1), T(2) * T(S) - T(1));
    auto sf = reflect(T(2) * s, T(-1), T(2) * T(S) - T(1));
    sf = sf * T(0.5);
    coef *= get_grad_coef_with_repeat_pad(sf, S);
    return coef;
  }
}

template <typename T, bool channel_last = false>
__forceinline__ __device__ T get_pixel_value_2d(const T *input, int b, int c,
                                                int h, int w, const int H,
                                                const int W, const int2 istride,
                                                const int iisize) {
  if ((h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto ind_index = channel_last ? make_int3(h, w, c) : make_int3(c, h, w);
    auto iidx = device_3d_to_flat(ind_index, istride);
    auto b_iidx = iidx + b * iisize;
    return input[b_iidx];
  } else {
    return T(0);
  }
}

template <typename T, bool channel_last = false>
__forceinline__ __device__ T get_pixel_value_3d(const T *input, int b, int c,
                                                int d, int h, int w,
                                                const int D, const int H,
                                                const int W, const int3 istride,
                                                const int iisize) {
  if ((d >= 0 && d < D) && (h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto ind_index =
        channel_last ? make_int4(d, h, w, c) : make_int4(c, d, h, w);
    auto iidx = device_4d_to_flat(ind_index, istride);
    auto b_iidx = iidx + b * iisize;
    return input[b_iidx];
  } else {
    return T(0);
  }
}

template <typename T, bool channel_last = false>
__forceinline__ __device__ void
backward_data_2d(T *igrad, const T ograd, const T p, const T q, int b, int c,
                 int h, int w, const int H, const int W, const int2 istride,
                 const int iisize) {
  if ((h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto ind_index = channel_last ? make_int3(h, w, c) : make_int3(c, h, w);
    auto iidx = device_3d_to_flat(ind_index, istride);
    auto b_iidx = iidx + b * iisize;
    atomic_add(igrad + b_iidx, ograd * p * q);
  }
}

template <typename T, bool channel_last = false>
__forceinline__ __device__ void
backward_data_3d(T *igrad, const T ograd, const T p, const T q, const T r,
                 int b, int c, int d, int h, int w, const int D, const int H,
                 const int W, const int3 istride, const int iisize) {
  if ((d >= 0 && d < D) && (h >= 0 && h < H) && (w >= 0 && w < W)) {
    auto ind_index =
        channel_last ? make_int4(d, h, w, c) : make_int4(c, d, h, w);
    auto iidx = device_4d_to_flat(ind_index, istride);
    auto b_iidx = iidx + b * iisize;
    atomic_add(igrad + b_iidx, ograd * p * q * r);
  }
}

/*
  Forward implementations
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_forward_2d(
    const int oisize, const int iisize, const int gisize, T *output,
    const T *input, const T *grid, const int3 ishape, const int2 istride,
    const int2 gstride, const int2 ostride, const int B) {
  auto Hi = channel_last ? ishape.x : ishape.y;
  auto Wi = channel_last ? ishape.y : ishape.z;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto c = channel_last ? ond_index.z : ond_index.x;
    auto h = channel_last ? ond_index.x : ond_index.y;
    auto w = channel_last ? ond_index.y : ond_index.z;
    auto gnd_index = make_int3(h, w, 0);
    auto gidx = device_3d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;

      auto v_y0x0 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi0, xi0, Hi, Wi, istride, iisize);
      auto v_y0x1 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi0, xi1, Hi, Wi, istride, iisize);
      auto v_y1x0 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi1, xi0, Hi, Wi, istride, iisize);
      auto v_y1x1 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi1, xi1, Hi, Wi, istride, iisize);

      auto b_oidx = oidx + b * oisize;
      auto val = (v_y0x0 * py1 * px1) + (v_y0x1 * py1 * px0) +
                 (v_y1x0 * py0 * px1) + (v_y1x1 * py0 * px0);
      output[b_oidx] = val;
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_forward_3d(
    const int oisize, const int iisize, const int gisize, T *output,
    const T *input, const T *grid, const int4 ishape, const int3 istride,
    const int3 gstride, const int3 ostride, const int B) {
  auto Di = channel_last ? ishape.x : ishape.y;
  auto Hi = channel_last ? ishape.y : ishape.z;
  auto Wi = channel_last ? ishape.z : ishape.w;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto c = channel_last ? ond_index.w : ond_index.x;
    auto d = channel_last ? ond_index.x : ond_index.y;
    auto h = channel_last ? ond_index.y : ond_index.z;
    auto w = channel_last ? ond_index.z : ond_index.w;
    auto gnd_index = make_int4(d, h, w, 0);
    auto gidx = device_4d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto zn = grid[b_gidx + 2];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto zf0 = unnormalize_grid(zn, Di);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto zf = get_src_findex_with_pad(zf0, Di);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto zi0 = static_cast<int>(std::floor(zf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto zi1 = zi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto pz0 = zf - zi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;
      auto pz1 = T(1) - pz0;

      auto v_z0y0x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi0, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z0y0x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi0, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z0y1x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi1, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z0y1x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi1, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z1y0x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi0, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z1y0x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi0, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z1y1x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi1, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z1y1x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi1, xi1, Di, Hi, Wi, istride, iisize);
      auto val = v_z0y0x0 * pz1 * py1 * px1 + v_z0y0x1 * pz1 * py1 * px0 +
                 v_z0y1x0 * pz1 * py0 * px1 + v_z0y1x1 * pz1 * py0 * px0 +
                 v_z1y0x0 * pz0 * py1 * px1 + v_z1y0x1 * pz0 * py1 * px0 +
                 v_z1y1x0 * pz0 * py0 * px1 + v_z1y1x1 * pz0 * py0 * px0;

      auto b_oidx = oidx + b * oisize;
      output[b_oidx] = val;
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_nearest_forward_2d(
    const int oisize, const int iisize, const int gisize, T *output,
    const T *input, const T *grid, const int3 ishape, const int2 istride,
    const int2 gstride, const int2 ostride, const int B) {
  auto Hi = channel_last ? ishape.x : ishape.y;
  auto Wi = channel_last ? ishape.y : ishape.z;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto c = channel_last ? ond_index.z : ond_index.x;
    auto h = channel_last ? ond_index.x : ond_index.y;
    auto w = channel_last ? ond_index.y : ond_index.z;
    auto gnd_index = make_int3(h, w, 0);
    auto gidx = device_3d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto xi = static_cast<int>(std::floor(xf + T(0.5)));
      auto yi = static_cast<int>(std::floor(yf + T(0.5)));

      auto b_oidx = oidx + b * oisize;
      auto val = get_pixel_value_2d<T, channel_last>(input, b, c, yi, xi, Hi,
                                                     Wi, istride, iisize);
      output[b_oidx] = val;
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_nearest_forward_3d(
    const int oisize, const int iisize, const int gisize, T *output,
    const T *input, const T *grid, const int4 ishape, const int3 istride,
    const int3 gstride, const int3 ostride, const int B) {
  auto Di = channel_last ? ishape.x : ishape.y;
  auto Hi = channel_last ? ishape.y : ishape.z;
  auto Wi = channel_last ? ishape.z : ishape.w;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto c = channel_last ? ond_index.w : ond_index.x;
    auto d = channel_last ? ond_index.x : ond_index.y;
    auto h = channel_last ? ond_index.y : ond_index.z;
    auto w = channel_last ? ond_index.z : ond_index.w;
    auto gnd_index = make_int4(d, h, w, 0);
    auto gidx = device_4d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto zn = grid[b_gidx + 2];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto zf0 = unnormalize_grid(zn, Di);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto zf = get_src_findex_with_pad(zf0, Di);
      auto xi = static_cast<int>(std::floor(xf + T(0.5)));
      auto yi = static_cast<int>(std::floor(yf + T(0.5)));
      auto zi = static_cast<int>(std::floor(zf + T(0.5)));

      auto b_oidx = oidx + b * oisize;
      auto val = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi, yi, xi, Di, Hi, Wi, istride, iisize);
      output[b_oidx] = val;
    }
  }
}

/*
  Backward implementations wrt data.
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_backward_data_2d(
    const int oisize, const int iisize, const int gisize, T *igrad,
    const T *ograd, const T *grid, const int3 ishape, const int2 istride,
    const int2 gstride, const int2 ostride, const int B) {
  auto Hi = channel_last ? ishape.x : ishape.y;
  auto Wi = channel_last ? ishape.y : ishape.z;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto c = channel_last ? ond_index.z : ond_index.x;
    auto h = channel_last ? ond_index.x : ond_index.y;
    auto w = channel_last ? ond_index.y : ond_index.z;
    auto gnd_index = make_int3(h, w, 0);
    auto gidx = device_3d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;

      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];
      backward_data_2d<T, channel_last>(igrad, grad, py1, px1, b, c, yi0, xi0,
                                        Hi, Wi, istride, iisize);
      backward_data_2d<T, channel_last>(igrad, grad, py1, px0, b, c, yi0, xi1,
                                        Hi, Wi, istride, iisize);
      backward_data_2d<T, channel_last>(igrad, grad, py0, px1, b, c, yi1, xi0,
                                        Hi, Wi, istride, iisize);
      backward_data_2d<T, channel_last>(igrad, grad, py0, px0, b, c, yi1, xi1,
                                        Hi, Wi, istride, iisize);
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_nearest_backward_data_2d(
    const int oisize, const int iisize, const int gisize, T *igrad,
    const T *ograd, const T *grid, const int3 ishape, const int2 istride,
    const int2 gstride, const int2 ostride, const int B) {
  auto Hi = channel_last ? ishape.x : ishape.y;
  auto Wi = channel_last ? ishape.y : ishape.z;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto c = channel_last ? ond_index.z : ond_index.x;
    auto h = channel_last ? ond_index.x : ond_index.y;
    auto w = channel_last ? ond_index.y : ond_index.z;
    auto gnd_index = make_int3(h, w, 0);
    auto gidx = device_3d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto xi = static_cast<int>(std::floor(xf + T(0.5)));
      auto yi = static_cast<int>(std::floor(yf + T(0.5)));

      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];
      backward_data_2d<T, channel_last>(igrad, grad, T(1.0), T(1.0), b, c, yi,
                                        xi, Hi, Wi, istride, iisize);
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_backward_data_3d(
    const int oisize, const int iisize, const int gisize, T *igrad,
    const T *ograd, const T *grid, const int4 ishape, const int3 istride,
    const int3 gstride, const int3 ostride, const int B) {
  auto Di = channel_last ? ishape.x : ishape.y;
  auto Hi = channel_last ? ishape.y : ishape.z;
  auto Wi = channel_last ? ishape.z : ishape.w;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto c = channel_last ? ond_index.w : ond_index.x;
    auto d = channel_last ? ond_index.x : ond_index.y;
    auto h = channel_last ? ond_index.y : ond_index.z;
    auto w = channel_last ? ond_index.z : ond_index.w;
    auto gnd_index = make_int4(d, h, w, 0);
    auto gidx = device_4d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto zn = grid[b_gidx + 2];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto zf0 = unnormalize_grid(zn, Di);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto zf = get_src_findex_with_pad(zf0, Di);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto zi0 = static_cast<int>(std::floor(zf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto zi1 = zi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto pz0 = zf - zi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;
      auto pz1 = T(1) - pz0;

      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];
      if (channel_last) {
        backward_data_3d<T, true>(igrad, grad, pz1, py1, px1, b, c, zi0, yi0,
                                  xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz1, py1, px0, b, c, zi0, yi0,
                                  xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz1, py0, px1, b, c, zi0, yi1,
                                  xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz1, py0, px0, b, c, zi0, yi1,
                                  xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz0, py1, px1, b, c, zi1, yi0,
                                  xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz0, py1, px0, b, c, zi1, yi0,
                                  xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz0, py0, px1, b, c, zi1, yi1,
                                  xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, true>(igrad, grad, pz0, py0, px0, b, c, zi1, yi1,
                                  xi1, Di, Hi, Wi, istride, iisize);
      } else {
        backward_data_3d<T, false>(igrad, grad, pz1, py1, px1, b, c, zi0, yi0,
                                   xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz1, py1, px0, b, c, zi0, yi0,
                                   xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz1, py0, px1, b, c, zi0, yi1,
                                   xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz1, py0, px0, b, c, zi0, yi1,
                                   xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz0, py1, px1, b, c, zi1, yi0,
                                   xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz0, py1, px0, b, c, zi1, yi0,
                                   xi1, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz0, py0, px1, b, c, zi1, yi1,
                                   xi0, Di, Hi, Wi, istride, iisize);
        backward_data_3d<T, false>(igrad, grad, pz0, py0, px0, b, c, zi1, yi1,
                                   xi1, Di, Hi, Wi, istride, iisize);
      }
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_nearest_backward_data_3d(
    const int oisize, const int iisize, const int gisize, T *igrad,
    const T *ograd, const T *grid, const int4 ishape, const int3 istride,
    const int3 gstride, const int3 ostride, const int B) {
  auto Di = channel_last ? ishape.x : ishape.y;
  auto Hi = channel_last ? ishape.y : ishape.z;
  auto Wi = channel_last ? ishape.z : ishape.w;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto c = channel_last ? ond_index.w : ond_index.x;
    auto d = channel_last ? ond_index.x : ond_index.y;
    auto h = channel_last ? ond_index.y : ond_index.z;
    auto w = channel_last ? ond_index.z : ond_index.w;
    auto gnd_index = make_int4(d, h, w, 0);
    auto gidx = device_4d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto zn = grid[b_gidx + 2];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto zf0 = unnormalize_grid(zn, Di);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto zf = get_src_findex_with_pad(zf0, Di);
      auto xi = static_cast<int>(std::floor(xf + T(0.5)));
      auto yi = static_cast<int>(std::floor(yf + T(0.5)));
      auto zi = static_cast<int>(std::floor(zf + T(0.5)));

      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];
      backward_data_3d<T, channel_last>(igrad, grad, T(1.0), T(1.0), T(1.0), b,
                                        c, zi, yi, xi, Di, Hi, Wi, istride,
                                        iisize);
    }
  }
}

/*
  Backward implementations wrt grid.
 */

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_backward_grid_2d(
    const int oisize, const int iisize, const int gisize, T *ggrad,
    const T *ograd, const T *input, const T *grid, const int3 ishape,
    const int2 istride, const int2 gstride, const int2 ostride, const int B) {
  auto Hi = channel_last ? ishape.x : ishape.y;
  auto Wi = channel_last ? ishape.y : ishape.z;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };
  auto get_grad_coef_with_pad = [&](const T s, const Size_t S) {
    T coef;
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      coef = get_grad_coef_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      coef = align_corners ? get_grad_coef_with_reflect_pad<T, true>(s, S)
                           : get_grad_coef_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      coef = get_grad_coef_with_repeat_pad(s, S);
    } else {
      return T(0);
    }
    return align_corners ? coef * T(S - 1) / T(2) : coef * T(S) / T(2);
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto c = channel_last ? ond_index.z : ond_index.x;
    auto h = channel_last ? ond_index.x : ond_index.y;
    auto w = channel_last ? ond_index.y : ond_index.z;
    auto gnd_index = make_int3(h, w, 0);
    auto gidx = device_3d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;

      auto v_y0x0 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi0, xi0, Hi, Wi, istride, iisize);
      auto v_y0x1 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi0, xi1, Hi, Wi, istride, iisize);
      auto v_y1x0 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi1, xi0, Hi, Wi, istride, iisize);
      auto v_y1x1 = get_pixel_value_2d<T, channel_last>(
          input, b, c, yi1, xi1, Hi, Wi, istride, iisize);
      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];

      // d_grid = d_output * local_grad{output/pad(x)} * local_grad{pad(x)/x} *
      // unnormalized_coef
      auto grad_x = grad * ((v_y0x1 - v_y0x0) * py1 + (v_y1x1 - v_y1x0) * py0);
      auto grad_y = grad * ((v_y1x0 - v_y0x0) * px1 + (v_y1x1 - v_y0x1) * px0);
      auto coef_x = get_grad_coef_with_pad(xf0, Wi);
      auto coef_y = get_grad_coef_with_pad(yf0, Hi);
      atomic_add(ggrad + b_gidx + 0, grad_x * coef_x);
      atomic_add(ggrad + b_gidx + 1, grad_y * coef_y);
    }
  }
}

template <typename T, warp_by_grid::PADDING_MODE padding_mode =
                          warp_by_grid::PADDING_MODE::zero,
          bool align_corners = false, bool channel_last = false>
__global__ void kernel_warp_linear_backward_grid_3d(
    const int oisize, const int iisize, const int gisize, T *ggrad,
    const T *ograd, const T *input, const T *grid, const int4 ishape,
    const int3 istride, const int3 gstride, const int3 ostride, const int B) {
  auto Di = channel_last ? ishape.x : ishape.y;
  auto Hi = channel_last ? ishape.y : ishape.z;
  auto Wi = channel_last ? ishape.z : ishape.w;

  auto unnormalize_grid = align_corners ? unnormalize_grid_with<T, true>
                                        : unnormalize_grid_with<T, false>;
  auto get_src_findex_with_pad = [&](const T s, const Size_t S) {
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      return get_src_findex_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      return align_corners ? get_src_findex_with_reflect_pad<T, true>(s, S)
                           : get_src_findex_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      return get_src_findex_with_repeat_pad(s, S);
    } else {
      return T(-1);
    }
  };
  auto get_grad_coef_with_pad = [&](const T s, const Size_t S) {
    T coef;
    if (padding_mode == warp_by_grid::PADDING_MODE::zero) {
      coef = get_grad_coef_with_zero_pad(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::reflect) {
      coef = align_corners ? get_grad_coef_with_reflect_pad<T, true>(s, S)
                           : get_grad_coef_with_reflect_pad<T, false>(s, S);
    } else if (padding_mode == warp_by_grid::PADDING_MODE::repeat) {
      coef = get_grad_coef_with_repeat_pad(s, S);
    } else {
      return T(0);
    }
    return align_corners ? coef * T(S - 1) / T(2) : coef * T(S) / T(2);
  };

  NBLA_CUDA_KERNEL_LOOP(oidx, oisize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto c = channel_last ? ond_index.w : ond_index.x;
    auto d = channel_last ? ond_index.x : ond_index.y;
    auto h = channel_last ? ond_index.y : ond_index.z;
    auto w = channel_last ? ond_index.z : ond_index.w;
    auto gnd_index = make_int4(d, h, w, 0);
    auto gidx = device_4d_to_flat(gnd_index, gstride);

    for (auto b = 0; b < B; ++b) {
      auto b_gidx = gidx + b * gisize;
      auto xn = grid[b_gidx + 0];
      auto yn = grid[b_gidx + 1];
      auto zn = grid[b_gidx + 2];
      auto xf0 = unnormalize_grid(xn, Wi);
      auto yf0 = unnormalize_grid(yn, Hi);
      auto zf0 = unnormalize_grid(zn, Di);
      auto xf = get_src_findex_with_pad(xf0, Wi);
      auto yf = get_src_findex_with_pad(yf0, Hi);
      auto zf = get_src_findex_with_pad(zf0, Di);
      auto xi0 = static_cast<int>(std::floor(xf));
      auto yi0 = static_cast<int>(std::floor(yf));
      auto zi0 = static_cast<int>(std::floor(zf));
      auto xi1 = xi0 + 1;
      auto yi1 = yi0 + 1;
      auto zi1 = zi0 + 1;
      auto px0 = xf - xi0;
      auto py0 = yf - yi0;
      auto pz0 = zf - zi0;
      auto px1 = T(1) - px0;
      auto py1 = T(1) - py0;
      auto pz1 = T(1) - pz0;

      auto v_z0y0x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi0, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z0y0x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi0, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z0y1x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi1, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z0y1x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi0, yi1, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z1y0x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi0, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z1y0x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi0, xi1, Di, Hi, Wi, istride, iisize);
      auto v_z1y1x0 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi1, xi0, Di, Hi, Wi, istride, iisize);
      auto v_z1y1x1 = get_pixel_value_3d<T, channel_last>(
          input, b, c, zi1, yi1, xi1, Di, Hi, Wi, istride, iisize);
      auto b_oidx = oidx + b * oisize;
      auto grad = ograd[b_oidx];

      // d_grid = d_output * local_grad{output/pad(x)} * local_grad{pad(x)/x} *
      // unnormalized_coef
      auto grad_x = grad * ((v_z0y0x1 - v_z0y0x0) * pz1 * py1 +
                            (v_z0y1x1 - v_z0y1x0) * pz1 * py0 +
                            (v_z1y0x1 - v_z1y0x0) * pz0 * py1 +
                            (v_z1y1x1 - v_z1y1x0) * pz0 * py0);
      auto grad_y = grad * ((v_z0y1x0 - v_z0y0x0) * pz1 * px1 +
                            (v_z0y1x1 - v_z0y0x1) * pz1 * px0 +
                            (v_z1y1x0 - v_z1y0x0) * pz0 * px1 +
                            (v_z1y1x1 - v_z1y0x1) * pz0 * px0);
      auto grad_z = grad * ((v_z1y0x0 - v_z0y0x0) * py1 * px1 +
                            (v_z1y0x1 - v_z0y0x1) * py1 * px0 +
                            (v_z1y1x0 - v_z0y1x0) * py0 * px1 +
                            (v_z1y1x1 - v_z0y1x1) * py0 * px0);
      auto coef_x = get_grad_coef_with_pad(xf0, Wi);
      auto coef_y = get_grad_coef_with_pad(yf0, Hi);
      auto coef_z = get_grad_coef_with_pad(zf0, Di);
      atomic_add(ggrad + b_gidx + 0, grad_x * coef_x);
      atomic_add(ggrad + b_gidx + 1, grad_y * coef_y);
      atomic_add(ggrad + b_gidx + 2, grad_z * coef_z);
    }
  }
}

template <typename T>
void WarpByGridCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  WarpByGrid<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void WarpByGridCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  auto ndims = inputs[1]->shape().size();
  auto channel_last = this->channel_last_;
  auto align_corners = this->align_corners_;
  auto padding_mode_t = this->padding_mode_t_;
  using PADDING_MODE = warp_by_grid::PADDING_MODE;
  auto zero = PADDING_MODE::zero;
  auto repeat = PADDING_MODE::repeat;
  auto reflect = PADDING_MODE::reflect;

  if (ndims == 4) {
    auto B = inputs[0]->shape()[0];
    auto Ci = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[1];
    auto Hi = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
    auto Wi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
    auto Ho = channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
    auto Wo = channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];

    auto ishape = channel_last ? make_int3(Hi, Wi, Ci) : make_int3(Ci, Hi, Wi);

    auto istride =
        channel_last ? make_int2(Wi * Ci, Ci) : make_int2(Hi * Wi, Wi);
    auto gstride = make_int2(Wo * 2, 2);
    auto ostride =
        channel_last ? make_int2(Wo * Ci, Ci) : make_int2(Ho * Wo, Wo);

    auto output = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);
    auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
    auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

    auto oisize = Ci * Ho * Wo;
    auto iisize = Ci * Hi * Wi;
    auto gisize = Ho * Wo * 2;

    if (this->mode_ == "linear") {
      if (channel_last) {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::zero,
                                                      true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::repeat,
                                                      true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::reflect, true,
                                            true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::zero,
                                                      false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::repeat,
                                                      false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::reflect, false,
                                            true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      } else {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::zero,
                                                      true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::repeat,
                                                      true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::reflect, true,
                                            false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::zero,
                                                      false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel = kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::repeat,
                                                      false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_linear_forward_2d<Tcu, PADDING_MODE::reflect, false,
                                            false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      }
    } else if (this->mode_ == "nearest") {
      if (channel_last) {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::zero,
                                                       true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::repeat, true,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::reflect, true,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::zero,
                                                       false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::repeat, false,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::reflect, false,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      } else {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::zero,
                                                       true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::repeat, true,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::reflect, true,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::zero,
                                                       false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::repeat, false,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_2d<Tcu, PADDING_MODE::reflect, false,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      }
    }
  } else if (ndims == 5) {
    auto B = inputs[0]->shape()[0];
    auto Ci = channel_last ? inputs[0]->shape()[4] : inputs[0]->shape()[1];
    auto Di = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
    auto Hi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
    auto Wi = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[4];
    auto Do = channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
    auto Ho = channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];
    auto Wo = channel_last ? outputs[0]->shape()[3] : outputs[0]->shape()[4];

    auto ishape =
        channel_last ? make_int4(Di, Hi, Wi, Ci) : make_int4(Ci, Di, Hi, Wi);

    auto istride = channel_last ? make_int3(Hi * Wi * Ci, Wi * Ci, Ci)
                                : make_int3(Di * Hi * Wi, Hi * Wi, Wi);
    auto gstride = make_int3(Ho * Wo * 3, Wo * 3, 3);
    auto ostride = channel_last ? make_int3(Ho * Wo * Ci, Wo * Ci, Ci)
                                : make_int3(Do * Ho * Wo, Ho * Wo, Wo);

    auto output = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);
    auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
    auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

    auto oisize = Ci * Do * Ho * Wo;
    auto iisize = Ci * Di * Hi * Wi;
    auto gisize = Do * Ho * Wo * 3;

    if (this->mode_ == "linear") {
      if (channel_last) {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::zero,
                                                      true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::repeat,
                                                      true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::reflect, true,
                                            true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::zero,
                                                      false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::repeat,
                                                      false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::reflect, false,
                                            true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      } else {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::zero,
                                                      true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::repeat,
                                                      true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::reflect, true,
                                            false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::zero,
                                                      false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel = kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::repeat,
                                                      false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_linear_forward_3d<Tcu, PADDING_MODE::reflect, false,
                                            false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      }
    } else if (this->mode_ == "nearest") {
      if (channel_last) {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::zero,
                                                       true, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::repeat, true,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::reflect, true,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::zero,
                                                       false, true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::repeat, false,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::reflect, false,
                                             true>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      } else {
        if (padding_mode_t == zero && align_corners) {
          auto kernel = kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::zero,
                                                       true, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::repeat, true,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::reflect, true,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == zero && !align_corners) {
          auto kernel = kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::zero,
                                                       false, false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == repeat && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::repeat, false,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        } else if (padding_mode_t == reflect && !align_corners) {
          auto kernel =
              kernel_warp_nearest_forward_3d<Tcu, PADDING_MODE::reflect, false,
                                             false>;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize, output,
                                         input, grid, ishape, istride, gstride,
                                         ostride, B);
        }
      }
    }
  }
}

template <typename T>
void WarpByGridCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);
  auto ndims = inputs[1]->shape().size();
  auto channel_last = this->channel_last_;
  auto align_corners = this->align_corners_;
  auto padding_mode_t = this->padding_mode_t_;
  using PADDING_MODE = warp_by_grid::PADDING_MODE;
  auto zero = PADDING_MODE::zero;
  auto repeat = PADDING_MODE::repeat;
  auto reflect = PADDING_MODE::reflect;

  // w.r.t. data
  if (propagate_down[0]) {
    if (ndims == 4) {
      auto B = inputs[0]->shape()[0];
      auto Ci = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[1];
      auto Hi = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
      auto Wi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
      auto Ho = channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
      auto Wo = channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];

      auto ishape =
          channel_last ? make_int3(Hi, Wi, Ci) : make_int3(Ci, Hi, Wi);

      auto istride =
          channel_last ? make_int2(Wi * Ci, Ci) : make_int2(Hi * Wi, Wi);
      auto gstride = make_int2(Wo * 2, 2);
      auto ostride =
          channel_last ? make_int2(Wo * Ci, Ci) : make_int2(Ho * Wo, Wo);

      auto output = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
      auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
      auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
      auto ograd = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
      auto igrad = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
      auto ggrad = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_);

      auto oisize = Ci * Ho * Wo;
      auto iisize = Ci * Hi * Wi;
      auto gisize = Ho * Wo * 2;

      if (this->mode_ == "linear") {
        if (channel_last) {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        } else {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        }
      } else if (this->mode_ == "nearest") {
        if (channel_last) {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        } else {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::zero,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::repeat,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_2d<Tcu, PADDING_MODE::reflect,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        }
      }
    } else if (ndims == 5) {
      auto B = inputs[0]->shape()[0];
      auto Ci = channel_last ? inputs[0]->shape()[4] : inputs[0]->shape()[1];
      auto Di = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
      auto Hi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
      auto Wi = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[4];
      auto Do = channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
      auto Ho = channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];
      auto Wo = channel_last ? outputs[0]->shape()[3] : outputs[0]->shape()[4];

      auto ishape =
          channel_last ? make_int4(Di, Hi, Wi, Ci) : make_int4(Ci, Di, Hi, Wi);

      auto istride = channel_last ? make_int3(Hi * Wi * Ci, Wi * Ci, Ci)
                                  : make_int3(Di * Hi * Wi, Hi * Wi, Wi);
      auto gstride = make_int3(Ho * Wo * 3, Wo * 3, 3);
      auto ostride = channel_last ? make_int3(Ho * Wo * Ci, Wo * Ci, Ci)
                                  : make_int3(Do * Ho * Wo, Ho * Wo, Wo);

      auto output = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
      auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
      auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
      auto ograd = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
      auto igrad = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
      auto ggrad = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_);

      auto oisize = Ci * Do * Ho * Wo;
      auto iisize = Ci * Di * Hi * Wi;
      auto gisize = Do * Ho * Wo * 3;

      if (this->mode_ == "linear") {
        if (channel_last) {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                    true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                    false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        } else {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                    true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_linear_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                    false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        }
      } else if (this->mode_ == "nearest") {
        if (channel_last) {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                     true, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                     false, true>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        } else {
          if (padding_mode_t == zero && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                     true, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == zero && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::zero,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == repeat && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::repeat,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          } else if (padding_mode_t == reflect && !align_corners) {
            auto kernel =
                kernel_warp_nearest_backward_data_3d<Tcu, PADDING_MODE::reflect,
                                                     false, false>;
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                           igrad, ograd, grid, ishape, istride,
                                           gstride, ostride, B);
          }
        }
      }
    }

    // w.r.t. grid
    if (propagate_down[1]) {
      if (ndims == 4) {
        auto B = inputs[0]->shape()[0];
        auto Ci = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[1];
        auto Hi = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
        auto Wi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
        auto Ho =
            channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
        auto Wo =
            channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];

        auto ishape =
            channel_last ? make_int3(Hi, Wi, Ci) : make_int3(Ci, Hi, Wi);

        auto istride =
            channel_last ? make_int2(Wi * Ci, Ci) : make_int2(Hi * Wi, Wi);
        auto gstride = make_int2(Wo * 2, 2);
        auto ostride =
            channel_last ? make_int2(Wo * Ci, Ci) : make_int2(Ho * Wo, Wo);

        auto output = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
        auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
        auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
        auto ograd = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
        auto igrad = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
        auto ggrad = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_);

        auto oisize = Ci * Ho * Wo;
        auto iisize = Ci * Hi * Wi;
        auto gisize = Ho * Wo * 2;

        if (this->mode_ == "linear") {
          if (channel_last) {
            if (padding_mode_t == zero && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::zero,
                                                      true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::repeat,
                                                      true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_2d<
                  Tcu, PADDING_MODE::reflect, true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == zero && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::zero,
                                                      false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::repeat,
                                                      false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && !align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_2d<
                  Tcu, PADDING_MODE::reflect, false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            }
          } else {
            if (padding_mode_t == zero && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::zero,
                                                      true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::repeat,
                                                      true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_2d<
                  Tcu, PADDING_MODE::reflect, true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == zero && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::zero,
                                                      false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_2d<Tcu, PADDING_MODE::repeat,
                                                      false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && !align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_2d<
                  Tcu, PADDING_MODE::reflect, false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            }
          }
        } else if (this->mode_ == "nearest") {
          NBLA_ERROR(
              error_code::not_implemented,
              "Backward wrt the grid is not supported in the nearest mode. "
              "Use the `linear` mode.");
        }
      } else if (ndims == 5) {
        auto B = inputs[0]->shape()[0];
        auto Ci = channel_last ? inputs[0]->shape()[4] : inputs[0]->shape()[1];
        auto Di = channel_last ? inputs[0]->shape()[1] : inputs[0]->shape()[2];
        auto Hi = channel_last ? inputs[0]->shape()[2] : inputs[0]->shape()[3];
        auto Wi = channel_last ? inputs[0]->shape()[3] : inputs[0]->shape()[4];
        auto Do =
            channel_last ? outputs[0]->shape()[1] : outputs[0]->shape()[2];
        auto Ho =
            channel_last ? outputs[0]->shape()[2] : outputs[0]->shape()[3];
        auto Wo =
            channel_last ? outputs[0]->shape()[3] : outputs[0]->shape()[4];

        auto ishape = channel_last ? make_int4(Di, Hi, Wi, Ci)
                                   : make_int4(Ci, Di, Hi, Wi);

        auto istride = channel_last ? make_int3(Hi * Wi * Ci, Wi * Ci, Ci)
                                    : make_int3(Di * Hi * Wi, Hi * Wi, Wi);
        auto gstride = make_int3(Ho * Wo * 3, Wo * 3, 3);
        auto ostride = channel_last ? make_int3(Ho * Wo * Ci, Wo * Ci, Ci)
                                    : make_int3(Do * Ho * Wo, Ho * Wo, Wo);

        auto output = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
        auto input = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
        auto grid = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
        auto ograd = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
        auto igrad = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_);
        auto ggrad = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_);

        auto oisize = Ci * Do * Ho * Wo;
        auto iisize = Ci * Di * Hi * Wi;
        auto gisize = Do * Ho * Wo * 3;

        if (this->mode_ == "linear") {
          if (channel_last) {
            if (padding_mode_t == zero && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::zero,
                                                      true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::repeat,
                                                      true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_3d<
                  Tcu, PADDING_MODE::reflect, true, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == zero && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::zero,
                                                      false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::repeat,
                                                      false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && !align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_3d<
                  Tcu, PADDING_MODE::reflect, false, true>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            }
          } else {
            if (padding_mode_t == zero && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::zero,
                                                      true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::repeat,
                                                      true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_3d<
                  Tcu, PADDING_MODE::reflect, true, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == zero && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::zero,
                                                      false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == repeat && !align_corners) {
              auto kernel =
                  kernel_warp_linear_backward_grid_3d<Tcu, PADDING_MODE::repeat,
                                                      false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            } else if (padding_mode_t == reflect && !align_corners) {
              auto kernel = kernel_warp_linear_backward_grid_3d<
                  Tcu, PADDING_MODE::reflect, false, false>;
              NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, oisize, iisize, gisize,
                                             ggrad, ograd, input, grid, ishape,
                                             istride, gstride, ostride, B);
            }
          }
        } else if (this->mode_ == "nearest") {
          NBLA_ERROR(
              error_code::not_implemented,
              "Backward wrt the grid is not supported in the nearest mode. "
              "Use the `linear` mode.");
        }
      }
    }
  }
}
}

// Copyright 2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/max_pooling_backward.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace nbla {

namespace max_pooling_backward {

template <typename T> __global__ void kernel_zeroing(int size, T *x) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { x[idx] = T(0); }
}

template <typename T, bool channel_last = false>
__global__ void kernel_max_pooling_2d_forward(
    int y_isize, int x_isize, T *dx, const T *dy, const T *x, int Cx, int Hx,
    int Wx, int2 xstride, int By, int Cy, int Hy, int Wy, int2 ystride,
    int wkernel, int hkernel, int wstride, int hstride, int wpad, int hpad) {
  NBLA_CUDA_KERNEL_LOOP(yidx, y_isize) {
    auto ynd_index = device_flat_to_3d(yidx, ystride);
    auto c = channel_last ? ynd_index.z : ynd_index.x;
    auto h = channel_last ? ynd_index.x : ynd_index.y;
    auto w = channel_last ? ynd_index.y : ynd_index.z;
    for (auto b = 0; b < By; ++b) {
      auto dx_b = dx + b * x_isize;
      auto x_b = x + b * x_isize;
      auto dy_b = dy + b * y_isize;
      // region
      auto hi_pool_start = h * hstride - hpad;
      auto wi_pool_start = w * wstride - wpad;
      auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
      auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
      hi_pool_start = max(hi_pool_start, 0);
      wi_pool_start = max(wi_pool_start, 0);
      // pool
      auto xnd_idx = channel_last ? make_int3(hi_pool_start, wi_pool_start, c)
                                  : make_int3(c, hi_pool_start, wi_pool_start);
      auto max_idx = device_3d_to_flat(xnd_idx, xstride);
      auto max_val = x_b[max_idx];
      for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
        for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
          xnd_idx = channel_last ? make_int3(rh, rw, c) : make_int3(c, rh, rw);
          auto xidx = device_3d_to_flat(xnd_idx, xstride);
          auto x_bidx = x_b[xidx];
          if (max_val < x_bidx) {
            max_val = x_bidx;
            max_idx = xidx;
          }
        }
      }
      atomic_add(dx_b + max_idx, dy_b[yidx]);
    }
  }
}

template <typename T, bool channel_last = false>
__global__ void kernel_max_pooling_3d_forward(
    int y_isize, int x_isize, T *dx, const T *dy, const T *x, int Cx, int Dx,
    int Hx, int Wx, int3 xstride, int By, int Cy, int Dy, int Hy, int Wy,
    int3 ystride, int wkernel, int hkernel, int dkernel, int wstride,
    int hstride, int dstride, int wpad, int hpad, int dpad) {
  NBLA_CUDA_KERNEL_LOOP(yidx, y_isize) {
    auto ynd_index = device_flat_to_4d(yidx, ystride);
    auto c = channel_last ? ynd_index.w : ynd_index.x;
    auto d = channel_last ? ynd_index.x : ynd_index.y;
    auto h = channel_last ? ynd_index.y : ynd_index.z;
    auto w = channel_last ? ynd_index.z : ynd_index.w;
    for (auto b = 0; b < By; ++b) {
      auto dx_b = dx + b * x_isize;
      auto x_b = x + b * x_isize;
      auto dy_b = dy + b * y_isize;
      // region
      auto di_pool_start = d * dstride - dpad;
      auto hi_pool_start = h * hstride - hpad;
      auto wi_pool_start = w * wstride - wpad;
      auto di_pool_end = min(di_pool_start + dkernel, Dx);
      auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
      auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
      di_pool_start = max(di_pool_start, 0);
      hi_pool_start = max(hi_pool_start, 0);
      wi_pool_start = max(wi_pool_start, 0);
      // pool
      auto xnd_idx =
          channel_last
              ? make_int4(di_pool_start, hi_pool_start, wi_pool_start, c)
              : make_int4(c, di_pool_start, hi_pool_start, wi_pool_start);
      auto max_idx = device_4d_to_flat(xnd_idx, xstride);
      auto max_val = x_b[max_idx];
      for (auto rd = di_pool_start; rd < di_pool_end; ++rd) {
        for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
          for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
            xnd_idx = channel_last ? make_int4(rd, rh, rw, c)
                                   : make_int4(c, rd, rh, rw);
            auto xidx = device_4d_to_flat(xnd_idx, xstride);
            auto x_bidx = x_b[xidx];
            if (max_val < x_bidx) {
              max_val = x_bidx;
              max_idx = xidx;
            }
          }
        }
      }
      atomic_add(dx_b + max_idx, dy_b[yidx]);
    }
  }
}

template <typename T, bool accum = false, bool channel_last = false>
__global__ void kernel_max_pooling_2d_backward(
    int y_isize, int x_isize, T *gdy, const T *gdx, const T *x, int Cx, int Hx,
    int Wx, int2 xstride, int By, int Cy, int Hy, int Wy, int2 ystride,
    int wkernel, int hkernel, int wstride, int hstride, int wpad, int hpad) {
  NBLA_CUDA_KERNEL_LOOP(yidx, y_isize) {
    auto ynd_index = device_flat_to_3d(yidx, ystride);
    auto c = channel_last ? ynd_index.z : ynd_index.x;
    auto h = channel_last ? ynd_index.x : ynd_index.y;
    auto w = channel_last ? ynd_index.y : ynd_index.z;
    for (auto b = 0; b < By; ++b) {
      auto gdx_b = gdx + b * x_isize;
      auto x_b = x + b * x_isize;
      auto gdy_b = gdy + b * y_isize;
      // region
      auto hi_pool_start = h * hstride - hpad;
      auto wi_pool_start = w * wstride - wpad;
      auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
      auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
      hi_pool_start = max(hi_pool_start, 0);
      wi_pool_start = max(wi_pool_start, 0);
      // pool
      auto xnd_idx = channel_last ? make_int3(hi_pool_start, wi_pool_start, c)
                                  : make_int3(c, hi_pool_start, wi_pool_start);
      auto max_idx = device_3d_to_flat(xnd_idx, xstride);
      auto max_val = x_b[max_idx];
      for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
        for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
          xnd_idx = channel_last ? make_int3(rh, rw, c) : make_int3(c, rh, rw);
          auto xidx = device_3d_to_flat(xnd_idx, xstride);
          auto x_bidx = x_b[xidx];
          if (max_val < x_bidx) {
            max_val = x_bidx;
            max_idx = xidx;
          }
        }
      }
      gdy_b[yidx] = accum ? gdy_b[yidx] + gdx_b[max_idx] : gdx_b[max_idx];
    }
  }
}

template <typename T, bool accum = false, bool channel_last = false>
__global__ void kernel_max_pooling_3d_backward(
    int y_isize, int x_isize, T *gdy, const T *gdx, const T *x, int Cx, int Dx,
    int Hx, int Wx, int3 xstride, int By, int Cy, int Dy, int Hy, int Wy,
    int3 ystride, int wkernel, int hkernel, int dkernel, int wstride,
    int hstride, int dstride, int wpad, int hpad, int dpad) {
  NBLA_CUDA_KERNEL_LOOP(yidx, y_isize) {
    auto ynd_index = device_flat_to_4d(yidx, ystride);
    auto c = channel_last ? ynd_index.w : ynd_index.x;
    auto d = channel_last ? ynd_index.x : ynd_index.y;
    auto h = channel_last ? ynd_index.y : ynd_index.z;
    auto w = channel_last ? ynd_index.z : ynd_index.w;
    for (auto b = 0; b < By; ++b) {
      auto gdx_b = gdx + b * x_isize;
      auto x_b = x + b * x_isize;
      auto gdy_b = gdy + b * y_isize;
      // region
      auto di_pool_start = d * dstride - dpad;
      auto hi_pool_start = h * hstride - hpad;
      auto wi_pool_start = w * wstride - wpad;
      auto di_pool_end = min(di_pool_start + dkernel, Dx);
      auto hi_pool_end = min(hi_pool_start + hkernel, Hx);
      auto wi_pool_end = min(wi_pool_start + wkernel, Wx);
      di_pool_start = max(di_pool_start, 0);
      hi_pool_start = max(hi_pool_start, 0);
      wi_pool_start = max(wi_pool_start, 0);
      // pool
      auto xnd_idx =
          channel_last
              ? make_int4(di_pool_start, hi_pool_start, wi_pool_start, c)
              : make_int4(c, di_pool_start, hi_pool_start, wi_pool_start);
      auto max_idx = device_4d_to_flat(xnd_idx, xstride);
      auto max_val = x_b[max_idx];
      for (auto rd = di_pool_start; rd < di_pool_end; ++rd) {
        for (auto rh = hi_pool_start; rh < hi_pool_end; ++rh) {
          for (auto rw = wi_pool_start; rw < wi_pool_end; ++rw) {
            xnd_idx = channel_last ? make_int4(rd, rh, rw, c)
                                   : make_int4(c, rd, rh, rw);
            auto xidx = device_4d_to_flat(xnd_idx, xstride);
            auto x_bidx = x_b[xidx];
            if (max_val < x_bidx) {
              max_val = x_bidx;
              max_idx = xidx;
            }
          }
        }
      }
      gdy_b[yidx] = accum ? gdy_b[yidx] + gdx_b[max_idx] : gdx_b[max_idx];
    }
  }
}
}

template <typename T>
void MaxPoolingBackwardCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  MaxPoolingBackward<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void MaxPoolingBackwardCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  // inputs[0]  : dy
  // inputs[1]  : x
  // outputs[0] : dx
  // dx = df(dy, x)

  auto sdim = this->kernel_.size();
  auto yshape = inputs[0]->shape();
  auto xshape = inputs[1]->shape();
  int ndim = xshape.size();
  // data
  auto dy = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto x = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto dx = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, false);
  // zeroing
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(max_pooling_backward::kernel_zeroing,
                                 outputs[0]->size(), dx);
  if (sdim == 2) {
    // pool params
    int hstride = this->stride_[0];
    int wstride = this->stride_[1];
    int hpad = this->pad_[0];
    int wpad = this->pad_[1];
    int hkernel = this->kernel_[0];
    int wkernel = this->kernel_[1];
    int Cx = this->channel_last_ ? xshape[ndim - 1] : xshape[ndim - 3];
    int Hx = this->channel_last_ ? xshape[ndim - 3] : xshape[ndim - 2];
    int Wx = this->channel_last_ ? xshape[ndim - 2] : xshape[ndim - 1];
    int Cy = this->channel_last_ ? yshape[ndim - 1] : yshape[ndim - 3];
    int Hy = this->channel_last_ ? yshape[ndim - 3] : yshape[ndim - 2];
    int Wy = this->channel_last_ ? yshape[ndim - 2] : yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Hy * Wy);
    auto y_isize = Cy * Hy * Wy;
    auto x_isize = Cx * Hx * Wx;
    auto ystride =
        this->channel_last_ ? make_int2(Wy * Cy, Cy) : make_int2(Hy * Wy, Wy);
    auto xstride =
        this->channel_last_ ? make_int2(Wx * Cx, Cx) : make_int2(Hx * Wx, Wx);
    // pool
    auto forward =
        this->channel_last_
            ? max_pooling_backward::kernel_max_pooling_2d_forward<Tcu, true>
            : max_pooling_backward::kernel_max_pooling_2d_forward<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        forward, y_isize, x_isize, dx, dy, x, Cx, Hx, Wx, xstride, By, Cy, Hy,
        Wy, ystride, wkernel, hkernel, wstride, hstride, wpad, hpad);

  } else if (sdim == 3) {
    // pool params
    int dstride = this->stride_[0];
    int hstride = this->stride_[1];
    int wstride = this->stride_[2];
    int dpad = this->pad_[0];
    int hpad = this->pad_[1];
    int wpad = this->pad_[2];
    int dkernel = this->kernel_[0];
    int hkernel = this->kernel_[1];
    int wkernel = this->kernel_[2];
    int Cx = this->channel_last_ ? xshape[ndim - 1] : xshape[ndim - 4];
    int Dx = this->channel_last_ ? xshape[ndim - 4] : xshape[ndim - 3];
    int Hx = this->channel_last_ ? xshape[ndim - 3] : xshape[ndim - 2];
    int Wx = this->channel_last_ ? xshape[ndim - 2] : xshape[ndim - 1];
    int Cy = this->channel_last_ ? yshape[ndim - 1] : yshape[ndim - 4];
    int Dy = this->channel_last_ ? yshape[ndim - 4] : yshape[ndim - 3];
    int Hy = this->channel_last_ ? yshape[ndim - 3] : yshape[ndim - 2];
    int Wy = this->channel_last_ ? yshape[ndim - 2] : yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Dy * Hy * Wy);
    auto y_isize = Cy * Dy * Hy * Wy;
    auto x_isize = Cx * Dx * Hx * Wx;
    auto ystride = this->channel_last_ ? make_int3(Hy * Wy * Cy, Wy * Cy, Cy)
                                       : make_int3(Dy * Hy * Wy, Hy * Wy, Wy);
    auto xstride = this->channel_last_ ? make_int3(Hx * Wx * Cx, Wx * Cx, Cx)
                                       : make_int3(Dx * Hx * Wx, Hx * Wx, Wx);
    // pool
    auto forward =
        this->channel_last_
            ? max_pooling_backward::kernel_max_pooling_3d_forward<Tcu, true>
            : max_pooling_backward::kernel_max_pooling_3d_forward<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward, y_isize, x_isize, dx, dy, x, Cx, Dx,
                                   Hx, Wx, xstride, By, Cy, Dy, Hy, Wy, ystride,
                                   wkernel, hkernel, dkernel, wstride, hstride,
                                   dstride, wpad, hpad, dpad);
  }
}

template <typename T>
void MaxPoolingBackwardCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);
  auto sdim = this->kernel_.size();
  auto yshape = inputs[0]->shape();
  auto xshape = inputs[1]->shape();
  int ndim = xshape.size();
  // data
  auto gdy = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  auto x = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto gdx = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  if (sdim == 2) {
    // pool params
    int hstride = this->stride_[0];
    int wstride = this->stride_[1];
    int hpad = this->pad_[0];
    int wpad = this->pad_[1];
    int hkernel = this->kernel_[0];
    int wkernel = this->kernel_[1];
    int Cx = this->channel_last_ ? xshape[ndim - 1] : xshape[ndim - 3];
    int Hx = this->channel_last_ ? xshape[ndim - 3] : xshape[ndim - 2];
    int Wx = this->channel_last_ ? xshape[ndim - 2] : xshape[ndim - 1];
    int Cy = this->channel_last_ ? yshape[ndim - 1] : yshape[ndim - 3];
    int Hy = this->channel_last_ ? yshape[ndim - 3] : yshape[ndim - 2];
    int Wy = this->channel_last_ ? yshape[ndim - 2] : yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Hy * Wy);
    auto y_isize = Cy * Hy * Wy;
    auto x_isize = Cx * Hx * Wx;
    auto ystride =
        this->channel_last_ ? make_int2(Wy * Cy, Cy) : make_int2(Hy * Wy, Wy);
    auto xstride =
        this->channel_last_ ? make_int2(Wx * Cx, Cx) : make_int2(Hx * Wx, Wx);
    // pool
    auto backward =
        accum[0]
            ? this->channel_last_
                  ? max_pooling_backward::kernel_max_pooling_2d_backward<
                        Tcu, true, true>
                  : max_pooling_backward::kernel_max_pooling_2d_backward<
                        Tcu, true, false>
            : this->channel_last_
                  ? max_pooling_backward::kernel_max_pooling_2d_backward<
                        Tcu, false, true>
                  : max_pooling_backward::kernel_max_pooling_2d_backward<
                        Tcu, false, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        backward, y_isize, x_isize, gdy, gdx, x, Cx, Hx, Wx, xstride, By, Cy,
        Hy, Wy, ystride, wkernel, hkernel, wstride, hstride, wpad, hpad);
  } else if (sdim == 3) {
    // pool params
    int dstride = this->stride_[0];
    int hstride = this->stride_[1];
    int wstride = this->stride_[2];
    int dpad = this->pad_[0];
    int hpad = this->pad_[1];
    int wpad = this->pad_[2];
    int dkernel = this->kernel_[0];
    int hkernel = this->kernel_[1];
    int wkernel = this->kernel_[2];
    int Cx = this->channel_last_ ? xshape[ndim - 1] : xshape[ndim - 4];
    int Dx = this->channel_last_ ? xshape[ndim - 4] : xshape[ndim - 3];
    int Hx = this->channel_last_ ? xshape[ndim - 3] : xshape[ndim - 2];
    int Wx = this->channel_last_ ? xshape[ndim - 2] : xshape[ndim - 1];
    int Cy = this->channel_last_ ? yshape[ndim - 1] : yshape[ndim - 4];
    int Dy = this->channel_last_ ? yshape[ndim - 4] : yshape[ndim - 3];
    int Hy = this->channel_last_ ? yshape[ndim - 3] : yshape[ndim - 2];
    int Wy = this->channel_last_ ? yshape[ndim - 2] : yshape[ndim - 1];
    int By = inputs[0]->size() / (Cy * Dy * Hy * Wy);
    auto y_isize = Cy * Dy * Hy * Wy;
    auto x_isize = Cx * Dx * Hx * Wx;
    auto ystride = this->channel_last_ ? make_int3(Hy * Wy * Cy, Wy * Cy, Cy)
                                       : make_int3(Dy * Hy * Wy, Hy * Wy, Wy);
    auto xstride = this->channel_last_ ? make_int3(Hx * Wx * Cx, Wx * Cx, Cx)
                                       : make_int3(Dx * Hx * Wx, Hx * Wx, Wx);
    // pool
    auto backward =
        accum[0]
            ? this->channel_last_
                  ? max_pooling_backward::kernel_max_pooling_3d_backward<
                        Tcu, true, true>
                  : max_pooling_backward::kernel_max_pooling_3d_backward<
                        Tcu, true, false>
            : this->channel_last_
                  ? max_pooling_backward::kernel_max_pooling_3d_backward<
                        Tcu, false, true>
                  : max_pooling_backward::kernel_max_pooling_3d_backward<
                        Tcu, false, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward, y_isize, x_isize, gdy, gdx, x, Cx,
                                   Dx, Hx, Wx, xstride, By, Cy, Dy, Hy, Wy,
                                   ystride, wkernel, hkernel, dkernel, wstride,
                                   hstride, dstride, wpad, hpad, dpad);
  }
}
}

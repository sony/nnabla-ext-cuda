// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/slice.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/cuda/utils/nd_index.hpp>
#include <nbla/variable.hpp>

namespace nbla {

// 1d slice
template <typename T>
__global__ void kernel_slice_1d_forward(const int size, const T *x, T *y,
                                        const int start, const int step) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = x[start + idx * step]; }
}

template <typename T, bool accum>
__global__ void kernel_slice_1d_backward(const int size, const T *dy, T *dx,
                                         const int start, const int step) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    dx[start + idx * step] = accum ? dx[start + idx * step] + dy[idx] : dy[idx];
  }
}

template <typename T>
void slice_1d_forward(const T *x_data, T *y_data, Size_t ndim, Size_t ysize,
                      const Shape_t &xshape, const Shape_t &yshape,
                      const Shape_t &xstrides, const Shape_t &ystrides,
                      const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_1d_forward, ysize, x_data, y_data,
                                 start[0], step[0]);
}

template <typename T, bool accum>
void slice_1d_backward(const T *y_grad, T *x_grad, Size_t ndim, Size_t ysize,
                       const Shape_t &xshape, const Shape_t &yshape,
                       const Shape_t &xstrides, const Shape_t &ystrides,
                       const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_slice_1d_backward<T, accum>), ysize,
                                 y_grad, x_grad, start[0], step[0]);
}

// 2d slice
template <typename T>
__global__ void kernel_slice_2d_forward(const int size, const T *x, T *y,
                                        const int xstrides, const int ystrides,
                                        const int2 start, const int2 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_2d(yidx, ystrides);
    auto nd_xidx =
        make_int2(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y);
    auto xidx = device_2d_to_flat(nd_xidx, xstrides);
    y[yidx] = x[xidx];
  }
}

template <typename T, bool accum>
__global__ void kernel_slice_2d_backward(const int size, const T *dy, T *dx,
                                         const int xstrides, const int ystrides,
                                         const int2 start, const int2 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_2d(yidx, ystrides);
    auto nd_xidx =
        make_int2(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y);
    auto xidx = device_2d_to_flat(nd_xidx, xstrides);
    dx[xidx] = accum ? dy[yidx] + dx[xidx] : dy[yidx];
  }
}

template <typename T>
void slice_2d_forward(const T *x_data, T *y_data, Size_t ndim, Size_t ysize,
                      const Shape_t &xshape, const Shape_t &yshape,
                      const Shape_t &xstrides, const Shape_t &ystrides,
                      const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_2d_forward, ysize, x_data, y_data,
                                 xstrides[0], ystrides[0], to_int2(start),
                                 to_int2(step));
}

template <typename T, bool accum>
void slice_2d_backward(const T *y_grad, T *x_grad, Size_t ndim, Size_t ysize,
                       const Shape_t &xshape, const Shape_t &yshape,
                       const Shape_t &xstrides, const Shape_t &ystrides,
                       const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_slice_2d_backward<T, accum>), ysize,
                                 y_grad, x_grad, xstrides[0], ystrides[0],
                                 to_int2(start), to_int2(step));
}

// 3d slice
template <typename T>
__global__ void kernel_slice_3d_forward(const int size, const T *x, T *y,
                                        const int2 xstrides,
                                        const int2 ystrides, const int3 start,
                                        const int3 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_3d(yidx, ystrides);
    auto nd_xidx =
        make_int3(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y,
                  start.z + nd_yidx.z * step.z);
    auto xidx = device_3d_to_flat(nd_xidx, xstrides);
    y[yidx] = x[xidx];
  }
}

template <typename T, bool accum>
__global__ void kernel_slice_3d_backward(const int size, const T *dy, T *dx,
                                         const int2 xstrides,
                                         const int2 ystrides, const int3 start,
                                         const int3 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_3d(yidx, ystrides);
    auto nd_xidx =
        make_int3(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y,
                  start.z + nd_yidx.z * step.z);
    auto xidx = device_3d_to_flat(nd_xidx, xstrides);
    dx[xidx] = accum ? dy[yidx] + dx[xidx] : dy[yidx];
  }
}

template <typename T>
void slice_3d_forward(const T *x_data, T *y_data, Size_t ndim, Size_t ysize,
                      const Shape_t &xshape, const Shape_t &yshape,
                      const Shape_t &xstrides, const Shape_t &ystrides,
                      const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_3d_forward, ysize, x_data, y_data,
                                 to_int2(xstrides), to_int2(ystrides),
                                 to_int3(start), to_int3(step));
}

template <typename T, bool accum>
void slice_3d_backward(const T *y_grad, T *x_grad, Size_t ndim, Size_t ysize,
                       const Shape_t &xshape, const Shape_t &yshape,
                       const Shape_t &xstrides, const Shape_t &ystrides,
                       const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_slice_3d_backward<T, accum>), ysize, y_grad, x_grad,
      to_int2(xstrides), to_int2(ystrides), to_int3(start), to_int3(step));
}

// 4d slice
template <typename T>
__global__ void kernel_slice_4d_forward(const int size, const T *x, T *y,
                                        const int3 xstrides,
                                        const int3 ystrides, const int4 start,
                                        const int4 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_4d(yidx, ystrides);
    auto nd_xidx =
        make_int4(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y,
                  start.z + nd_yidx.z * step.z, start.w + nd_yidx.w * step.w);
    auto xidx = device_4d_to_flat(nd_xidx, xstrides);
    y[yidx] = x[xidx];
  }
}

template <typename T, bool accum>
__global__ void kernel_slice_4d_backward(const int size, const T *dy, T *dx,
                                         const int3 xstrides,
                                         const int3 ystrides, const int4 start,
                                         const int4 step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_4d(yidx, ystrides);
    auto nd_xidx =
        make_int4(start.x + nd_yidx.x * step.x, start.y + nd_yidx.y * step.y,
                  start.z + nd_yidx.z * step.z, start.w + nd_yidx.w * step.w);
    auto xidx = device_4d_to_flat(nd_xidx, xstrides);
    dx[xidx] = accum ? dy[yidx] + dx[xidx] : dy[yidx];
  }
}

template <typename T>
void slice_4d_forward(const T *x_data, T *y_data, Size_t ndim, Size_t ysize,
                      const Shape_t &xshape, const Shape_t &yshape,
                      const Shape_t &xstrides, const Shape_t &ystrides,
                      const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_4d_forward, ysize, x_data, y_data,
                                 to_int3(xstrides), to_int3(ystrides),
                                 to_int4(start), to_int4(step));
}

template <typename T, bool accum>
void slice_4d_backward(const T *y_grad, T *x_grad, Size_t ndim, Size_t ysize,
                       const Shape_t &xshape, const Shape_t &yshape,
                       const Shape_t &xstrides, const Shape_t &ystrides,
                       const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_slice_4d_backward<T, accum>), ysize, y_grad, x_grad,
      to_int3(xstrides), to_int3(ystrides), to_int4(start), to_int4(step));
}

// nd slice with template
template <typename T, int NDIM>
__global__ void kernel_slice_nd_forward(const int size, const T *x, T *y,
                                        const NdIndex<NDIM> xstrides,
                                        const NdIndex<NDIM> ystrides,
                                        const NdIndex<NDIM> start,
                                        const NdIndex<NDIM> step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_nd(yidx, ystrides);
    NdIndex<NDIM> nd_xidx;
    for (int i = 0; i < NDIM; i++) {
      nd_xidx.nd_idx[i] = start.nd_idx[i] + nd_yidx.nd_idx[i] * step.nd_idx[i];
    }
    auto xidx = device_nd_to_flat(nd_xidx, xstrides);
    y[yidx] = x[xidx];
  }
}

template <typename T, int NDIM>
void slice_nd_forward(const T *x_data, T *y_data, Size_t ndim, Size_t ysize,
                      const Shape_t &xshape, const Shape_t &yshape,
                      const Shape_t &xstrides, const Shape_t &ystrides,
                      const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_slice_nd_forward<T, NDIM>), ysize, x_data, y_data,
      to_nd_index<NDIM>(xstrides), to_nd_index<NDIM>(ystrides),
      to_nd_index<NDIM>(start), to_nd_index<NDIM>(step));
}

template <typename T, int NDIM, bool accum>
__global__ void kernel_slice_nd_backward(const int size, const T *dy, T *dx,
                                         const NdIndex<NDIM> xstrides,
                                         const NdIndex<NDIM> ystrides,
                                         const NdIndex<NDIM> start,
                                         const NdIndex<NDIM> step) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    auto nd_yidx = device_flat_to_nd(yidx, ystrides);
    NdIndex<NDIM> nd_xidx;
    for (int i = 0; i < NDIM; i++) {
      nd_xidx.nd_idx[i] = start.nd_idx[i] + nd_yidx.nd_idx[i] * step.nd_idx[i];
    }
    auto xidx = device_nd_to_flat(nd_xidx, xstrides);
    dx[xidx] = accum ? dy[yidx] + dx[xidx] : dy[yidx];
  }
}

template <typename T, int NDIM, bool accum>
void slice_nd_backward(const T *y_grad, T *x_grad, Size_t ndim, Size_t ysize,
                       const Shape_t &xshape, const Shape_t &yshape,
                       const Shape_t &xstrides, const Shape_t &ystrides,
                       const vector<int> &start, const vector<int> &step) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_slice_nd_backward<T, NDIM, accum>), ysize, y_grad, x_grad,
      to_nd_index<NDIM>(xstrides), to_nd_index<NDIM>(ystrides),
      to_nd_index<NDIM>(start), to_nd_index<NDIM>(step));
}

// nd slice in general
template <typename T>
void slice_nd_forward_loop(const T *x_data, T *y_data, Size_t ndim,
                           Size_t ysize, const Shape_t &xshape,
                           const Shape_t &yshape, const Shape_t &xstrides,
                           const Shape_t &ystrides, const vector<int> &start,
                           const vector<int> &step) {
  // (D_1, ..., D_N) -> (D1 * ...* D_{N-1}, D_N) -> (O, I) ->
  // o -> nd_index_y -> nd_index_x -> x_idx -> slice_1d for each x_idx
  auto inner_size = yshape[ndim - 1];
  auto outer_size = ysize / inner_size;
  Shape_t outer_shape(yshape.begin(), yshape.end() - 1);
  auto outer_strides = ndi::strides(outer_shape);
  auto start_n = start[ndim - 1];
  auto step_n = step[ndim - 1];
  for (Size_t o = 0; o < outer_size; o++) {
    auto nd_yidx = ndi::flat2nd(o, outer_strides);
    Shape_t nd_xidx(ndim);
    for (Size_t d = 0; d < ndim - 1; d++) {
      auto iy = nd_yidx[d];
      auto ix = start[d] + iy * step[d];
      nd_xidx[d] = ix;
    }
    nd_xidx[ndim - 1] = 0;
    auto x_idx = ndi::nd2flat(nd_xidx, xstrides);
    auto y_idx = o * inner_size;
    auto x_o = x_data + x_idx;
    auto y_o = y_data + y_idx;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_slice_1d_forward, inner_size, x_o,
                                   y_o, start_n, step_n);
  }
}

template <typename T, bool accum>
void slice_nd_backward_loop(const T *y_grad, T *x_grad, Size_t ndim,
                            Size_t ysize, const Shape_t &xshape,
                            const Shape_t &yshape, const Shape_t &xstrides,
                            const Shape_t &ystrides, const vector<int> &start,
                            const vector<int> &step) {
  // (D_1, ..., D_N) -> (D1 * ...* D_{N-1}, D_N) -> (O, I) ->
  // o -> nd_index_y -> nd_index_x -> x_idx -> slice_1d for each x_idx
  auto inner_size = yshape[ndim - 1];
  auto outer_size = ysize / inner_size;
  Shape_t outer_shape(yshape.begin(), yshape.end() - 1);
  auto outer_strides = ndi::strides(outer_shape);
  auto start_n = start[ndim - 1];
  auto step_n = step[ndim - 1];
  for (Size_t o = 0; o < outer_size; o++) {
    auto nd_yidx = ndi::flat2nd(o, outer_strides);
    Shape_t nd_xidx(ndim);
    for (Size_t d = 0; d < ndim - 1; d++) {
      auto iy = nd_yidx[d];
      auto ix = start[d] + iy * step[d];
      nd_xidx[d] = ix;
    }
    nd_xidx[ndim - 1] = 0;
    auto x_idx = ndi::nd2flat(nd_xidx, xstrides);
    auto y_idx = o * inner_size;
    auto dx_o = x_grad + x_idx;
    auto dy_o = y_grad + y_idx;
    auto kernel = kernel_slice_1d_backward<T, accum>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, inner_size, dy_o, dx_o, start_n,
                                   step_n);
  }
}

template <typename T>
void SliceCuda<T>::setup_impl(const Variables &inputs,
                              const Variables &outputs) {
  Slice<T>::setup_impl(inputs, outputs);
  if (outputs[0]->size() == 0)
    return;
}

template <typename T>
void SliceCuda<T>::forward_impl(const Variables &inputs,
                                const Variables &outputs) {
  if (outputs[0]->size() == 0)
    return;

  cuda_set_device(std::stoi(this->ctx_.device_id));

  auto x = inputs[0];
  auto y = outputs[0];
  auto start = this->start_[0];
  auto step = this->step_[0];
  auto xshape = x->shape();
  auto yshape = y->shape();
  auto xstrides = x->strides();
  auto ystrides = y->strides();
  auto ndim = x->ndim();
  auto ysize = y->size();
  auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_);

  if (ndim == 1) {
    slice_1d_forward(x_data, y_data, ndim, ysize, xshape, yshape, xstrides,
                     ystrides, start, step);
  } else if (ndim == 2) {
    slice_2d_forward(x_data, y_data, ndim, ysize, xshape, yshape, xstrides,
                     ystrides, start, step);
  } else if (ndim == 3) {
    slice_3d_forward(x_data, y_data, ndim, ysize, xshape, yshape, xstrides,
                     ystrides, start, step);
  } else if (ndim == 4) {
    slice_4d_forward(x_data, y_data, ndim, ysize, xshape, yshape, xstrides,
                     ystrides, start, step);
  } else if (ndim == 5) {
    slice_nd_forward<Tcu, 5>(x_data, y_data, ndim, ysize, xshape, yshape,
                             xstrides, ystrides, start, step);
  } else if (ndim == 6) {
    slice_nd_forward<Tcu, 6>(x_data, y_data, ndim, ysize, xshape, yshape,
                             xstrides, ystrides, start, step);
  } else if (ndim == 7) {
    slice_nd_forward<Tcu, 7>(x_data, y_data, ndim, ysize, xshape, yshape,
                             xstrides, ystrides, start, step);
  } else {
    slice_nd_forward_loop(x_data, y_data, ndim, ysize, xshape, yshape, xstrides,
                          ystrides, start, step);
  }
}

template <typename T>
void SliceCuda<T>::backward_impl(const Variables &inputs,
                                 const Variables &outputs,
                                 const vector<bool> &propagate_down,
                                 const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  if (outputs[0]->size() == 0)
    return;

  cuda_set_device(std::stoi(this->ctx_.device_id));

  auto x = inputs[0];
  auto y = outputs[0];
  auto start = this->start_[0];
  auto step = this->step_[0];
  auto xshape = x->shape();
  auto yshape = y->shape();
  auto xstrides = x->strides();
  auto ystrides = y->strides();
  auto ndim = x->ndim();
  auto ysize = y->size();
  auto x_grad = x->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);

  if (ndim == 1) {
    auto slice =
        accum[0] ? slice_1d_backward<Tcu, true> : slice_1d_backward<Tcu, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 2) {
    auto slice =
        accum[0] ? slice_2d_backward<Tcu, true> : slice_2d_backward<Tcu, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 3) {
    auto slice =
        accum[0] ? slice_3d_backward<Tcu, true> : slice_3d_backward<Tcu, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 4) {
    auto slice =
        accum[0] ? slice_4d_backward<Tcu, true> : slice_4d_backward<Tcu, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 5) {
    auto slice = accum[0] ? slice_nd_backward<Tcu, 5, true>
                          : slice_nd_backward<Tcu, 5, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 6) {
    auto slice = accum[0] ? slice_nd_backward<Tcu, 6, true>
                          : slice_nd_backward<Tcu, 6, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else if (ndim == 7) {
    auto slice = accum[0] ? slice_nd_backward<Tcu, 7, true>
                          : slice_nd_backward<Tcu, 7, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  } else {
    auto slice = accum[0] ? slice_nd_backward_loop<Tcu, true>
                          : slice_nd_backward_loop<Tcu, false>;
    slice(y_grad, x_grad, ndim, ysize, xshape, yshape, xstrides, ystrides,
          start, step);
  }
}
}

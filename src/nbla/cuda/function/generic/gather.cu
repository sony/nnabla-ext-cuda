// Copyright 2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/gather.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

#include <functional>
#include <numeric>

namespace nbla {

template <typename T>
__global__ void kernel_gather_forward(const int size, T *y_data,
                                      const T *x_data, const int *i_data,
                                      int2 xstrides_f, int istrides_f,
                                      int2 ystrides_f, int lstrides_f) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    // nd_index of y
    auto nd_yidx = device_flat_to_3d(yidx, ystrides_f);
    auto i = nd_yidx.x;
    auto j = nd_yidx.y;
    auto k = nd_yidx.z;
    // nd_index of leading one and i
    auto nd_lidx = device_flat_to_2d(i, lstrides_f);
    auto b = nd_lidx.x;
    auto nd_iidx = make_int2(b, j);
    auto iidx = device_2d_to_flat(nd_iidx, istrides_f);
    auto g = i_data[iidx];
    // nd_index of x
    auto nd_xidx = make_int3(i, g, k);
    auto xidx = device_3d_to_flat(nd_xidx, xstrides_f);
    y_data[yidx] = x_data[xidx];
  }
}

template <typename T>
__global__ void kernel_gather_backward(const int size, T *x_grad,
                                       const T *y_grad, const int *i_data,
                                       int2 xstrides_f, int istrides_f,
                                       int2 ystrides_f, int lstrides_f) {
  NBLA_CUDA_KERNEL_LOOP(yidx, size) {
    // nd_index of y
    auto nd_yidx = device_flat_to_3d(yidx, ystrides_f);
    auto i = nd_yidx.x;
    auto j = nd_yidx.y;
    auto k = nd_yidx.z;
    // nd_index of leading one and i
    auto nd_lidx = device_flat_to_2d(i, lstrides_f);
    auto b = nd_lidx.x;
    auto nd_iidx = make_int2(b, j);
    auto iidx = device_2d_to_flat(nd_iidx, istrides_f);
    auto g = i_data[iidx];
    // nd_index of x
    auto nd_xidx = make_int3(i, g, k);
    auto xidx = device_3d_to_flat(nd_xidx, xstrides_f);
    atomic_add(x_grad + xidx, y_grad[yidx]);
  }
}

template <typename T>
void GatherCuda<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {
  Gather<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void GatherCuda<T>::forward_impl(const Variables &inputs,
                                 const Variables &outputs) {
  cuda_set_device(this->device_);

  auto x = inputs[0];
  auto indices = inputs[1];
  auto y = outputs[0];
  auto xshape = x->shape();
  auto ishape = indices->shape();
  auto yshape = y->shape();

  auto prod = [](Shape_t &shape, int b, int e) {
    return std::accumulate(shape.begin() + b, shape.begin() + e, 1,
                           std::multiplies<int64_t>());
  };

  // Fold xshape: prod(xshape[:axis]), xshape[axis], prod(xshape[axis+1:])
  auto xsize0 = prod(xshape, 0, this->axis_);
  auto xsize1 = xshape[this->axis_];
  auto xsize2 = prod(xshape, this->axis_ + 1, xshape.size());
  auto xstrides_f = make_int2(xsize1 * xsize2, xsize2);
  // Fold ishape: prod(ishape[:batch_dims]), prod(ishape[batch_dims:])
  auto isize0 = prod(ishape, 0, this->batch_dims_);
  auto isize1 = prod(ishape, this->batch_dims_, ishape.size());
  auto istrides_f = isize1;
  // Fold yshape: prod(yshape[:axis]), prod(ishape[batch_dims:]),
  // prod(xshape[axis+1:])
  auto ysize0 = prod(yshape, 0, this->axis_);
  auto ysize1 = isize1;
  auto ysize2 = xsize2;
  auto ystrides_f = make_int2(ysize1 * ysize2, ysize2);
  // Fold leading dimensions of y
  auto bsize = isize0;
  auto leading_dsize = ysize0 / bsize;
  auto lstrides_f = leading_dsize;
  // Gather
  auto size = y->size();
  auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto i_data = indices->get_data_pointer<int>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_gather_forward<Tcu>), size, y_data,
                                 x_data, i_data, xstrides_f, istrides_f,
                                 ystrides_f, lstrides_f);
}

template <typename T>
void GatherCuda<T>::backward_impl(const Variables &inputs,
                                  const Variables &outputs,
                                  const vector<bool> &propagate_down,
                                  const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(this->device_);

  auto x = inputs[0];
  auto indices = inputs[1];
  auto y = outputs[0];
  auto xshape = x->shape();
  auto ishape = indices->shape();
  auto yshape = y->shape();

  auto prod = [](Shape_t &shape, int b, int e) {
    return std::accumulate(shape.begin() + b, shape.begin() + e, 1,
                           std::multiplies<int>());
  };

  // Fold xshape: prod(xshape[:axis]), xshape[axis], prod(xshape[axis+1:])
  auto xsize0 = prod(xshape, 0, this->axis_);
  auto xsize1 = xshape[this->axis_];
  auto xsize2 = prod(xshape, this->axis_ + 1, xshape.size());
  auto xstrides_f = make_int2(xsize1 * xsize2, xsize2);
  // Fold ishape: prod(ishape[:batch_dims]), prod(ishape[batch_dims:])
  auto isize0 = prod(ishape, 0, this->batch_dims_);
  auto isize1 = prod(ishape, this->batch_dims_, ishape.size());
  auto istrides_f = isize1;
  // Fold yshape: prod(yshape[:axis]), prod(ishape[batch_dims:]),
  // prod(xshape[axis+1:])
  auto ysize0 = prod(yshape, 0, this->axis_);
  auto ysize1 = isize1;
  auto ysize2 = xsize2;
  auto ystrides_f = make_int2(ysize1 * ysize2, ysize2);
  // Fold leading dimensions of y
  auto bsize = isize0;
  auto leading_dsize = ysize0 / bsize;
  auto lstrides_f = leading_dsize;
  // Gather
  auto size = y->size();
  auto x_grad = x->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  auto i_data = indices->get_data_pointer<int>(this->ctx_);
  auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);
  auto kernel = kernel_gather_backward<Tcu>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, x_grad, y_grad, i_data,
                                 xstrides_f, istrides_f, ystrides_f,
                                 lstrides_f);
}
}

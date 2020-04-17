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
#include <nbla/cuda/function/unpooling.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/cuda/utils/nd_index.cuh>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_forward_1d(const int osize, T *dst, const T *src,
                            int outer_size, const int iinner_size,
                            const int oinner_size, const int istride,
                            const int ostride, const int kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_2d(oidx, ostride);
    auto oc = channel_last ? ond_index.y : 0;
    auto ow = ond_index.x;
    auto iw = ow / kernel;
    auto ind_index = make_int2(iw, oc);
    auto iidx = device_2d_to_flat(ind_index, istride);
    do {
      dst[oidx] = src[iidx];
      src += iinner_size;
      dst += oinner_size;
    } while (--outer_size);
  }
}

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_forward_2d(const int osize, T *dst, const T *src,
                            int outer_size, const int iinner_size,
                            const int oinner_size, const int2 istride,
                            const int2 ostride, const int2 kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto oc = channel_last ? ond_index.z : 0;
    auto oh = ond_index.x;
    auto ow = ond_index.y;
    auto ih = oh / kernel.x;
    auto iw = ow / kernel.y;
    auto ind_index = make_int3(ih, iw, oc);
    auto iidx = device_3d_to_flat(ind_index, istride);
    do {
      dst[oidx] = src[iidx];
      src += iinner_size;
      dst += oinner_size;
    } while (--outer_size);
  }
}

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_forward_3d(const int osize, T *dst, const T *src,
                            int outer_size, const int iinner_size,
                            const int oinner_size, const int3 istride,
                            const int3 ostride, const int3 kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto oc = channel_last ? ond_index.w : 0;
    auto od = ond_index.x;
    auto oh = ond_index.y;
    auto ow = ond_index.z;
    auto id = od / kernel.x;
    auto ih = oh / kernel.y;
    auto iw = ow / kernel.z;
    auto ind_index = make_int4(id, ih, iw, oc);
    auto iidx = device_4d_to_flat(ind_index, istride);
    do {
      dst[oidx] = src[iidx];
      src += iinner_size;
      dst += oinner_size;
    } while (--outer_size);
  }
}

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_backward_1d(const int osize, T *dst, const T *src,
                             int outer_size, const int iinner_size,
                             const int oinner_size, const int istride,
                             const int ostride, const int kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_2d(oidx, ostride);
    auto oc = channel_last ? ond_index.y : 0;
    auto ow = ond_index.x;
    auto iw = ow / kernel;
    auto ind_index = make_int2(iw, oc);
    auto iidx = device_2d_to_flat(ind_index, istride);
    do {
      atomic_add(dst + iidx, src[oidx]);
      src += oinner_size;
      dst += iinner_size;
    } while (--outer_size);
  }
}

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_backward_2d(const int osize, T *dst, const T *src,
                             int outer_size, const int iinner_size,
                             const int oinner_size, const int2 istride,
                             const int2 ostride, const int2 kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_3d(oidx, ostride);
    auto oc = channel_last ? ond_index.z : 0;
    auto oh = ond_index.x;
    auto ow = ond_index.y;
    auto ih = oh / kernel.x;
    auto iw = ow / kernel.y;
    auto ind_index = make_int3(ih, iw, oc);
    auto iidx = device_3d_to_flat(ind_index, istride);
    do {
      atomic_add(dst + iidx, src[oidx]);
      src += oinner_size;
      dst += iinner_size;
    } while (--outer_size);
  }
}

template <typename T, bool channel_last = false>
__global__ void
kernel_unpooling_backward_3d(const int osize, T *dst, const T *src,
                             int outer_size, const int iinner_size,
                             const int oinner_size, const int3 istride,
                             const int3 ostride, const int3 kernel) {
  NBLA_CUDA_KERNEL_LOOP(oidx, osize) {
    auto ond_index = device_flat_to_4d(oidx, ostride);
    auto oc = channel_last ? ond_index.w : 0;
    auto od = ond_index.x;
    auto oh = ond_index.y;
    auto ow = ond_index.z;
    auto id = od / kernel.x;
    auto ih = oh / kernel.y;
    auto iw = ow / kernel.z;
    auto ind_index = make_int4(id, ih, iw, oc);
    auto iidx = device_4d_to_flat(ind_index, istride);
    do {
      atomic_add(dst + iidx, src[oidx]);
      src += oinner_size;
      dst += iinner_size;
    } while (--outer_size);
  }
}

template <typename T>
void UnpoolingCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(this->device_);

  auto x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  auto size = outputs[0]->size();
  auto ndim = inputs[0]->ndim();
  auto kdim = this->kernel_.size();

  auto ishape = inputs[0]->shape();
  auto oshape = outputs[0]->shape();
  if (kdim == 1) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * ow : ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * iw : iw;
    auto oinner_size = osize;
    auto istride = this->channel_last_ ? (ic) : 1;
    auto ostride = this->channel_last_ ? (oc) : 1;
    auto kernel = this->kernel_[0];
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_forward_1d<Tc, true>
                           : kernel_unpooling_forward_1d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, y, x, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else if (kdim == 2) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 3];
    auto oh = this->channel_last_ ? oshape[ndim - 3] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 3];
    auto ih = this->channel_last_ ? ishape[ndim - 3] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * oh * ow : oh * ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * ih * iw : ih * iw;
    auto oinner_size = osize;
    auto istride =
        this->channel_last_ ? make_int2(iw * ic, ic) : make_int2(iw, 1);
    auto ostride =
        this->channel_last_ ? make_int2(ow * oc, oc) : make_int2(ow, 1);
    auto kernel = make_int2(this->kernel_[0], this->kernel_[1]);
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_forward_2d<Tc, true>
                           : kernel_unpooling_forward_2d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, y, x, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else if (kdim == 3) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 4];
    auto od = this->channel_last_ ? oshape[ndim - 4] : oshape[ndim - 3];
    auto oh = this->channel_last_ ? oshape[ndim - 3] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 4];
    auto id = this->channel_last_ ? ishape[ndim - 4] : ishape[ndim - 3];
    auto ih = this->channel_last_ ? ishape[ndim - 3] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * od * oh * ow : od * oh * ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * id * ih * iw : id * ih * iw;
    auto oinner_size = osize;
    auto istride = this->channel_last_ ? make_int3(ih * iw * ic, iw * ic, ic)
                                       : make_int3(ih * iw, iw, 1);
    auto ostride = this->channel_last_ ? make_int3(oh * ow * oc, ow * oc, oc)
                                       : make_int3(oh * ow, ow, 1);
    auto kernel =
        make_int3(this->kernel_[0], this->kernel_[1], this->kernel_[2]);
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_forward_3d<Tc, true>
                           : kernel_unpooling_forward_3d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, y, x, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else {
    NBLA_ERROR(error_code::value, "1D, 2D, 3D unpooling are supported.");
  }
}

template <typename T>
void UnpoolingCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }
  cuda_set_device(this->device_);

  auto dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
  auto dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
  auto size = outputs[0]->size();
  auto ndim = inputs[0]->ndim();
  auto kdim = this->kernel_.size();

  auto ishape = inputs[0]->shape();
  auto oshape = outputs[0]->shape();
  if (kdim == 1) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * ow : ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * iw : iw;
    auto oinner_size = osize;
    auto istride = this->channel_last_ ? (ic) : 1;
    auto ostride = this->channel_last_ ? (oc) : 1;
    auto kernel = this->kernel_[0];
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_backward_1d<Tc, true>
                           : kernel_unpooling_backward_1d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, dx, dy, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else if (kdim == 2) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 3];
    auto oh = this->channel_last_ ? oshape[ndim - 3] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 3];
    auto ih = this->channel_last_ ? ishape[ndim - 3] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * oh * ow : oh * ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * ih * iw : ih * iw;
    auto oinner_size = osize;
    auto istride =
        this->channel_last_ ? make_int2(iw * ic, ic) : make_int2(iw, 1);
    auto ostride =
        this->channel_last_ ? make_int2(ow * oc, oc) : make_int2(ow, 1);
    auto kernel = make_int2(this->kernel_[0], this->kernel_[1]);
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_backward_2d<Tc, true>
                           : kernel_unpooling_backward_2d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, dx, dy, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else if (kdim == 3) {
    auto oc = this->channel_last_ ? oshape[ndim - 1] : oshape[ndim - 4];
    auto od = this->channel_last_ ? oshape[ndim - 4] : oshape[ndim - 3];
    auto oh = this->channel_last_ ? oshape[ndim - 3] : oshape[ndim - 2];
    auto ow = this->channel_last_ ? oshape[ndim - 2] : oshape[ndim - 1];
    auto ic = this->channel_last_ ? ishape[ndim - 1] : ishape[ndim - 4];
    auto id = this->channel_last_ ? ishape[ndim - 4] : ishape[ndim - 3];
    auto ih = this->channel_last_ ? ishape[ndim - 3] : ishape[ndim - 2];
    auto iw = this->channel_last_ ? ishape[ndim - 2] : ishape[ndim - 1];
    auto osize = this->channel_last_ ? oc * od * oh * ow : od * oh * ow;
    auto outer_size = size / osize;
    auto iinner_size = this->channel_last_ ? ic * id * ih * iw : id * ih * iw;
    auto oinner_size = osize;
    auto istride = this->channel_last_ ? make_int3(ih * iw * ic, iw * ic, ic)
                                       : make_int3(ih * iw, iw, 1);
    auto ostride = this->channel_last_ ? make_int3(oh * ow * oc, ow * oc, oc)
                                       : make_int3(oh * ow, ow, 1);
    auto kernel =
        make_int3(this->kernel_[0], this->kernel_[1], this->kernel_[2]);
    auto cuda_kernel = this->channel_last_
                           ? kernel_unpooling_backward_3d<Tc, true>
                           : kernel_unpooling_backward_3d<Tc, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(cuda_kernel, osize, dx, dy, outer_size,
                                   iinner_size, oinner_size, istride, ostride,
                                   kernel);
  } else {
    NBLA_ERROR(error_code::value, "Only 1D, 2D, 3D unpooling are supported.");
  }
}
}

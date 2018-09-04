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
#include <nbla/variable.hpp>

namespace nbla {
template <typename T>
T get_interpolate_scale_from_dest(int src, int dst, bool align_corners) {
  if (dst == 1)
    return 0;
  return align_corners ? ((T)(src - 1) / (dst - 1)) : ((T)(src) / dst);
}

template <typename T>
__device__ __forceinline__ T get_interpolate_source_index(T scale, int dst,
                                                          bool align_corners) {
  return align_corners ? (scale * dst)
                       : (max((T)0, scale * (dst + (T)0.5) - (T)0.5));
}

template <typename T>
__global__ void kernel_bilinear_interpolate_2d(
    const int num_kernels, const T *in, T *out, int outer_dim, int iw, int ih,
    int ow, int oh, bool align_corners, typename CudaTypeForceFloat<T>::type sx,
    typename CudaTypeForceFloat<T>::type sy) {
  typedef typename CudaTypeForceFloat<T>::type Tf;
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_kernels)
    return;
  const int inner_dim = iw * ih;
  const int inner_dim_o = ow * oh;
  const int oy = index / ow;
  const int ox = index % ow;
  const Tf fy = get_interpolate_source_index(sy, oy, align_corners);
  const int y = fy;
  const int yp1 = min(y + 1, ih - 1);
  const Tf ly1 = fy - y;
  const Tf ly0 = (Tf)1 - ly1;
  const Tf fx = get_interpolate_source_index(sx, ox, align_corners);
  const int x = fx;
  const int xp1 = min(x + 1, iw - 1);
  const Tf lx1 = fx - x;
  const Tf lx0 = (Tf)1 - lx1;
  for (int o = 0; o < outer_dim; o++) {
#define _I(o, y, x) ((o)*inner_dim + (y)*iw + (x))
    const Tf val0 = ly0 * (lx0 * in[_I(o, y, x)] + lx1 * in[_I(o, y, xp1)]);
    const Tf val1 = ly1 * (lx0 * in[_I(o, yp1, x)] + lx1 * in[_I(o, yp1, xp1)]);
#undef _I
    out[o * inner_dim_o + oy * ow + ox] = val0 + val1;
  }
}

template <typename T>
__global__ void kernel_bilinear_interpolate_2d_backward(
    const int num_kernels, typename CudaTypeForceFloat<T>::type *gin,
    const T *gout, int outer_dim, int iw, int ih, int ow, int oh,
    bool align_corners, typename CudaTypeForceFloat<T>::type sx,
    typename CudaTypeForceFloat<T>::type sy) {
  typedef typename CudaTypeForceFloat<T>::type Tf;
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= num_kernels)
    return;
  const int inner_dim = iw * ih;
  const int inner_dim_o = ow * oh;
  const int oy = index / ow;
  const int ox = index % ow;
  const Tf fy = get_interpolate_source_index(sy, oy, align_corners);
  const int y = fy;
  const int yp1 = (y < ih - 1) ? (y + 1) : y;
  const Tf ly1 = fy - y;
  const Tf ly0 = (Tf)1 - ly1;
  const Tf fx = get_interpolate_source_index(sx, ox, align_corners);
  const int x = fx;
  const int xp1 = (x < iw - 1) ? (x + 1) : x;
  const Tf lx1 = fx - x;
  const Tf lx0 = (Tf)1 - lx1;
  for (int o = 0; o < outer_dim; o++) {
    const Tf g = gout[o * inner_dim_o + oy * ow + ox];
#define _I(o, y, x) ((o)*inner_dim + (y)*iw + (x))
    const Tf v0 = ly0 * lx0 * g;
    const Tf v1 = ly0 * lx1 * g;
    const Tf v2 = ly1 * lx0 * g;
    const Tf v3 = ly1 * lx1 * g;
    atomicAdd(gin + _I(o, y, x), v0);
    atomicAdd(gin + _I(o, y, xp1), v1);
    atomicAdd(gin + _I(o, yp1, x), v2);
    atomicAdd(gin + _I(o, yp1, xp1), v3);
#undef _I
  }
}

template <typename T>
void InterpolateCuda<T>::forward_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  // Inputs
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);

  // Outputs
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  // 2D bilinear
  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];
  const int outer_dim = inputs[0]->size() / (iw * ih);
  const bool align_corners = this->align_corners_;
  typedef typename CudaTypeForceFloat<Tcu>::type U;
  const U sx = get_interpolate_scale_from_dest<U>(iw, ow, align_corners);
  const U sy = get_interpolate_scale_from_dest<U>(ih, oh, align_corners);

  // Invoke kernels
  const int num_kernels = ow * oh;
  cudaDeviceProp prop = cuda_get_current_device_properties();
  const int num_threads = prop.maxThreadsPerBlock;
  kernel_bilinear_interpolate_2d<<<NBLA_CEIL_INT_DIV(num_kernels, num_threads),
                                   num_threads>>>(
      num_kernels, x, y, outer_dim, iw, ih, ow, oh, align_corners, sx, sy);
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void InterpolateCuda<T>::backward_impl(const Variables &inputs,
                                       const Variables &outputs,
                                       const vector<bool> &propagate_down,
                                       const vector<bool> &accum) {
  // TODO: half backward, i.e. atomicAdd in half.

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  typedef typename CudaTypeForceFloat<Tcu>::type U;
  U *g_x = inputs[0]->cast_grad_and_get_pointer<U>(this->ctx_);

  // 2D bilinear
  const int ndim = inputs[0]->ndim();
  const int iw = inputs[0]->shape()[ndim - 1];
  const int ih = inputs[0]->shape()[ndim - 2];
  const int ow = outputs[0]->shape()[ndim - 1];
  const int oh = outputs[0]->shape()[ndim - 2];
  const int outer_dim = inputs[0]->size() / (iw * ih);
  const bool align_corners = this->align_corners_;
  const U sx = get_interpolate_scale_from_dest<U>(iw, ow, align_corners);
  const U sy = get_interpolate_scale_from_dest<U>(ih, oh, align_corners);

  // Invoke kernels
  const int num_kernels = ow * oh;
  cudaDeviceProp prop = cuda_get_current_device_properties();
  const int num_threads = prop.maxThreadsPerBlock;
  kernel_bilinear_interpolate_2d_backward<<<
      NBLA_CEIL_INT_DIV(num_kernels, num_threads), num_threads>>>(
      num_kernels, g_x, g_y, outer_dim, iw, ih, ow, oh, align_corners, sx, sy);
  NBLA_CUDA_KERNEL_CHECK();
}
}

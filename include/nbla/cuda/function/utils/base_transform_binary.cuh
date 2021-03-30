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

/** Base class of binary operations for CUDA.
 */
#ifndef __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_BINARY_CUH__
#define __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_BINARY_CUH__

#include <nbla/cuda/function/utils/base_transform_binary.hpp>
#include <nbla/cuda/half.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>

#include <assert.h>
#include <string>
#include <tuple>

// Note:
// The all kernels in this file tries to use more precise type intermediately
// (e.g. float instead of half) to keep the numerical precision.

namespace nbla {
using std::tuple;
using std::string;
using std::is_same;

// ----------------------------------------------------------------------------
// Base class to store a binary operation
// ----------------------------------------------------------------------------
class BaseBinaryOpCuda {
public:
  template <typename T>
  __forceinline__ __device__ T operator()(const T x0, const T x1) {
    return 0;
  }
  template <typename T>
  __forceinline__ __device__ T g0(const T dy, const T x0, const T x1, const T y,
                                  const bool inplace) {
    return 0;
  }
  template <typename T>
  __forceinline__ __device__ T g1(const T dy, const T x0, const T x1, const T y,
                                  const bool inplace) {
    return 0;
  }
  __host__ void verify_g0() {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 0 is not implemented.");
  }
  __host__ void verify_g1() {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 1 is not implemented.");
  }
};

namespace transform_binary_cuda {
// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------
#define TRANSFORM_BINARY_DIV_SIZE 128

/** CUDA grid-strided loop of Size_t*/
#define TRANSFORM_BINARY_CUDA_KERNEL_LOOP(idx, num)                            \
  for (Size_t idx = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;             \
       idx < (num); idx += (Size_t)blockDim.x * gridDim.x)

enum Term { x0, x1 };

struct Dim3KernelParams {
  Size_t stride_x0_0, stride_x0_1, stride_x0_2;
  Size_t stride_x1_0, stride_x1_1, stride_x1_2;
  Size_t stride_y_0, stride_y_1, stride_y_2;
  Size_t shape_y_0, shape_y_1, shape_y_2;
};

/* Get the block size for the kernels specialized for three-dimensional data.

 If #dim of blocks is 3, blockDim.x * blockDim.y * blockDim.z <= 512
 If #dim of blocks is 2, blockDim.x * blockDim.y <= 512, blockDim.z = 1
 If #dim of blocks is 1, blockDim.x <= 512, blockDim.y = 1, blockDim.z = 1

 Template is used to suppress the duplicated definition of this function
 among binary operators. It would be avoidable by declaration and definition
 file separation.
*/
template <typename T, typename BinaryOp>
dim3 get_blocks_dim3(const Shape_t &shape, const int num_block_dims) {
  if (shape.size() != 3) {
    NBLA_ERROR(error_code::value, "Shape is not three-dimensional.");
  }

  // blockDim.x = 2^pow[2], blockDim.x = 2^pow[1], blockDim.x = 2^pow[0],
  Size_t pow[3] = {0, 0, 0};

  // Get the smallest power of 2 greater than or equal to each shape
  pow[2] = (Size_t)std::ceil(std::log2(shape[2]));
  if (num_block_dims != 1)
    pow[1] = (Size_t)std::ceil(std::log2(shape[1]));
  if (num_block_dims == 3)
    pow[0] = (Size_t)std::ceil(std::log2(shape[0]));

  // Adjast block sizes <= 512 (it can be converted from Shape_t to dim3 type.)
  pow[2] = std::min(pow[2], (Size_t)9);
  if (num_block_dims != 1)
    pow[1] = std::min(pow[1], 9 - pow[2]);
  if (num_block_dims == 3)
    pow[0] = std::min(pow[0], 9 - pow[2] - pow[1]);

  // Power and return
  return dim3(1 << pow[2], 1 << pow[1], 1 << pow[0]);
}

inline __device__ void get_indices(Size_t *idxes /* Size_t[2] */, Size_t idx,
                                   const Size_t ndim, const Size_t *strides_x0,
                                   const Size_t *strides_x1,
                                   const Size_t *strides_y,
                                   const Size_t *shape_y) {
  idxes[0] = 0;
  idxes[1] = 0;
  for (Size_t i = 0; i < ndim; ++i) {
    const Size_t dim_idx = idx / strides_y[i];
    idxes[0] += dim_idx * strides_x0[i];
    idxes[1] += dim_idx * strides_x1[i];
    idx -= dim_idx * strides_y[i];
  }
}

inline __device__ Size_t flatten_idx(const Size_t x, const Size_t y,
                                     const Size_t z, const Size_t stride_0,
                                     const Size_t stride_1,
                                     const Size_t stride_2) {
  return x * stride_0 + y * stride_1 + z * stride_2;
}

// This kernel is used to store the results into the required type
// (e.g. half to float). This process is unnecessary when T == PRECISE_T
// (e.g. T == float).
template <typename T, typename PRECISE_T, bool accum>
__global__ void kernel_precise_add(const Size_t size, T *dst,
                                   const PRECISE_T *src) {
  TRANSFORM_BINARY_CUDA_KERNEL_LOOP(idx, size) {
    dst[idx] = (accum ? (PRECISE_T)dst[idx] : (PRECISE_T)0) + src[idx];
  }
}

// ----------------------------------------------------------------------------
// Forward kernels for any dimensions
// ----------------------------------------------------------------------------
// Forward
template <typename T, typename PRECISE_T, typename BinaryOp>
__global__ void
kernel_forward_ndim(const Size_t size, BinaryOp op, const T *__restrict__ x0,
                    const T *__restrict__ x1, T *y, const Size_t ndim,
                    const Size_t *strides_x0, const Size_t *strides_x1,
                    const Size_t *strides_y, const Size_t *shape_y) {
  TRANSFORM_BINARY_CUDA_KERNEL_LOOP(idx, size) {
    Size_t idxes[2];
    get_indices(idxes, idx, ndim, strides_x0, strides_x1, strides_y, shape_y);
    y[idx] = op(static_cast<PRECISE_T>(x0[idxes[0]]),
                static_cast<PRECISE_T>(x1[idxes[1]]));
  }
}

// Backward
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
__global__ void
kernel_backward_ndim(const Size_t size, BinaryOp op, const T *__restrict__ dy,
                     const T *__restrict__ x0, const T *__restrict__ x1,
                     const T *__restrict__ y, PRECISE_T *dst,
                     const bool inplace, const Size_t ndim,
                     const Size_t *strides_x0, const Size_t *strides_x1,
                     const Size_t *strides_y, const Size_t *shape_y) {
  TRANSFORM_BINARY_CUDA_KERNEL_LOOP(idx, size) {
    Size_t idxes[2];
    get_indices(idxes, idx, ndim, strides_x0, strides_x1, strides_y, shape_y);
    if (term == Term::x0) {
      atomic_add(&dst[idxes[0]],
                 op.g0(static_cast<PRECISE_T>(dy[idx]),
                       static_cast<PRECISE_T>(x0[idxes[0]]),
                       static_cast<PRECISE_T>(x1[idxes[1]]),
                       static_cast<PRECISE_T>(y[idx]), inplace));
    } else {
      atomic_add(&dst[idxes[1]],
                 op.g1(static_cast<PRECISE_T>(dy[idx]),
                       static_cast<PRECISE_T>(x0[idxes[0]]),
                       static_cast<PRECISE_T>(x1[idxes[1]]),
                       static_cast<PRECISE_T>(y[idx]), inplace));
    }
  }
}

// ----------------------------------------------------------------------------
// Forward kernels for three dimensions
// ----------------------------------------------------------------------------
// Perform binary operation while broadcasting by using only strides.
template <typename T, typename PRECISE_T, typename BinaryOp>
__global__ void kernel_forward_dim3(BinaryOp op, const T *__restrict__ x0,
                                    const T *__restrict__ x1, T *y,
                                    const Dim3KernelParams p) {
  const Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const Size_t iy = (Size_t)blockIdx.y * blockDim.y + threadIdx.y;
  const Size_t iz = (Size_t)blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= p.shape_y_2 || iy >= p.shape_y_1 || iz >= p.shape_y_0)
    return;

  const Size_t idx =
      flatten_idx(ix, iy, iz, p.stride_y_2, p.stride_y_1, p.stride_y_0);
  const Size_t idx0 =
      flatten_idx(ix, iy, iz, p.stride_x0_2, p.stride_x0_1, p.stride_x0_0);
  const Size_t idx1 =
      flatten_idx(ix, iy, iz, p.stride_x1_2, p.stride_x1_1, p.stride_x1_0);
  y[idx] =
      op(static_cast<PRECISE_T>(x0[idx0]), static_cast<PRECISE_T>(x1[idx1]));
}

// Perform binary operation without broadcast for both terms
template <typename T, typename PRECISE_T, typename BinaryOp>
__global__ void kernel_forward_dim3_not_broadcasted_both_terms(
    BinaryOp op, const T *__restrict__ x0, const T *__restrict__ x1, T *y,
    const Dim3KernelParams p) {
  const Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= p.shape_y_2)
    return;

  const Size_t idx = ix * p.stride_y_2;
  const Size_t idx0 = ix * p.stride_x0_2;
  const Size_t idx1 = ix * p.stride_x1_2;
  y[idx] =
      op(static_cast<PRECISE_T>(x0[idx0]), static_cast<PRECISE_T>(x1[idx1]));
}

// ----------------------------------------------------------------------------
// Backward kernels without reduction for three dimensions
// ----------------------------------------------------------------------------
// The case where th specified "term" by template is not broadcasted.
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
__global__ void kernel_backward_dim3_broadcasted_other_term(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y, T *dx,
    const bool inplace, const Dim3KernelParams p) {
  const Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const Size_t iy = (Size_t)blockIdx.y * blockDim.y + threadIdx.y;
  const Size_t iz = (Size_t)blockIdx.z * blockDim.z + threadIdx.z;
  if (ix >= p.shape_y_2 || iy >= p.shape_y_1 || iz >= p.shape_y_0)
    return;

  const Size_t idx =
      flatten_idx(ix, iy, iz, p.stride_y_2, p.stride_y_1, p.stride_y_0);
  const Size_t idx0 =
      flatten_idx(ix, iy, iz, p.stride_x0_2, p.stride_x0_1, p.stride_x0_0);
  const Size_t idx1 =
      flatten_idx(ix, iy, iz, p.stride_x1_2, p.stride_x1_1, p.stride_x1_0);

  if (term == Term::x0) {
    dx[idx0] =
        static_cast<PRECISE_T>(dx[idx0]) +
        op.g0(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  } else {
    dx[idx1] =
        static_cast<PRECISE_T>(dx[idx1]) +
        op.g1(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  }
}

// The case where both term are not broadcasted can reduce the computational
// complexity at the index calculation because the dimension is compressed to
// one.
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
__global__ void kernel_backward_dim3_not_broadcasted_both_terms(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y, T *dx,
    const bool inplace, const Dim3KernelParams p) {
  Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= p.shape_y_2)
    return;

  const Size_t idx = ix * p.stride_y_2;
  const Size_t idx0 = ix * p.stride_x0_2;
  const Size_t idx1 = ix * p.stride_x1_2;

  if (term == Term::x0) {
    dx[idx0] =
        static_cast<PRECISE_T>(dx[idx0]) +
        op.g0(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  } else {
    dx[idx1] =
        static_cast<PRECISE_T>(dx[idx1]) +
        op.g1(static_cast<PRECISE_T>(dy[idx]), static_cast<PRECISE_T>(x0[idx0]),
              static_cast<PRECISE_T>(x1[idx1]), static_cast<PRECISE_T>(y[idx]),
              inplace);
  }
}

// ----------------------------------------------------------------------------
// Backward kernels of x-axis reduction for three dimensions
// ----------------------------------------------------------------------------
// Sub-routine of x-axis reduction
template <typename PRECISE_T, Size_t blockSize>
__device__ PRECISE_T kernel_backward_dim3_block_reduce_x(Size_t tid,
                                                         PRECISE_T *buf) {
  if (blockSize >= 512 && tid < 256) {
    buf[tid] += buf[tid + 256];
  }
  __syncthreads();
  if (blockSize >= 256 && tid < 128) {
    buf[tid] += buf[tid + 128];
  }
  __syncthreads();
  if (blockSize >= 128 && tid < 64) {
    buf[tid] += buf[tid + 64];
  }
  __syncthreads();

  // warp reduce
  PRECISE_T sum = buf[tid];
  if (tid < 32) {
    if (blockSize >= 64) {
      sum += buf[tid + 32];
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }
  return sum;
}

// x-axis reduction after z-axis reduction
template <typename T, typename PRECISE_T, Size_t blockSize>
__global__ void kernel_backward_dim3_reduce_x_after_z(const PRECISE_T *src,
                                                      T *dst,
                                                      const Size_t x_size) {
  const Size_t tid = threadIdx.x;
  const Size_t y = blockIdx.y;

  extern __shared__ PRECISE_T sbuf[];

  PRECISE_T sum = 0;
  for (Size_t x = tid; x < x_size; x += blockSize) {
    sum += (PRECISE_T)src[x + y * x_size];
  }

  sbuf[tid] = sum;
  __syncthreads();

  sum = kernel_backward_dim3_block_reduce_x<PRECISE_T, blockSize>(tid, sbuf);

  if (tid == 0) {
    dst[y] = (PRECISE_T)dst[y] + sum;
  }
}

// x-axis reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term,
          Size_t blockSize>
__global__ void kernel_backward_dim3_reduce_x(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y, T *dx,
    const bool inplace, const Dim3KernelParams p, const Size_t x_size,
    const Size_t y_size, const Size_t z_size) {
  const Size_t tid = threadIdx.x;
  const Size_t iy = (Size_t)blockIdx.y * blockDim.y + threadIdx.y;
  const Size_t iz = (Size_t)blockIdx.z * blockDim.z + threadIdx.z;
  if (iy >= y_size || iz >= z_size)
    return;

  extern __shared__ PRECISE_T sbuf[];

  PRECISE_T sum = 0;
  for (Size_t ix = tid; ix < x_size; ix += blockSize) {
    const Size_t idx =
        flatten_idx(ix, iy, iz, p.stride_y_2, p.stride_y_1, p.stride_y_0);
    const Size_t idx0 =
        flatten_idx(ix, iy, iz, p.stride_x0_2, p.stride_x0_1, p.stride_x0_0);
    const Size_t idx1 =
        flatten_idx(ix, iy, iz, p.stride_x1_2, p.stride_x1_1, p.stride_x1_0);
    if (term == Term::x0) {
      sum += op.g0(static_cast<PRECISE_T>(dy[idx]),
                   static_cast<PRECISE_T>(x0[idx0]),
                   static_cast<PRECISE_T>(x1[idx1]),
                   static_cast<PRECISE_T>(y[idx]), inplace);
    } else {
      sum += op.g1(static_cast<PRECISE_T>(dy[idx]),
                   static_cast<PRECISE_T>(x0[idx0]),
                   static_cast<PRECISE_T>(x1[idx1]),
                   static_cast<PRECISE_T>(y[idx]), inplace);
    }
  }

  sbuf[tid] = sum;
  __syncthreads();

  sum = kernel_backward_dim3_block_reduce_x<PRECISE_T, blockSize>(tid, sbuf);

  if (tid == 0) {
    dx[iy + iz * y_size] = static_cast<PRECISE_T>(dx[iy + iz * y_size]) + sum;
  }
}

// ----------------------------------------------------------------------------
// Backward kernels of y-axis reduction for three dimensions
// ----------------------------------------------------------------------------
// y-axis reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term,
          Size_t y_div_size>
__global__ void kernel_backward_dim3_reduce_y(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y,
    PRECISE_T *dst, const bool inplace, const Dim3KernelParams p,
    const Size_t x_size, const Size_t y_size, const Size_t z_size) {
  const Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  Size_t iy = (Size_t)blockIdx.y * y_div_size;
  const Size_t iz = (Size_t)blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < x_size && iz < z_size) {
    PRECISE_T sum = 0;

    for (const Size_t yend = min(iy + y_div_size, y_size); iy < yend; ++iy) {
      const Size_t idx =
          flatten_idx(ix, iy, iz, p.stride_y_2, p.stride_y_1, p.stride_y_0);
      const Size_t idx0 =
          flatten_idx(ix, iy, iz, p.stride_x0_2, p.stride_x0_1, p.stride_x0_0);
      const Size_t idx1 =
          flatten_idx(ix, iy, iz, p.stride_x1_2, p.stride_x1_1, p.stride_x1_0);

      if (term == Term::x0) {
        sum += op.g0(static_cast<PRECISE_T>(dy[idx]),
                     static_cast<PRECISE_T>(x0[idx0]),
                     static_cast<PRECISE_T>(x1[idx1]),
                     static_cast<PRECISE_T>(y[idx]), inplace);
      } else {
        sum += op.g1(static_cast<PRECISE_T>(dy[idx]),
                     static_cast<PRECISE_T>(x0[idx0]),
                     static_cast<PRECISE_T>(x1[idx1]),
                     static_cast<PRECISE_T>(y[idx]), inplace);
      }
    }
    atomic_add(&dst[ix + iz * x_size], sum);
  }
}

// ----------------------------------------------------------------------------
// Backward kernels of z-axis reduction for three dimensions
// ----------------------------------------------------------------------------
// z-axis reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term,
          Size_t z_div_size>
__global__ void kernel_backward_dim3_reduce_z(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y,
    PRECISE_T *dst, const bool inplace, const Dim3KernelParams p,
    const Size_t x_size, const Size_t y_size, const Size_t z_size) {
  const Size_t ix = (Size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const Size_t iy = (Size_t)blockIdx.y * blockDim.y + threadIdx.y;
  Size_t iz = (Size_t)blockIdx.z * z_div_size;

  if (ix < x_size && iy < y_size) {
    PRECISE_T sum = 0;

    for (const Size_t zend = min(iz + z_div_size, z_size); iz < zend; ++iz) {
      const Size_t idx =
          flatten_idx(ix, iy, iz, p.stride_y_2, p.stride_y_1, p.stride_y_0);
      const Size_t idx0 =
          flatten_idx(ix, iy, iz, p.stride_x0_2, p.stride_x0_1, p.stride_x0_0);
      const Size_t idx1 =
          flatten_idx(ix, iy, iz, p.stride_x1_2, p.stride_x1_1, p.stride_x1_0);

      if (term == Term::x0) {
        sum += op.g0(static_cast<PRECISE_T>(dy[idx]),
                     static_cast<PRECISE_T>(x0[idx0]),
                     static_cast<PRECISE_T>(x1[idx1]),
                     static_cast<PRECISE_T>(y[idx]), inplace);
      } else {
        sum += op.g1(static_cast<PRECISE_T>(dy[idx]),
                     static_cast<PRECISE_T>(x0[idx0]),
                     static_cast<PRECISE_T>(x1[idx1]),
                     static_cast<PRECISE_T>(y[idx]), inplace);
      }
    }
    atomic_add(&dst[ix + iy * x_size], sum);
  }
}

// ----------------------------------------------------------------------------
// The actual part of forward_iml
// ----------------------------------------------------------------------------
// The actual part of forward_impl
template <typename T, typename BinaryOp>
void forward_impl(const Context &ctx, BinaryOp op, const Variables &inputs,
                  const Variables &outputs, const bool inplace,
                  const Size_t ndim, Variable &v_strides_x0,
                  Variable &v_strides_x1, Variable &v_strides_y,
                  Variable &v_shape_y) {
  using PRECISE_T = typename CudaTypeForceFloat<T>::type;
  cuda_set_device(std::stoi(ctx.device_id));

  const auto *x0 = inputs[0]->get_data_pointer<T>(ctx);
  const auto *x1 = inputs[1]->get_data_pointer<T>(ctx);
  auto *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, !inplace);

  if (ndim == 3) {
    // Three-dimensional case
    // SyncedArray::get_data_pointer only request array class by filter_context.
    Context cpu_ctx = Context().set_array_class(
        SingletonManager::get<Cpu>()->array_classes()[0]);
    const auto *stride_x0 = v_strides_x0.get_data_pointer<Size_t>(cpu_ctx);
    const auto *stride_x1 = v_strides_x1.get_data_pointer<Size_t>(cpu_ctx);
    const auto *stride_y = v_strides_y.get_data_pointer<Size_t>(cpu_ctx);
    const auto *shape_y = v_shape_y.get_data_pointer<Size_t>(cpu_ctx);
    // The parameters are passed to kernels as constants.
    Dim3KernelParams params{stride_x0[0], stride_x0[1], stride_x0[2],
                            stride_x1[0], stride_x1[1], stride_x1[2],
                            stride_y[0],  stride_y[1],  stride_y[2],
                            shape_y[0],   shape_y[1],   shape_y[2]};

    const Size_t x_size = shape_y[2];
    const Size_t y_size = shape_y[1];
    const Size_t z_size = shape_y[0];
    const Shape_t shape = {z_size, y_size, x_size};

    if ((stride_x0[0] == 0) || (stride_x0[1] == 0) || (stride_x0[2] == 0) ||
        (stride_x1[0] == 0) || (stride_x1[1] == 0) || (stride_x1[2] == 0)) {
      // Broadcast
      dim3 blockDim = get_blocks_dim3<T, BinaryOp>(shape, 3);
      const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
      const Size_t ceiled_y = (y_size + blockDim.y - 1) / blockDim.y;
      const Size_t ceiled_z = (z_size + blockDim.z - 1) / blockDim.z;
      NBLA_CHECK(
          ceiled_x <= UINT_MAX || ceiled_y <= UINT_MAX || ceiled_z <= UINT_MAX,
          error_code::type, "The input size is too large to be converted to "
                            "the gridSize of a CUDA kernel.");
      dim3 gridDim(static_cast<unsigned int>(ceiled_x),
                   static_cast<unsigned int>(ceiled_y),
                   static_cast<unsigned int>(ceiled_z));

      kernel_forward_dim3<T, PRECISE_T, BinaryOp><<<gridDim, blockDim>>>(
          op, x0, x1, y, params);
      NBLA_CUDA_KERNEL_CHECK();
    } else {
      // Not broadcast
      dim3 blockDim = get_blocks_dim3<T, BinaryOp>(shape, 1);
      const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
      NBLA_CHECK(ceiled_x <= UINT_MAX, error_code::type,
                 "The input size is too large to be converted to the gridSize "
                 "of a CUDA kernel.");
      dim3 gridDim(static_cast<unsigned int>(ceiled_x));

      kernel_forward_dim3_not_broadcasted_both_terms<
          T, PRECISE_T, BinaryOp><<<gridDim, blockDim>>>(op, x0, x1, y, params);
      NBLA_CUDA_KERNEL_CHECK();
    }
  } else {
    // Otherwise
    // setup_impl gurantees more than four dimensional data.
    const auto size = outputs[0]->size();
    const auto *stride_x0 = v_strides_x0.get_data_pointer<Size_t>(ctx);
    const auto *stride_x1 = v_strides_x1.get_data_pointer<Size_t>(ctx);
    const auto *stride_y = v_strides_y.get_data_pointer<Size_t>(ctx);
    const auto *shape_y = v_shape_y.get_data_pointer<Size_t>(ctx);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_forward_ndim<T, PRECISE_T, BinaryOp>), size, op, x0, x1, y,
        ndim, stride_x0, stride_x1, stride_y, shape_y);
  }
}

// ----------------------------------------------------------------------------
// The actual part of backward_iml
// ----------------------------------------------------------------------------
// This function issues the kernel without reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
void backward_impl_dim3_without_reduction(
    BinaryOp op, const T *dy, const T *x0, const T *x1, const T *y, T *dx,
    const bool inplace, const Dim3KernelParams &params, const Size_t x_size,
    const Size_t y_size, const Size_t z_size,
    const bool broadcasted_the_other_term) {
  const auto shape = Shape_t{z_size, y_size, x_size};

  if (broadcasted_the_other_term) {
    // The other term is broadcasted.
    dim3 blockDim = get_blocks_dim3<T, BinaryOp>(shape, 3);
    const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
    const Size_t ceiled_y = (y_size + blockDim.y - 1) / blockDim.y;
    const Size_t ceiled_z = (z_size + blockDim.z - 1) / blockDim.z;
    NBLA_CHECK(ceiled_x <= UINT_MAX || ceiled_y <= UINT_MAX ||
                   ceiled_z <= UINT_MAX,
               error_code::type, "The input size is too large to be converted "
                                 "to the gridSize of a CUDA kernel.");
    dim3 gridDim(static_cast<unsigned int>(ceiled_x),
                 static_cast<unsigned int>(ceiled_y),
                 static_cast<unsigned int>(ceiled_z));
    kernel_backward_dim3_broadcasted_other_term<T, PRECISE_T, BinaryOp,
                                                term><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, dx, inplace, params);
    NBLA_CUDA_KERNEL_CHECK();
  } else {
    // The other term is not broadcasted too. The computation becomes easier.
    dim3 blockDim = get_blocks_dim3<T, BinaryOp>(shape, 1);
    const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
    NBLA_CHECK(ceiled_x <= UINT_MAX, error_code::type,
               "The input size is too large to be converted to the gridSize of "
               "a CUDA kernel.");
    dim3 gridDim(static_cast<unsigned int>(ceiled_x));
    kernel_backward_dim3_not_broadcasted_both_terms<
        T, PRECISE_T, BinaryOp, term><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, dx, inplace, params);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

// This function issues the kernel of x-axis reduction
// x-axis reduction requires more complicated algorithm than y and z axis
// beacuse of memory layout.
template <typename T, typename PRECISE_T, typename BinaryOp, Term term,
          unsigned int blockSize>
void backward_impl_dim3_reduce_x(BinaryOp op, const T *dy, const T *x0,
                                 const T *x1, const T *y,
                                 const PRECISE_T *z_reduced_buff, T *dx,
                                 const bool inplace,
                                 const Dim3KernelParams &params,
                                 const Size_t x_size, const Size_t y_size,
                                 const Size_t z_size) {
  dim3 blockDim(blockSize);
  const size_t smem_size = blockSize * sizeof(PRECISE_T);

  if (z_reduced_buff) {
    // z axis is already reduced.
    dim3 gridDim(1, y_size); // z_size == 1
    kernel_backward_dim3_reduce_x_after_z<
        T, PRECISE_T, blockSize><<<gridDim, blockDim, smem_size>>>(
        z_reduced_buff, dx, x_size);
    NBLA_CUDA_KERNEL_CHECK();
  } else {
    // x axis is only reduced.
    dim3 gridDim(1, y_size, z_size);
    kernel_backward_dim3_reduce_x<T, PRECISE_T, BinaryOp, term,
                                  blockSize><<<gridDim, blockDim, smem_size>>>(
        op, dy, x0, x1, y, dx, inplace, params, x_size, y_size, z_size);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

// This function issues the kernel of y-axis reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
void backward_impl_dim3_reduce_y(const Context &ctx, BinaryOp op, const T *dy,
                                 const T *x0, const T *x1, const T *y, T *dx,
                                 const bool inplace,
                                 const Dim3KernelParams &params,
                                 const Size_t x_size, const Size_t y_size,
                                 const Size_t z_size) {
  dim3 blockDim =
      get_blocks_dim3<T, BinaryOp>(Shape_t{z_size, y_size, x_size}, 1);
  const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
  const Size_t ceiled_y =
      (y_size + TRANSFORM_BINARY_DIV_SIZE - 1) / TRANSFORM_BINARY_DIV_SIZE;
  const Size_t ceiled_z = (z_size + blockDim.z - 1) / blockDim.z;
  NBLA_CHECK(ceiled_x <= UINT_MAX || ceiled_y <= UINT_MAX ||
                 ceiled_z <= UINT_MAX,
             error_code::type, "The input size is too large to be converted to "
                               "the gridSize of a CUDA kernel.");
  dim3 gridDim(static_cast<unsigned int>(ceiled_x),
               static_cast<unsigned int>(ceiled_y),
               static_cast<unsigned int>(ceiled_z));

  if (is_same<T, PRECISE_T>::value) {
    kernel_backward_dim3_reduce_y<
        T, T, BinaryOp, term, TRANSFORM_BINARY_DIV_SIZE><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, dx, inplace, params, x_size, y_size, z_size);
    NBLA_CUDA_KERNEL_CHECK();
  } else {
    const auto dx_shape = Shape_t{z_size, 1, x_size};
    const auto dx_size = z_size * x_size;
    NdArray tmp_arr(dx_shape);
    tmp_arr.zero();
    auto tmp = tmp_arr.cast(get_dtype<PRECISE_T>(), ctx)->pointer<PRECISE_T>();

    kernel_backward_dim3_reduce_y<
        T, PRECISE_T, BinaryOp, term,
        TRANSFORM_BINARY_DIV_SIZE><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, tmp, inplace, params, x_size, y_size, z_size);
    NBLA_CUDA_KERNEL_CHECK();

    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_precise_add<T, PRECISE_T, true>),
                                   dx_size, dx, tmp);
  }
}

// This function issues the kernel of y-axis reduction
template <typename T, typename PRECISE_T, typename BinaryOp, Term term>
PRECISE_T *backward_impl_dim3_reduce_z(
    const Context &ctx, BinaryOp op, const T *dy, const T *x0, const T *x1,
    const T *y, NdArray &tmp_arr, T *dx, const bool inplace,
    const Dim3KernelParams &params, const Size_t x_size, const Size_t y_size,
    const Size_t z_size, const bool reduce_x) {
  dim3 blockDim =
      get_blocks_dim3<T, BinaryOp>(Shape_t{z_size, y_size, x_size}, 2);
  const Size_t ceiled_x = (x_size + blockDim.x - 1) / blockDim.x;
  const Size_t ceiled_y = (y_size + blockDim.y - 1) / blockDim.y;
  const Size_t ceiled_z =
      (z_size + TRANSFORM_BINARY_DIV_SIZE - 1) / TRANSFORM_BINARY_DIV_SIZE;
  NBLA_CHECK(ceiled_x <= UINT_MAX || ceiled_y <= UINT_MAX ||
                 ceiled_z <= UINT_MAX,
             error_code::type, "The input size is too large to be converted to "
                               "the gridSize of a CUDA kernel.");
  dim3 gridDim(static_cast<unsigned int>(ceiled_x),
               static_cast<unsigned int>(ceiled_y),
               static_cast<unsigned int>(ceiled_z));

  if (is_same<T, PRECISE_T>::value && !reduce_x) {
    kernel_backward_dim3_reduce_z<
        T, T, BinaryOp, term, TRANSFORM_BINARY_DIV_SIZE><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, dx, inplace, params, x_size, y_size, z_size);
    NBLA_CUDA_KERNEL_CHECK();
    return nullptr;
  } else {
    const auto dx_shape = Shape_t{1, y_size, x_size};
    const auto dx_size = y_size * x_size;
    tmp_arr.reshape(dx_shape, true);
    tmp_arr.zero();
    auto tmp = tmp_arr.cast(get_dtype<PRECISE_T>(), ctx)->pointer<PRECISE_T>();

    kernel_backward_dim3_reduce_z<
        T, PRECISE_T, BinaryOp, term,
        TRANSFORM_BINARY_DIV_SIZE><<<gridDim, blockDim>>>(
        op, dy, x0, x1, y, tmp, inplace, params, x_size, y_size, z_size);
    NBLA_CUDA_KERNEL_CHECK();

    if (reduce_x) {
      return tmp; // reuturn the intermediate buffer.
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_precise_add<T, PRECISE_T, true>),
                                     dx_size, dx, tmp);
      return nullptr;
    }
  }
}

// Three dimensional case of backward_impl
template <class T, typename PRECISE_T, typename BinaryOp, Term term>
void backward_impl_dim3(const Context &ctx, BinaryOp op, const T *dy,
                        const T *x0, const T *x1, const T *y, T *dx,
                        const bool inplace,
                        const Size_t *stride_x0, /* host pointer */
                        const Size_t *stride_x1, /* host pointer */
                        const Size_t *stride_y,  /* host pointer */
                        const Size_t *shape_y /* host pointer */) {
  const auto x_size = shape_y[2];
  const auto y_size = shape_y[1];
  const auto z_size = shape_y[0];
  const auto shape = Shape_t{z_size, y_size, x_size};
  const auto stride = (term == Term::x0) ? stride_x0 : stride_x1;
  const auto the_other_stride = (term != Term::x0) ? stride_x0 : stride_x1;
  const bool reduce_x = (stride[2] == 0);
  const bool reduce_y = (stride[1] == 0);
  const bool reduce_z = (stride[0] == 0);
  const bool broadcasted_the_other_term =
      ((the_other_stride[0] == 0) || (the_other_stride[1] == 0) ||
       (the_other_stride[2] == 0));

  // The parameters are passed to kernels as constants.
  const Dim3KernelParams params{stride_x0[0], stride_x0[1], stride_x0[2],
                                stride_x1[0], stride_x1[1], stride_x1[2],
                                stride_y[0],  stride_y[1],  stride_y[2],
                                shape_y[0],   shape_y[1],   shape_y[2]};

  if (!reduce_x && !reduce_y && !reduce_z) {
    // This term is not broadcasted. Reduction is not required.
    backward_impl_dim3_without_reduction<T, PRECISE_T, BinaryOp, term>(
        op, dy, x0, x1, y, dx, inplace, params, x_size, y_size, z_size,
        broadcasted_the_other_term);
  } else if (reduce_y) {
    // This term is broadcasted along y-axis. Reduce y-axis
    backward_impl_dim3_reduce_y<T, PRECISE_T, BinaryOp, term>(
        ctx, op, dy, x0, x1, y, dx, inplace, params, x_size, y_size, z_size);
  } else {
    // This term is broadcasted along x, z, or (x and z) axis.
    // Reduce them respectively. An intermediate buffer is used when
    // (x and z)-axis reduction as PRECISE_T to preserve precision.
    NdArray tmp_arr;
    PRECISE_T *tmp = nullptr;

    if (reduce_z) {
      tmp = backward_impl_dim3_reduce_z<T, PRECISE_T, BinaryOp, term>(
          ctx, op, dy, x0, x1, y, tmp_arr, dx, inplace, params, x_size, y_size,
          z_size, reduce_x);
    }

    if (reduce_x) {
      dim3 blockDim = get_blocks_dim3<T, BinaryOp>(shape, 1);
      if (blockDim.x == 512) {
        backward_impl_dim3_reduce_x<T, PRECISE_T, BinaryOp, term, 512>(
            op, dy, x0, x1, y, tmp, dx, inplace, params, x_size, y_size,
            z_size);
      } else if (blockDim.x == 256) {
        backward_impl_dim3_reduce_x<T, PRECISE_T, BinaryOp, term, 256>(
            op, dy, x0, x1, y, tmp, dx, inplace, params, x_size, y_size,
            z_size);
      } else if (blockDim.x == 128) {
        backward_impl_dim3_reduce_x<T, PRECISE_T, BinaryOp, term, 128>(
            op, dy, x0, x1, y, tmp, dx, inplace, params, x_size, y_size,
            z_size);
      } else if (blockDim.x == 64) {
        backward_impl_dim3_reduce_x<T, PRECISE_T, BinaryOp, term, 64>(
            op, dy, x0, x1, y, tmp, dx, inplace, params, x_size, y_size,
            z_size);
      } else { // blockDim.x <= 32
        blockDim.x = 32;
        backward_impl_dim3_reduce_x<T, PRECISE_T, BinaryOp, term, 32>(
            op, dy, x0, x1, y, tmp, dx, inplace, params, x_size, y_size,
            z_size);
      }
    }
  }
}

// Any dimensional case of backward_impl
template <class T, typename PRECISE_T, typename BinaryOp, Term term>
void backward_impl_ndim(const Size_t size, const Context &ctx, BinaryOp op,
                        const T *dy, const T *x0, const T *x1, const T *y,
                        Variable *v_x, const bool accum, const bool inplace,
                        const Size_t ndim,
                        const Size_t *stride_x0, /* device pointer */
                        const Size_t *stride_x1, /* device pointer */
                        const Size_t *stride_y,  /* device pointer */
                        const Size_t *shape_y /* device pointer */) {
  if (is_same<T, PRECISE_T>::value) {
    if (!accum)
      v_x->grad()->zero();
    auto *dx = v_x->cast_grad_and_get_pointer<T>(ctx);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_backward_ndim<T, T, BinaryOp, term>),
                                   size, op, dy, x0, x1, y, dx, inplace, ndim,
                                   stride_x0, stride_x1, stride_y, shape_y);
  } else {
    // Intermediate buffer to preserve precision.
    NdArray tmp_arr(v_x->shape());
    tmp_arr.zero();
    auto *tmp = tmp_arr.cast(get_dtype<PRECISE_T>(), ctx)->pointer<PRECISE_T>();

    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_backward_ndim<T, PRECISE_T, BinaryOp, term>), size, op, dy, x0,
        x1, y, tmp, inplace, ndim, stride_x0, stride_x1, stride_y, shape_y);

    auto *dx = v_x->cast_grad_and_get_pointer<T>(ctx, !accum);
    if (accum) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_precise_add<T, PRECISE_T, true>),
                                     v_x->size(), dx, tmp);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_precise_add<T, PRECISE_T, false>),
                                     v_x->size(), dx, tmp);
    }
  }
}

// The actual part of backward_impl
template <class T, typename BinaryOp>
void backward_impl(const Context &ctx, BinaryOp op, const Variables &inputs,
                   const Variables &outputs, const vector<bool> &propagate_down,
                   const vector<bool> &accum, const bool inplace,
                   const Size_t ndim, Variable &v_strides_x0,
                   Variable &v_strides_x1, Variable &v_strides_y,
                   Variable &v_shape_y) {
  if (!(propagate_down[0] || propagate_down[1]))
    return;
  cuda_set_device(std::stoi(ctx.device_id));
  using PRECISE_T = typename CudaTypeForceFloat<T>::type;

  const auto *dy = outputs[0]->get_grad_pointer<T>(ctx);
  const auto *x0 = inputs[0]->get_data_pointer<T>(ctx);
  const auto *x1 = inputs[1]->get_data_pointer<T>(ctx);
  const auto *y = outputs[0]->get_data_pointer<T>(ctx);

  if (ndim == 3) {
    // Three-dimensional case
    // SyncedArray::get_data_pointer only request array class by filter_context.
    Context cpu_ctx = Context().set_array_class(
        SingletonManager::get<Cpu>()->array_classes()[0]);
    const auto *stride_x0 = v_strides_x0.get_data_pointer<Size_t>(cpu_ctx);
    const auto *stride_x1 = v_strides_x1.get_data_pointer<Size_t>(cpu_ctx);
    const auto *stride_y = v_strides_y.get_data_pointer<Size_t>(cpu_ctx);
    const auto *shape_y = v_shape_y.get_data_pointer<Size_t>(cpu_ctx);

    if (propagate_down[0]) { // dx0
      op.verify_g0();
      if (!accum[0])
        inputs[0]->grad()->zero();
      auto *dx0 = inputs[0]->cast_grad_and_get_pointer<T>(ctx);
      backward_impl_dim3<T, PRECISE_T, BinaryOp, Term::x0>(
          ctx, op, dy, x0, x1, y, dx0, inplace, stride_x0, stride_x1, stride_y,
          shape_y);
    }

    if (propagate_down[1]) { // dx1
      op.verify_g1();
      if (!accum[1])
        inputs[1]->grad()->zero();
      auto *dx1 = inputs[1]->cast_grad_and_get_pointer<T>(ctx);
      backward_impl_dim3<T, PRECISE_T, BinaryOp, Term::x1>(
          ctx, op, dy, x0, x1, y, dx1, inplace, stride_x0, stride_x1, stride_y,
          shape_y);
    }
  } else {
    // Otherwise
    // setup_impl gurantees more than four dimensional data.
    const auto *stride_x0 = v_strides_x0.get_data_pointer<Size_t>(ctx);
    const auto *stride_x1 = v_strides_x1.get_data_pointer<Size_t>(ctx);
    const auto *stride_y = v_strides_y.get_data_pointer<Size_t>(ctx);
    const auto *shape_y = v_shape_y.get_data_pointer<Size_t>(ctx);
    const auto size_y = outputs[0]->size();

    if (propagate_down[0]) { // dx0
      op.verify_g0();
      backward_impl_ndim<T, PRECISE_T, BinaryOp, Term::x0>(
          size_y, ctx, op, dy, x0, x1, y, inputs[0], accum[0], inplace, ndim,
          stride_x0, stride_x1, stride_y, shape_y);
    }

    if (propagate_down[1]) { // dx1
      op.verify_g1();
      backward_impl_ndim<T, PRECISE_T, BinaryOp, Term::x1>(
          size_y, ctx, op, dy, x0, x1, y, inputs[1], accum[1], inplace, ndim,
          stride_x0, stride_x1, stride_y, shape_y);
    }
  }
}
} // end of namespace "nbla::transform_binary_cuda"

// ----------------------------------------------------------------------------
// Common
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME)                                 \
  class NAME##BinaryOpCuda : public BaseBinaryOpCuda

#define NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                 \
  template <typename T>                                                        \
  __forceinline__ __device__ T operator()(const T x0, const T x1) {            \
    return OP;                                                                 \
  }

#define NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(NUM, GOP)                          \
  template <typename T>                                                        \
  __forceinline__ __device__ T g##NUM(const T dy, const T x0, const T x1,      \
                                      const T y, const bool inplace) {         \
    return GOP;                                                                \
  }                                                                            \
  __host__ void verify_g##NUM() {}

#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)               \
  template <typename T>                                                        \
  void NAME##Cuda<T>::forward_impl(const Variables &inputs,                    \
                                   const Variables &outputs) {                 \
    transform_binary_cuda::forward_impl<typename CudaType<T>::type>(           \
        this->ctx_, NAME##BinaryOpCuda(this->args_), inputs, outputs,          \
        this->inplace_, this->compressed_ndim_, this->strides_x0_,             \
        this->strides_x1_, this->strides_y_, this->shape_y_);                  \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  void NAME##Cuda<T>::backward_impl(                                           \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down, const vector<bool> &accum) {         \
    transform_binary_cuda::backward_impl<typename CudaType<T>::type>(          \
        this->ctx_, NAME##BinaryOpCuda(this->args_), inputs, outputs,          \
        propagate_down, accum, this->inplace_, this->compressed_ndim_,         \
        this->strides_x0_, this->strides_x1_, this->strides_y_,                \
        this->shape_y_);                                                       \
  }

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_CUDA_NO_GRAD(NAME, OP)                           \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    __inline__ __host__ __device__ NAME##BinaryOpCuda(const tuple<> &dummy) {} \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
  }

#define NBLA_DEFINE_BINARY_OP_CUDA(NAME, OP, GOP0, GOP1)                       \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    __inline__ __host__ __device__ NAME##BinaryOpCuda(const tuple<> &dummy) {} \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(0, GOP0)                               \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(1, GOP1)                               \
  }
#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_NO_GRAD(NAME, OP)                    \
  NBLA_DEFINE_BINARY_OP_CUDA_NO_GRAD(NAME, OP);                                \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)

#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA(NAME, OP, GOP0, GOP1)                \
  NBLA_DEFINE_BINARY_OP_CUDA(NAME, OP, GOP0, GOP1);                            \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_CUDA_1(NAME, OP, GOP0, GOP1, A0)                 \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    A0 a0;                                                                     \
    __inline__ NAME##BinaryOpCuda(const tuple<A0> &args)                       \
        : a0(std::get<0>(args)) {}                                             \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(0, GOP0)                               \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(1, GOP1)                               \
  }
#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_1(NAME, OP, GOP0, GOP1, A0)          \
  NBLA_DEFINE_BINARY_OP_CUDA_1(NAME, OP, GOP0, GOP1, A0);                      \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)
}
#endif

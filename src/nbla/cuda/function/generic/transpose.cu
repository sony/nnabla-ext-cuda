// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

#include <limits>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/transpose.hpp>
#include <nbla/variable.hpp>

namespace nbla {

namespace { // local namespace

template <typename T, bool accum = false>
__global__ void transpose_1d(const int size, const T *src, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    dst[idx] = accum ? dst[idx] + src[idx] : src[idx];
  }
}

template <typename T, bool accum = false>
__global__ void transpose_2d(const int2 shape, const T *src, T *dst) {
  // One extra column to avoid memory bank conflicts.
  __shared__ T tile[CUDA_WARP_SIZE][CUDA_WARP_SIZE + 1];

  int x = blockIdx.x * CUDA_WARP_SIZE + threadIdx.x;
  int y = blockIdx.y * CUDA_WARP_SIZE + threadIdx.y;

#pragma unroll
  for (int j = 0; j < CUDA_WARP_SIZE; j += 8) {
    if ((x < shape.x) && (y + j < shape.y)) {
      tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * shape.x + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * CUDA_WARP_SIZE + threadIdx.x; // transpose block offset
  y = blockIdx.x * CUDA_WARP_SIZE + threadIdx.y;

#pragma unroll
  for (int j = 0; j < CUDA_WARP_SIZE; j += 8) {
    if ((x < shape.y) && (y + j < shape.x)) {
      auto val = tile[threadIdx.x][threadIdx.y + j];
      auto idx = (y + j) * shape.y + x;
      dst[idx] = accum ? dst[idx] + val : val;
    }
  }
}

template <typename T, bool accum = false>
__global__ void transpose_3d(const int size, const int3 ostride,
                             const int3 tstride, const T *src, T *dst) {
  // ostride - strides of the transposed input shape
  // tstride - transpose of the input shape strides
  // Ex. transpose(ishape=(2, 3, 4), axes=(2, 1, 0)) => oshape (4, 3, 2)
  //     then ostride is (6, 2, 1) and tstride is (1, 4, 12)
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto z = (idx / ostride.z);
    auto y = (idx - z * ostride.z) / ostride.y;
    auto x = (idx - z * ostride.z - y * ostride.y);
    T val = src[z * tstride.z + y * tstride.y + x * tstride.x];
    dst[idx] = accum ? dst[idx] + val : val;
  }
}

template <typename T, bool accum = false>
__global__ void transpose_4d(const int size, const int4 ostride,
                             const int4 tstride, const T *src, T *dst) {
  // ostride - strides of the transposed input shape
  // tstride - transpose of the input shape strides
  // Ex. transpose(ishape=(2,3,4,5), axes=(3,2,1,0)) => oshape (5,4,3,2)
  //     then ostride is (24, 6, 2, 1) and tstride is (1, 5, 20, 60)
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    auto w = (idx / ostride.w);
    auto z = (idx - w * ostride.w) / ostride.z;
    auto y = (idx - w * ostride.w - z * ostride.z) / ostride.y;
    auto x = (idx - w * ostride.w - z * ostride.z - y * ostride.y);
    T val = src[w * tstride.w + z * tstride.z + y * tstride.y + x * tstride.x];
    dst[idx] = accum ? dst[idx] + val : val;
  }
}

template <typename T> struct TransposeStrides {
  T ostride; // strides of transposed input shape
  T tstride; // transposed strides of input shape
};

template <typename T, bool accum = false>
__global__ void transpose_nd(const int size, const T *src, T *dst,
                             const TransposeStrides<int> *strides,
                             const int ndim) {
  NBLA_CUDA_KERNEL_LOOP(dst_index, size) {
    int src_index = 0, tmp_index = dst_index;
    for (int axis = 0; axis < ndim; axis++) {
      const auto k = tmp_index / strides[axis].ostride;
      src_index += k * strides[axis].tstride;
      tmp_index -= k * strides[axis].ostride;
    }
    dst[dst_index] = accum ? dst[dst_index] + src[src_index] : src[src_index];
  }
}

} // end local namespace

template <typename T>
void TransposeCuda<T>::setup_impl(const Variables &inputs,
                                  const Variables &outputs) {
  Transpose<T>::setup_impl(inputs, outputs);

  const int ndim = this->x_shape_.size();
  const int size = outputs[0]->size();

  NBLA_CHECK(size <= std::numeric_limits<int>::max(), error_code::value,
             "Maximum supported array size is %d elements",
             std::numeric_limits<int>::max());

  if (ndim > 4) {
    Shape_t shape = {2, ndim * (int)sizeof(TransposeStrides<int>)};
    this->var_strides_ = std::make_shared<Variable>();
    this->var_strides_->reshape(shape, true);
    auto var = static_cast<VariablePtr>(this->var_strides_);
    auto ptr = var->cast_data_and_get_pointer<char>(Context());
    auto strides = reinterpret_cast<TransposeStrides<int> *>(ptr);
    for (int i = 0; i < ndim; i++) {
      // strides for forward where we iterate the output shape
      strides[i].ostride = this->y_strides_[i];
      strides[i].tstride = this->x_strides_transposed_[i];
      // strides for backward where we iterate the input shape
      strides[i + ndim].ostride = this->x_strides_[i];
      strides[i + ndim].tstride = this->y_strides_transposed_[i];
    }
  }
}

template <typename T> inline int2 make_int2_from(vector<T> vec, int skip = 0) {
  return make_int2(vec[1 + skip], vec[0 + skip]);
}

template <typename T> inline int3 make_int3_from(vector<T> vec, int skip = 0) {
  return make_int3(vec[2 + skip], vec[1 + skip], vec[0 + skip]);
}

template <typename T> inline int4 make_int4_from(vector<T> vec, int skip = 0) {
  return make_int4(vec[3 + skip], vec[2 + skip], vec[1 + skip], vec[0 + skip]);
}

#define WARPS_FOR(N) NBLA_CEIL_INT_DIV(N, CUDA_WARP_SIZE)

template <class T>
void TransposeCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  cuda_set_device(this->device_);
  auto x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  const int ndim = this->x_shape_.size();
  const int size = outputs[0]->size();

  if (ndim == 1) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(transpose_1d, size, x, y);
    return;
  }
  if (ndim == 2) {
    const auto shape = make_int2_from(this->x_shape_);
    const dim3 grid_dim(WARPS_FOR(shape.x), WARPS_FOR(shape.y));
    if (grid_dim.y < 65536) {
      const dim3 block_dim(CUDA_WARP_SIZE, 8);
      transpose_2d<<<grid_dim, block_dim>>>(shape, x, y);
      NBLA_CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (ndim == 3 && this->axes_[0] == 0) {
    const auto shape = make_int2_from(this->x_shape_, 1);
    const dim3 grid_dim(WARPS_FOR(shape.x), WARPS_FOR(shape.y));
    if (grid_dim.y < 65536) {
      const dim3 block_dim(CUDA_WARP_SIZE, 8);
      for (int i = 0, w = shape.x * shape.y; i < this->x_shape_[0]; i++)
        transpose_2d<<<grid_dim, block_dim>>>(shape, x + i * w, y + i * w);
      NBLA_CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (ndim == 3) {
    const auto ostride = make_int3_from(this->y_strides_);
    const auto tstride = make_int3_from(this->x_strides_transposed_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(transpose_3d, size, ostride, tstride, x, y);
    return;
  }
  if (ndim == 4) {
    const auto ostride = make_int4_from(this->y_strides_);
    const auto tstride = make_int4_from(this->x_strides_transposed_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(transpose_4d, size, ostride, tstride, x, y);
    return;
  }
  auto var = static_cast<VariablePtr>(this->var_strides_);
  auto ptr = var->get_data_pointer<char>(this->ctx_);
  auto strides = reinterpret_cast<const TransposeStrides<int> *>(ptr);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(transpose_nd, size, x, y, strides, ndim);
}

template <class T>
void TransposeCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0])
    return;

  cuda_set_device(this->device_);
  auto dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  const int ndim = this->x_shape_.size();
  const int size = outputs[0]->size();

  if (ndim == 1) {
    auto kernel = accum[0] ? transpose_1d<Tcu, true> : transpose_1d<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, dy, dx);
    return;
  }
  if (ndim == 2) {
    const auto shape = make_int2_from(this->y_shape_);
    const dim3 grid_dim(WARPS_FOR(shape.x), WARPS_FOR(shape.y));
    if (grid_dim.y < 65536) {
      const dim3 block_dim(CUDA_WARP_SIZE, 8);
      auto kernel = accum[0] ? transpose_2d<Tcu, true> : transpose_2d<Tcu>;
      kernel<<<grid_dim, block_dim>>>(shape, dy, dx);
      NBLA_CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (ndim == 3 && this->axes_[0] == 0) {
    const auto shape = make_int2_from(this->y_shape_, 1);
    const dim3 grid_dim(WARPS_FOR(shape.x), WARPS_FOR(shape.y));
    if (grid_dim.y < 65536) {
      const dim3 block_dim(CUDA_WARP_SIZE, 8);
      auto kernel = accum[0] ? transpose_2d<Tcu, true> : transpose_2d<Tcu>;
      for (int i = 0, w = shape.x * shape.y; i < this->x_shape_[0]; i++)
        kernel<<<grid_dim, block_dim>>>(shape, dy + i * w, dx + i * w);
      NBLA_CUDA_KERNEL_CHECK();
      return;
    }
  }
  if (ndim == 3) {
    const auto ostride = make_int3_from(this->x_strides_);
    const auto tstride = make_int3_from(this->y_strides_transposed_);
    auto kernel = accum[0] ? transpose_3d<Tcu, true> : transpose_3d<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ostride, tstride, dy, dx);
    return;
  }
  if (ndim == 4) {
    const auto ostride = make_int4_from(this->x_strides_);
    const auto tstride = make_int4_from(this->y_strides_transposed_);
    auto kernel = accum[0] ? transpose_4d<Tcu, true> : transpose_4d<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ostride, tstride, dy, dx);
    return;
  }
  auto var = static_cast<VariablePtr>(this->var_strides_);
  auto ptr = var->get_data_pointer<char>(this->ctx_);
  auto strides = reinterpret_cast<const TransposeStrides<int> *>(ptr);
  auto kernel = accum[0] ? transpose_nd<Tcu, true> : transpose_nd<Tcu>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, dy, dx, strides + ndim, ndim);
}
}

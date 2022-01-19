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

#ifndef __NBLA_CUDA_UTILS_FAST_REDUCE_CUH__
#define __NBLA_CUDA_UTILS_FAST_REDUCE_CUH__

#include <assert.h>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/index_converter.cuh>
#include <nbla/cuda/utils/reduce.hpp>
#include <nbla/cuda/utils/warp_shuffle.cuh>
#include <numeric>

// TODO: Rename fast_reduct to device_reduce after removing other duplicated
// reduction source codes.

namespace nbla {

#define NBLA_CUDA_REDUCE_MAX_BLOCKS 65535
#define NBLA_CUDA_REDUCE_UNROLL_XY 4
#define NBLA_CUDA_REDUCE_UNROLL_Y 4

/** Determine which block calls this kernel last.
 */
template <class IndexT>
__device__ bool is_last_block(const int block_local_idx, // idx in a block
                              const IndexT block_idx,    // idx of blocks
                              const int num_blocks, int *block_counter) {
  __threadfence();
  int last = 0;
  if (block_local_idx == 0) {
    last = atomicAdd(block_counter + block_idx, 1);
  }
  return __syncthreads_or(last == num_blocks - 1);
}

/** Sequential reduction by each thread
 */
template <int unroll_size, class Op>
__device__ typename Op::StorageT kernel_thread_reduce(
    Op op, typename Op::IndexT inner_idx, const typename Op::IndexT outer_idx,
    const typename Op::IndexT inner_size, const int inner_grid_stride,
    const IndexConverter<typename Op::IndexT> inner_idx_conv,
    const IndexConverter<typename Op::IndexT> outer_idx_conv) {
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  StorageT reduced[unroll_size];
#pragma unroll
  for (int i = 0; i < unroll_size; i++) {
    reduced[i] = op.init();
  }

  IndexT global_outer_idx = outer_idx_conv.change_strides(0, outer_idx);
  StorageT loader[unroll_size];

  for (; inner_idx + inner_grid_stride * (unroll_size - 1) < inner_size;
       inner_idx += inner_grid_stride * unroll_size) {
#pragma unroll
    for (int i = 0; i < unroll_size; i++) {
      // Load first. The loads become independent and the latencies are hidden.
      IndexT global_idx = inner_idx_conv.change_strides(
          global_outer_idx, inner_idx + inner_grid_stride * i);
      loader[i] = op.make_storage(op.input[global_idx],
                                  inner_idx + inner_grid_stride * i);
    }
#pragma unroll
    for (int i = 0; i < unroll_size; i++) {
      // Reduce
      reduced[i] = op(reduced[i], loader[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < unroll_size; i++) {
    // Load of the tail misaligned elements
    // This loop unroll keeps "reduced" in register cache. If no loop unroll,
    // it is located in local memory, causing bad performance.
    const IndexT idx = inner_idx + inner_grid_stride * i;
    if (idx >= inner_size) {
      break;
    }
    IndexT global_idx = inner_idx_conv.change_strides(global_outer_idx, idx);
    reduced[i] = op(reduced[i], op.make_storage(op.input[global_idx], idx));
  }

#pragma unroll
  for (int i = 1; i < unroll_size; i++) {
    reduced[0] = op(reduced[0], reduced[i]);
  }

  return reduced[0];
}

/** Determine which block calls this kernel last.
 */
template <int vec_size, class VecT, class Op>
__device__ typename Op::StorageT kernel_vectrized_thread_reduce_x(
    Op op, typename Op::IndexT inner_idx, const typename Op::IndexT outer_idx,
    const typename Op::IndexT inner_size, const int inner_grid_stride) {
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  StorageT reduced[vec_size];
#pragma unroll
  for (int i = 0; i < vec_size; i++) {
    reduced[i] = op.init();
  }

  // Load of the head misaligned elements
  auto input_slice = op.input + inner_size * outer_idx;
  constexpr int align_byte = sizeof(VecT);
  constexpr int elem_byte = sizeof(typename Op::Tcu);
  constexpr int aligned_elems = align_byte / elem_byte;
  auto num_heads = (int64_t)input_slice % align_byte / elem_byte;
  if (num_heads > 0) {
    num_heads = aligned_elems - num_heads;
    // Assuming blockDim.x * gridDim.x >= vec_size;
    if (inner_idx < num_heads) {
      reduced[inner_idx] =
          op(reduced[inner_idx],
             op.make_storage(input_slice[inner_idx], inner_idx));
    }
    // Make aligned address
    input_slice += num_heads;
  }

  // Vectorized load of the aligned elements
  const VecT *const load_input = reinterpret_cast<const VecT *>(input_slice);
  typename Op::Tcu load_reg[vec_size];
  VecT *load_vec = reinterpret_cast<VecT *>(&load_reg[0]);

  for (; num_heads + vec_size - 1 + vec_size * inner_idx < inner_size;
       inner_idx += inner_grid_stride) {
    *load_vec = load_input[inner_idx]; // vectrized load

#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      reduced[i] =
          op(reduced[i], op.make_storage(load_reg[i],
                                         num_heads + i + vec_size * inner_idx));
    }
  }

  // Load of the tail misaligned elements
  // Assuming blockDim.x >= vec_size.
  const IndexT idx =
      vec_size * IndexT((inner_size - num_heads) / vec_size) + threadIdx.x;
  if (blockIdx.x == 0 && num_heads + idx < inner_size) {
    reduced[0] =
        op(reduced[0], op.make_storage(input_slice[idx], num_heads + idx));
  }

  for (int i = 1; i < vec_size; i++) {
    reduced[0] = op(reduced[0], reduced[i]);
  }

  return reduced[0];
}

template <class SHFL_T, class T>
__device__ T kernel_shuffle_down(T val, const int offset) {
  SHFL_T shfl_v = *reinterpret_cast<SHFL_T *>(&val);
  SHFL_T v = warp::shuffle_down(shfl_v, offset);
  return *reinterpret_cast<T *>(&v);
}

template <class Op>
__device__ typename Op::StorageT kernel_warp_reduce(Op op,
                                                    typename Op::StorageT val) {
  using StorageT = typename Op::StorageT;
#pragma unroll
  for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
    if (sizeof(StorageT) == 16) {
      val = op(val, kernel_shuffle_down<float4>(val, offset));
    } else if (sizeof(StorageT) == 8) {
      val = op(val, kernel_shuffle_down<float2>(val, offset));
    } else if (sizeof(StorageT) == 4) {
      val = op(val, kernel_shuffle_down<float>(val, offset));
    } else if (sizeof(StorageT) == 2) {
      val = op(val, kernel_shuffle_down<half>(val, offset));
    } else {
      assert(false);
    }
  }
  return val;
}

template <class Op>
__device__ typename Op::StorageT
kernel_block_reduce_xy(Op op, typename Op::StorageT reduced,
                       typename Op::StorageT *sbuf) {
  // Reduction along a binary tree.
  if (blockDim.x > CUDA_WARP_SIZE) {
    const int block_local_idx = threadIdx.x + blockDim.x * threadIdx.y;
    sbuf[block_local_idx] = reduced;

    __syncthreads();

#pragma unroll
    for (int offset = blockDim.x / 2; offset >= CUDA_WARP_SIZE; offset /= 2) {
      if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
        reduced = op(reduced, sbuf[block_local_idx + offset]);
        sbuf[block_local_idx] = reduced;
      }
      __syncthreads();
    }
  }

  reduced = kernel_warp_reduce(op, reduced);

  return reduced;
}

template <class Op>
__device__ typename Op::StorageT
kernel_block_reduce_y(Op op, typename Op::StorageT reduced,
                      typename Op::StorageT *sbuf) {
  // Reduction along a binary tree.
  if (blockDim.y > 1) {
    const int block_local_idx = threadIdx.x + blockDim.x * threadIdx.y;
    sbuf[block_local_idx] = reduced;

#pragma unroll
    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
      __syncthreads();
      if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
        reduced = op(reduced,
                     sbuf[threadIdx.x + blockDim.x * (threadIdx.y + offset)]);
        sbuf[block_local_idx] = reduced;
      }
    }
  }

  // Warp reduce is not performed because warps are assigned the memory
  // continuous part. (x part)
  return reduced;
}

template <class Op>
__device__ typename Op::StorageT
kernel_inter_block_reduce_xy(Op op, const typename Op::IndexT outer_idx,
                             typename Op::StorageT *sbuf) {
  // Assuming blockDim.y == 1;

  // x reduction
  const auto num_blocks = gridDim.x; // the elements in buf per outer_idx
  const auto inner_buf = op.buf + num_blocks * outer_idx;

  // Load from global memory
  // Thread reduction on registers while loading.
  typename Op::StorageT reduced = op.init();

  for (auto idx = threadIdx.x; idx < num_blocks; idx += blockDim.x) {
    reduced = op(reduced, inner_buf[idx]);
  }

  // Block reduce
  return kernel_block_reduce_xy(op, reduced, sbuf);
}

template <class Op>
__device__ typename Op::StorageT
kernel_inter_block_reduce_y(Op op, const typename Op::IndexT outer_idx,
                            typename Op::StorageT *sbuf) {
  // x reduction
  const auto num_blocks = gridDim.y; // the elements in buf per outer_idx
  const auto inner_buf = op.buf + outer_idx;

  // Load from global memory
  // Thread reduction on registers while loading.
  typename Op::StorageT reduced = op.init();

  for (auto idx = threadIdx.y; idx < num_blocks; idx += blockDim.y) {
    reduced = op(reduced, inner_buf[blockDim.x * gridDim.x * idx]);
  }

  // Block reduce
  return kernel_block_reduce_y(op, reduced, sbuf);
}

template <class VecT, class Op>
__global__ void
kernel_reduce_xy(Op op, int *const block_counter,
                 const typename Op::IndexT inner_size,
                 const typename Op::IndexT outer_size,
                 const IndexConverter<typename Op::IndexT> inner_idx_conv,
                 const IndexConverter<typename Op::IndexT> outer_idx_conv) {
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  // "extern __shared__" for template type cannot be used in CUDA.
  // This is a workaround.
  extern __shared__ uint8_t sbuf_char[];
  StorageT *sbuf = reinterpret_cast<StorageT *>(sbuf_char);

  // Grid-strided loop for outer dimensions
  for (IndexT outer_idx = threadIdx.y + (IndexT)blockIdx.y * blockDim.y;
       outer_idx < outer_size; outer_idx += blockDim.y * gridDim.y) {
    // Sequential reduce by each thread
    const IndexT inner_idx = threadIdx.x + (IndexT)blockIdx.x * blockDim.x;
    StorageT reduced;

    if (sizeof(VecT) == sizeof(typename Op::Tcu)) {
      // Without vectrized load for general case
      reduced = kernel_thread_reduce<NBLA_CUDA_REDUCE_UNROLL_XY>(
          op, inner_idx, outer_idx, inner_size, blockDim.x * gridDim.x,
          inner_idx_conv, outer_idx_conv);
    } else {
      // With vectrized load for single-dimensional reduction.
      reduced =
          kernel_vectrized_thread_reduce_x<NBLA_CUDA_REDUCE_UNROLL_XY, VecT>(
              op, inner_idx, outer_idx, inner_size, blockDim.x * gridDim.x);
    }

    // Block reduce
    reduced = kernel_block_reduce_xy(op, reduced, sbuf);

    if (gridDim.x == 1) {
      // Reduction is completed by a block.
      if (threadIdx.x == 0) {
        op.store(outer_idx, reduced);
      }
    } else {
      // Inter-block reduce
      if (threadIdx.x == 0) {
        op.intermediate_store(blockIdx.x + (IndexT)gridDim.x * outer_idx,
                              reduced);
      }

      if (is_last_block(threadIdx.x, outer_idx, gridDim.x, block_counter)) {
        reduced = kernel_inter_block_reduce_xy(op, outer_idx, sbuf);
        if (threadIdx.x == 0) {
          op.store(outer_idx, reduced);
        }
      }
    }
  }
}

template <class Op>
__global__ void
kernel_reduce_y(Op op, int *const block_counter,
                const typename Op::IndexT inner_size,
                const typename Op::IndexT outer_size,
                const IndexConverter<typename Op::IndexT> inner_idx_conv,
                const IndexConverter<typename Op::IndexT> outer_idx_conv) {
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  // "extern __shared__" for template type cannot be used in CUDA.
  // This is a workaround.
  extern __shared__ uint8_t sbuf_char[];
  StorageT *sbuf = reinterpret_cast<StorageT *>(sbuf_char);

  // Grid-strided loop for outer dimensions
  for (IndexT outer_idx = threadIdx.x + (IndexT)blockIdx.x * blockDim.x;
       outer_idx < outer_size; outer_idx += blockDim.x * gridDim.x) {
    // Sequential reduce by each thread
    const IndexT inner_idx = threadIdx.y + (IndexT)blockIdx.y * blockDim.y;
    StorageT reduced = kernel_thread_reduce<NBLA_CUDA_REDUCE_UNROLL_Y>(
        op, inner_idx, outer_idx, inner_size, blockDim.y * gridDim.y,
        inner_idx_conv, outer_idx_conv);

    // Block reduce
    reduced = kernel_block_reduce_y(op, reduced, sbuf);

    if (gridDim.y == 1) {
      // Reduction is completed by a block.
      if (threadIdx.y == 0) {
        op.store(outer_idx, reduced);
      }
    } else {
      // Inter-block reduce
      if (threadIdx.y == 0) {
        op.intermediate_store(
            outer_idx + (IndexT)blockDim.x * gridDim.x * blockIdx.y, reduced);
      }

      if (is_last_block(threadIdx.y, outer_idx, gridDim.y, block_counter)) {
        reduced = kernel_inter_block_reduce_y(op, outer_idx, sbuf);
        if (threadIdx.y == 0) {
          op.store(outer_idx, reduced);
        }
      }
    }
  }
}

template <class Op> __global__ void kernel_copy(const Size_t size, Op op) {
  NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, size) {
    op.store(idx, op.make_storage(op.input[idx], 0));
  }
}

static uint32_t get_strided_grid(const Size_t grid_dim) {
  return static_cast<uint32_t>(NBLA_CEIL_SIZE_T_DIV(
      grid_dim, NBLA_CEIL_SIZE_T_DIV(grid_dim, NBLA_CUDA_REDUCE_MAX_BLOCKS)));
}

template <class Op>
void fast_reduce_xy(const Context &ctx, Op &op, const ReduceSetup &setup) {
  // Some procedures can be moved to ReduceSetup::operator() if implementing the
  // mechanism to define Op::IndexT when Function::setup.
  using Tcu = typename Op::Tcu;
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Firstly determine an ideal parallelism for large enough input shape.
  const auto size_x_pow2 = next_pow2_floor(setup.size_x);
  const auto size_y_pow2 = next_pow2_floor(setup.size_y);
  const auto block_y = std::min(
      size_y_pow2, Size_t(NBLA_CUDA_REDUCE_NUM_THREADS / CUDA_WARP_SIZE));
  const auto block_x = std::max(Size_t(CUDA_WARP_SIZE),
                                Size_t(NBLA_CUDA_REDUCE_NUM_THREADS / block_y));
  dim3 block_dim = dim3(block_x, block_y);
  dim3 grid_dim(1, NBLA_CEIL_SIZE_T_DIV(setup.size_y, block_dim.y));

  // Try to keep sufficient parallelism for any input shape.
  const auto min_elements_per_thread = NBLA_CUDA_REDUCE_UNROLL_XY;
  auto elements_per_thread = setup.size_x / block_dim.x;

  // Assign more threads into each reduction.
  while (block_dim.x * NBLA_CUDA_REDUCE_UNROLL_XY < size_x_pow2 &&
         block_dim.y > 1 && elements_per_thread > min_elements_per_thread) {
    block_dim.x *= 2;
    block_dim.y /= 2;
    grid_dim.y = NBLA_CEIL_SIZE_T_DIV(setup.size_y, block_dim.y);
    elements_per_thread /= 2;
  }

  // Assign more blocks into each reduction.
  if (block_dim.x < size_x_pow2 && block_dim.y == 1) {
    while (grid_dim.x < setup.min_blocks &&
           block_dim.x * grid_dim.x * NBLA_CUDA_REDUCE_UNROLL_XY <
               size_x_pow2 &&
           elements_per_thread > min_elements_per_thread) {
      grid_dim.x *= 2;
      elements_per_thread /= 2;
    }
  }

  // Determine the grid size for grid-strided loop.
  grid_dim.x = get_strided_grid(grid_dim.x);
  grid_dim.y = get_strided_grid(grid_dim.y);

  // Prepare the temporary buffers for inter-block reduce
  NdArray buf_arr;
  NdArray block_counter_arr;
  op.buf = nullptr;
  int *block_counter = nullptr;
  if (grid_dim.x > 1) {
    NBLA_CHECK(block_dim.x == NBLA_CUDA_REDUCE_NUM_THREADS && block_dim.y == 1,
               error_code::value, "Block division failed in reduction.",
               "Please report this error to the developer team.");

    buf_arr.reshape(Shape_t{block_dim.y * grid_dim.y,
                            grid_dim.x * static_cast<Size_t>(sizeof(StorageT))},
                    true);
    op.buf = buf_arr.cast(get_dtype<char>(), ctx, true)
                 ->template pointer<StorageT>();

    block_counter_arr.reshape(Shape_t{setup.size_y}, true);
    block_counter_arr.zero();
    block_counter =
        block_counter_arr.cast(get_dtype<int>(), ctx)->template pointer<int>();
  }

  // Determine the shared memory size for block reduce.
  int smem_size = 0;
  if (block_dim.x > 1) {
    smem_size = NBLA_CUDA_REDUCE_NUM_THREADS * sizeof(StorageT);
  }

  // Utility to calculate the indices fast.
  const IndexConverter<IndexT> inner_idx_conv(setup.strides_x,
                                              setup.strides_x_input);
  const IndexConverter<IndexT> outer_idx_conv(setup.strides_y,
                                              setup.strides_y_input);

  // Reduction kernel launch
  const bool only_x = (setup.ndim_x == 1 && setup.ndim_y <= 1);
  auto kernel = kernel_reduce_xy<Tcu, Op>;

  if (only_x) {
    // Call the faster implementation.
    if (sizeof(Tcu) == 4) {
      // A float4 stores the 4 elements of 32-bit type (float).
      kernel = kernel_reduce_xy<float4, Op>;
    } else if (sizeof(Tcu) == 2) {
      // A float2 stores the 4 elements of 16-bit type (half).
      kernel = kernel_reduce_xy<float2, Op>;
    } else {
      NBLA_ERROR(error_code::type,
                 "The size of types for reduction must be 2 or 4 bytes. "
                 "Please report this error to the developer team.");
    }
  }

  kernel<<<grid_dim, block_dim, smem_size>>>(op, block_counter, setup.size_x,
                                             setup.size_y, inner_idx_conv,
                                             outer_idx_conv);
  NBLA_CUDA_KERNEL_CHECK();
}

template <class Op>
void fast_reduce_y(const Context &ctx, Op &op, const ReduceSetup &setup) {
  // Some procedures can be moved to ReduceSetup::operator() if implementing the
  // mechanism to define Op::IndexT when Function::setup.
  using Tcu = typename Op::Tcu;
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  // Firstly determine an ideal parallelism for large enough input shape.
  dim3 block_dim(NBLA_CUDA_REDUCE_NUM_THREADS, 1);
  dim3 grid_dim(NBLA_CEIL_SIZE_T_DIV(setup.size_x, block_dim.x), 1);

  // Try to keep sufficient parallelism for any input shape.
  const auto size_y_pow2 = next_pow2_floor(setup.size_y);
  const auto min_elements_per_thread = NBLA_CUDA_REDUCE_UNROLL_Y;
  auto elements_per_thread = setup.size_y / block_dim.y;

  // Assign more threads into each reduction.
  while (block_dim.x > CUDA_WARP_SIZE &&
         block_dim.y * NBLA_CUDA_REDUCE_UNROLL_Y < size_y_pow2 &&
         elements_per_thread > min_elements_per_thread) {
    block_dim.x /= 2;
    block_dim.y *= 2;
    grid_dim.x = NBLA_CEIL_SIZE_T_DIV(setup.size_x, block_dim.x);
    elements_per_thread /= 2;
  }

  // Assign more blocks into each reduction.
  if (block_dim.x == CUDA_WARP_SIZE && block_dim.y < size_y_pow2) {
    while (grid_dim.x * grid_dim.y < setup.min_blocks &&
           block_dim.y * grid_dim.y * NBLA_CUDA_REDUCE_UNROLL_Y < size_y_pow2 &&
           elements_per_thread > min_elements_per_thread) {
      grid_dim.y *= 2;
      elements_per_thread /= 2;
    }
  }

  // Determine the grid size for grid-strided loop.
  grid_dim.x = get_strided_grid(grid_dim.x);
  grid_dim.y = get_strided_grid(grid_dim.y);

  // Prepare the temporary buffers for inter-block reduce
  NdArray buf_arr;
  NdArray block_counter_arr;
  op.buf = nullptr;
  int *block_counter = nullptr;
  if (grid_dim.y > 1) {
    buf_arr.reshape(Shape_t{block_dim.x * grid_dim.x,
                            grid_dim.y * static_cast<Size_t>(sizeof(StorageT))},
                    true);
    op.buf = buf_arr.cast(get_dtype<char>(), ctx, true)
                 ->template pointer<StorageT>();

    block_counter_arr.reshape(Shape_t{setup.size_x}, true);
    block_counter_arr.zero();
    block_counter =
        block_counter_arr.cast(get_dtype<int>(), ctx)->template pointer<int>();
  }

  // Determine the shared memory size for block reduce.
  int smem_size = 0;
  if (block_dim.y > 1) {
    smem_size = NBLA_CUDA_REDUCE_NUM_THREADS * sizeof(StorageT);
  }

  // Utility to calculate the indices fast.
  const IndexConverter<IndexT> inner_idx_conv(setup.strides_y,
                                              setup.strides_y_input);
  const IndexConverter<IndexT> outer_idx_conv(setup.strides_x,
                                              setup.strides_x_input);

  // Reduction kernel launch
  kernel_reduce_y<<<grid_dim, block_dim, smem_size>>>(
      op, block_counter, setup.size_y, setup.size_x, inner_idx_conv,
      outer_idx_conv);
  NBLA_CUDA_KERNEL_CHECK();
}

template <class Op>
void fast_reduce(const Context &ctx, Op op, const ReduceSetup &setup) {
  if (setup.copy_only) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T(kernel_copy, setup.size_input, op);
    return;
  }

  if (setup.reduce_x) {
    fast_reduce_xy(ctx, op, setup);
  } else {
    fast_reduce_y(ctx, op, setup);
  }
}
}
#endif
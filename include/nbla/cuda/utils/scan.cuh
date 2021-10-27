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

#ifndef __NBLA_CUDA_UTILS_SCAN_CUH__
#define __NBLA_CUDA_UTILS_SCAN_CUH__

#include <nbla/common.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/utils/warp_shuffle.cuh>

namespace nbla {

constexpr Size_t NBLA_CUDA_SCAN_MAX_BLOCKS = 65535;

template <class Op, bool accum, bool exclusive, bool reverse>
__global__ void
kernel_scan_sequential(const typename Op::IndexT size_outer_inner, Op op,
                       const typename Op::IndexT size_scan,
                       const typename Op::IndexT size_inner) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;
  NBLA_CUDA_KERNEL_LOOP(idx, size_outer_inner) {
    const IndexT i0 = idx / size_inner;
    const IndexT i2 = idx % size_inner;

    const IndexT offset = i0 * size_scan * size_inner + i2;
    StorageT cum = op.init();
    for (IndexT k = 0; k < size_scan; ++k) {
      const int i1 = reverse ? size_scan - k - 1 : k;
      const int idx = i1 * size_inner + offset;

      if (exclusive) {
        op.store<accum>(idx, cum);
        cum = op(cum, op.input[idx]);
      } else {
        cum = op(cum, op.input[idx]);
        op.store<accum>(idx, cum);
      }
    }
  }
}

constexpr Size_t NUM_ELEMENTS_PER_THREADS = 32;
constexpr Size_t NUM_THREADS_PER_BLOCK = NBLA_CUDA_NUM_THREADS;

template <class Op, bool exclusive, bool reverse>
__global__ void kernel_scan_sequential_inter_block_pre(
    Op op, const typename Op::IndexT size_outer,
    const typename Op::IndexT size_scan, const typename Op::IndexT size_inner,
    const typename Op::IndexT size_scan_buf) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;
  // Grid-stride loop for outer and inner axis.
  for (IndexT i02 = blockIdx.y * blockDim.y + threadIdx.y;
       i02 < size_outer * size_inner; i02 += gridDim.y * blockDim.y) {
    const IndexT i0 = i02 / size_inner;
    const IndexT i2 = i02 % size_inner;
    // Grid-stride loop for scan axis.
    for (IndexT i1_buf = blockIdx.x; i1_buf < size_scan_buf;
         i1_buf += gridDim.x) {
      StorageT accum = op.init();
      for (IndexT k = 0; k < NUM_ELEMENTS_PER_THREADS; k++) {
        const IndexT i1 = i1_buf * NUM_ELEMENTS_PER_THREADS +
                          (reverse ? NUM_ELEMENTS_PER_THREADS - k - 1 : k);
        const IndexT idx = (i0 * size_scan + i1) * size_inner + i2;
        if (i1 < size_scan) {
          if (exclusive) {
            op.store<false>(idx, accum);
            accum = op(accum, op.input[idx]);
          } else {
            accum = op(accum, op.input[idx]);
            op.store<false>(idx, accum);
          }
        }
      }
      const IndexT buf_idx = (i0 * size_scan_buf + i1_buf) * size_inner + i2;
      op.intermediate_store(buf_idx, accum);
    }
  }
}

template <class Op, bool accum>
__global__ void kernel_scan_sequential_inter_block_post(
    Op op, const typename Op::IndexT size_outer,
    const typename Op::IndexT size_scan, const typename Op::IndexT size_inner,
    const typename Op::IndexT size_scan_buf) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;
  // Grid-stride loop for outer and inner axis.
  for (IndexT i02 = blockIdx.y * blockDim.y + threadIdx.y;
       i02 < size_outer * size_inner; i02 += gridDim.y * blockDim.y) {
    const IndexT i0 = i02 / size_inner;
    const IndexT i2 = i02 % size_inner;
    // Grid-stride loop for scan axis.
    for (IndexT i1_buf = blockIdx.x; i1_buf < size_scan_buf;
         i1_buf += gridDim.x) {
      const IndexT buf_idx = (i0 * size_scan_buf + i1_buf) * size_inner + i2;
      const typename Op::StorageT buf_val = op.buf[buf_idx];
      for (IndexT k = 0; k < NUM_ELEMENTS_PER_THREADS; k++) {
        const IndexT i1 = i1_buf * NUM_ELEMENTS_PER_THREADS + k;
        if (i1 < size_scan) {
          const IndexT idx = (i0 * size_scan + i1) * size_inner + i2;
          op.store<accum>(idx, op(op.input[idx], buf_val));
        }
      }
    }
  }
}

/**
 * @brief Perform warp scan
 *
 * TODO: Extract load and store codes from this function when block scan is
 * implemented.
 */
template <class Op, bool exclusive, bool reverse, bool accum, int num_threads>
__device__ typename Op::StorageT
warp_scan(Op op, const typename Op::IndexT i0,
          const typename Op::IndexT i1_base,
          const typename Op::IndexT size_scan,
          const typename Op::StorageT &last_block_total) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // The following code assumes warpSize == 32
  static_assert(CUDA_WARP_SIZE == 32);
  static_assert(num_threads == CUDA_WARP_SIZE);

  const auto tidx = threadIdx.x;
  const IndexT i1 = i1_base + tidx;

  // Load from input global memory to register.
  StorageT v;
  if (i1 < size_scan) {
    const IndexT idx = i0 * size_scan + i1;
    if (tidx == 0 && !reverse) {
      v = op(op.input[idx], last_block_total);
    } else if (tidx == num_threads - 1 && reverse) {
      v = op(op.input[idx], last_block_total);
    } else {
      v = op.input[idx];
    }
  } else {
    v = op.init();
  }

  // Perform scan by shuffle operations.
  if (reverse) {
    for (int i = 1; i <= num_threads; i *= 2) {
      const StorageT v_pair = warp::shuffle_down(v, i);
      if (tidx < num_threads - i) {
        v = op(v, v_pair);
      }
    }
  } else {
    for (int i = 1; i <= num_threads; i *= 2) {
      const StorageT v_pair = warp::shuffle_up(v, i);
      if (tidx >= i) {
        v = op(v, v_pair);
      }
    }
  }

  // Store the results into output global memory.
  if (i1 < size_scan) {
    const IndexT idx = i0 * size_scan + i1;
    if (exclusive) {
      if (reverse) {
        if (tidx == num_threads - 1 || i1 == size_scan - 1) {
          op.store<accum>(idx, last_block_total);
        }
        if (tidx > 0 && i1 > 0) {
          op.store<accum>(idx - 1, v);
        }
      } else {
        if (tidx == 0) {
          op.store<accum>(idx, last_block_total);
        }
        if (tidx < num_threads - 1 && i1 + 1 < size_scan) {
          op.store<accum>(idx + 1, v);
        }
      }
    } else {
      op.store<accum>(idx, v);
    }
  }

  // Broadcast total value of the warp scan and return it.
  if (reverse) {
    v = warp::shuffle(v, 0);
  } else {
    v = warp::shuffle(v, num_threads - 1);
  }
  return v;
}

template <class Op, int block_dim_x, int block_dim_y, bool accum,
          bool exclusive, bool reverse>
__global__ void kernel_scan_parallel(Op op,
                                     const typename Op::IndexT size_outer,
                                     const typename Op::IndexT size_scan) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Grid-stride loop for outer axis
  for (IndexT i0 = blockIdx.x * block_dim_y + threadIdx.y; i0 < size_outer;
       i0 += gridDim.x * block_dim_y) {
    StorageT last_val = op.init();
    const IndexT n_itr = NBLA_CEIL_SIZE_T_DIV(size_scan, block_dim_x);
    for (IndexT k = 0; k < n_itr; k++) {
      IndexT i1_base = (reverse ? (n_itr - k - 1) : k) * block_dim_x;
      last_val = warp_scan<Op, exclusive, reverse, accum, block_dim_x>(
          op, i0, i1_base, size_scan, last_val);
    }
  }
}

template <class Op, int block_dim_x, int block_dim_y, bool exclusive,
          bool reverse>
__global__ void
kernel_scan_parallel_inter_block_pre(Op op,
                                     const typename Op::IndexT size_outer,
                                     const typename Op::IndexT size_scan,
                                     const typename Op::IndexT size_scan_buf) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Grid-stride loop for outer axis
  for (IndexT i0 = blockIdx.y * block_dim_y + threadIdx.y; i0 < size_outer;
       i0 += gridDim.y * block_dim_y) {
    // Grid-stride loop for scan axis
    for (IndexT i1_buf = blockIdx.x; i1_buf < size_scan_buf;
         i1_buf += gridDim.x) {
      const IndexT i1_base = i1_buf * block_dim_x;
      const auto last_val =
          warp_scan<Op, exclusive, reverse, false /* accum */, block_dim_x>(
              op, i0, i1_base, size_scan, op.init());
      const IndexT buf_idx = i0 * size_scan_buf + i1_buf;
      if (threadIdx.x == 0) {
        op.intermediate_store(buf_idx, last_val);
      }
    }
  }
}

template <class Op, int block_dim_x, int block_dim_y, bool accum>
__global__ void
kernel_scan_parallel_inter_block_post(Op op,
                                      const typename Op::IndexT size_outer,
                                      const typename Op::IndexT size_scan,
                                      const typename Op::IndexT size_scan_buf) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Grid-stride loop for outer axis
  for (IndexT i0 = blockIdx.y * block_dim_y + threadIdx.y; i0 < size_outer;
       i0 += gridDim.y * block_dim_y) {
    // Grid-stride loop for scan axis
    for (IndexT i1_buf = blockIdx.x; i1_buf < size_scan_buf;
         i1_buf += gridDim.x) {
      const IndexT i1_base = i1_buf * block_dim_x;
      const IndexT i1 = i1_base + threadIdx.x;

      if (i1 < size_scan) {
        const IndexT idx = i0 * size_scan + i1;
        const IndexT buf_idx = i0 * size_scan_buf + i1_buf;
        op.store<accum>(idx, op(op.input[idx], op.buf[buf_idx]));
      }
    }
  }
}

// This prototype declaration is needed because inter-block scan implementation
// calls `scan` recursively.
template <class Op>
void scan(const Context &ctx, Op &op, const ScanSetup &setup, const bool accum);

/**
 * @brief Perform sequential scan with single kernel.
 */
template <class Op, bool accum, bool exclusive, bool reverse>
void scan_sequential(const Context &ctx, Op &op, const ScanSetup &setup) {
  auto kernel = kernel_scan_sequential<Op, accum, exclusive, reverse>;
  const auto size = setup.size_outer * setup.size_inner;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, op, setup.size_scan,
                                 setup.size_inner);
}

/**
 * @brief Perform sequential scan with inter-block scanning.
 */
template <class Op, bool accum, bool exclusive, bool reverse>
void scan_sequential_inter_block(const Context &ctx, Op &op,
                                 const ScanSetup &setup) {
  using Tcu = typename Op::Tcu;
  using IndexT = typename Op::IndexT;

  // Step 1
  IndexT size_scan_buf =
      NBLA_CEIL_SIZE_T_DIV(setup.size_scan, NUM_ELEMENTS_PER_THREADS);
  Variable v_pre_buf({setup.size_outer, size_scan_buf, setup.size_inner});
  Variable v_pre_out({setup.size_outer, setup.size_scan, setup.size_inner});
  Op op_pre(op.input, v_pre_out.cast_data_and_get_pointer<Tcu>(ctx, true));
  op_pre.buf = v_pre_buf.cast_data_and_get_pointer<Tcu>(ctx, true);
  const dim3 grid_dim(
      std::min((Size_t)size_scan_buf, NBLA_CUDA_SCAN_MAX_BLOCKS),
      std::min(NBLA_CEIL_SIZE_T_DIV(setup.size_outer * setup.size_inner,
                                    NUM_THREADS_PER_BLOCK),
               NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(1, NUM_THREADS_PER_BLOCK);
  {
    auto kernel =
        kernel_scan_sequential_inter_block_pre<Op, exclusive, reverse>;
    kernel<<<grid_dim, block_dim>>>(op_pre, setup.size_outer, setup.size_scan,
                                    setup.size_inner, size_scan_buf);
  }

  // Step 2
  ScanSetup setup_mid;
  setup_mid(v_pre_buf.shape(), 1 /* axis */, true /* exclusive */,
            setup.reverse);
  Variable v_mid_out({setup.size_outer, size_scan_buf, setup.size_inner});
  Op op_mid(v_pre_buf.get_data_pointer<Tcu>(ctx),
            v_mid_out.cast_data_and_get_pointer<Tcu>(ctx, true));
  scan(ctx, op_mid, setup_mid, false /* accum */);

  // Step 3
  Op op_post(v_pre_out.get_data_pointer<Tcu>(ctx), op.output);
  op_post.buf = v_mid_out.cast_data_and_get_pointer<Tcu>(ctx, false);
  {
    auto kernel = kernel_scan_sequential_inter_block_post<Op, accum>;
    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan,
                                    setup.size_inner, size_scan_buf);
  }
}

/**
 * @brief Perform prefix scan algorithm with single kernel.
 */
template <class Op, bool accum, bool exclusive, bool reverse>
void scan_parallel(const Context &ctx, Op &op, const ScanSetup &setup) {
  using IndexT = typename Op::IndexT;

  constexpr IndexT block_dim_x = CUDA_WARP_SIZE;
  constexpr IndexT block_dim_y = NBLA_CUDA_NUM_THREADS / block_dim_x;

  const dim3 grid_dim(std::min(
      NBLA_CEIL_SIZE_T_DIV((Size_t)setup.size_outer, (Size_t)block_dim_x),
      (Size_t)NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(block_dim_x, block_dim_y);

  auto kernel = kernel_scan_parallel<Op, block_dim_x, block_dim_y, accum,
                                     exclusive, reverse>;
  kernel<<<grid_dim, block_dim>>>(op, setup.size_outer, setup.size_scan);
  NBLA_CUDA_KERNEL_CHECK();
}

/**
 * @brief Perform prefix scan algorithm with inter-block scanning.
 */
template <class Op, bool accum, bool exclusive, bool reverse>
void scan_parallel_inter_block(const Context &ctx, Op &op,
                               const ScanSetup &setup) {
  using Tcu = typename Op::Tcu;
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  // Step 1: Block wise scan
  constexpr IndexT block_dim_x = CUDA_WARP_SIZE;
  constexpr IndexT block_dim_y = NBLA_CUDA_NUM_THREADS / block_dim_x;
  IndexT size_scan_buf = NBLA_CEIL_SIZE_T_DIV(setup.size_scan, block_dim_x);

  Variable pre_buf({setup.size_outer, size_scan_buf});
  Variable pre_out({setup.size_outer, setup.size_scan});
  Op op_pre(op.input, pre_out.cast_data_and_get_pointer<Tcu>(ctx, true));
  op_pre.buf = pre_buf.cast_data_and_get_pointer<StorageT>(ctx, true);
  const dim3 grid_dim(
      std::min((Size_t)size_scan_buf, NBLA_CUDA_SCAN_MAX_BLOCKS),
      std::min(NBLA_CEIL_SIZE_T_DIV(setup.size_outer, block_dim_y),
               NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(block_dim_x, block_dim_y);

  {
    auto kernel =
        kernel_scan_parallel_inter_block_pre<Op, block_dim_x, block_dim_y,
                                             exclusive, reverse>;
    kernel<<<grid_dim, block_dim>>>(op_pre, setup.size_outer, setup.size_scan,
                                    size_scan_buf);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Step 2: Scan block total value
  Variable mid_out(pre_buf.shape());
  Op op_mid(pre_buf.get_data_pointer<Tcu>(ctx),
            mid_out.cast_data_and_get_pointer<Tcu>(ctx, true));

  ScanSetup setup_mid;
  setup_mid(mid_out.shape(), 1 /* axis */, true /* exclusive */, setup.reverse);
  scan(ctx, op_mid, setup_mid, false /* accum */);

  // Step 3: Add scanned block total value
  {
    Op op_post(pre_out.get_data_pointer<Tcu>(ctx), op.output);
    op_post.buf = mid_out.cast_data_and_get_pointer<StorageT>(ctx, false);

    auto kernel = kernel_scan_parallel_inter_block_post<Op, block_dim_x,
                                                        block_dim_y, accum>;
    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan,
                                    size_scan_buf);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

/**
 * @brief Switching scan algorithm by input shape and axis.
 */
template <class Op, bool accum, bool exclusive, bool reverse>
void scan_impl(const Context &ctx, Op &op, const ScanSetup &setup) {
  // Following switching condition is determined heauristically.
  if (setup.size_inner == 1) {
    if (setup.size_scan <= 2048) {
      scan_parallel<Op, accum, exclusive, reverse>(ctx, op, setup);
    } else {
      scan_parallel_inter_block<Op, accum, exclusive, reverse>(ctx, op, setup);
    }
  } else {
    if (setup.size_scan <= 128) {
      scan_sequential<Op, accum, exclusive, reverse>(ctx, op, setup);
    } else {
      scan_sequential_inter_block<Op, accum, exclusive, reverse>(ctx, op,
                                                                 setup);
    }
  }
}

// The following `dispatch_*` functions are only for the purpose of dealing with
// template arguments of `scan_impl`.

template <class Op, bool accum, bool exclusive>
void scan_dispatch_reverse(const Context &ctx, Op &op, const ScanSetup &setup) {
  if (setup.reverse) {
    scan_impl<Op, accum, exclusive, true>(ctx, op, setup);
  } else {
    scan_impl<Op, accum, exclusive, false>(ctx, op, setup);
  }
}

template <class Op, bool accum>
void scan_dispatch_exclusive(const Context &ctx, Op &op,
                             const ScanSetup &setup) {
  if (setup.exclusive) {
    scan_dispatch_reverse<Op, accum, true>(ctx, op, setup);
  } else {
    scan_dispatch_reverse<Op, accum, false>(ctx, op, setup);
  }
}

// Scan interface
template <class Op>
void scan(const Context &ctx, Op &op, const ScanSetup &setup,
          const bool accum) {
  if (accum) {
    scan_dispatch_exclusive<Op, true>(ctx, op, setup);
  } else {
    scan_dispatch_exclusive<Op, false>(ctx, op, setup);
  }
}
}

#endif
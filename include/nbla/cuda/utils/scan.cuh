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

template <class Op>
void scan(const Context &ctx, Op op, const ScanSetup &setup);

template <class Op, bool exclusive, bool reverse, bool accum>
__global__ void kernel_scan_naive(const typename Op::IndexT size_outer_inner,
                                  Op op, const typename Op::IndexT size_scan,
                                  const typename Op::IndexT size_inner) {
  NBLA_CUDA_KERNEL_LOOP(idx, size_outer_inner) {
    const typename Op::IndexT i0 = idx / size_inner;
    const typename Op::IndexT i2 = idx % size_inner;

    const typename Op::IndexT offset = i0 * size_scan * size_inner + i2;
    typename Op::StorageT acc = op.init();
    for (typename Op::IndexT k = 0; k < size_scan; ++k) {
      const int i1 = reverse ? size_scan - k - 1 : k;
      const int idx = i1 * size_inner + offset;

      if (exclusive) {
        op.store<accum>(idx, acc);
        acc = op(acc, op.input[idx]);
      } else {
        acc = op(acc, op.input[idx]);
        op.store<accum>(idx, acc);
      }
    }
  }
}

template <class Op>
void scan_naive(const Context &ctx, Op op, const ScanSetup &setup) {
  auto kernel = kernel_scan_naive<Op, false /* exclusive */,
                                  false /* reverse */, false /* accum */>;

  if (setup.exclusive) {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_naive<Op, true, true, true>
                           : kernel_scan_naive<Op, true, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_naive<Op, true, false, true>
                           : kernel_scan_naive<Op, true, false, false>;
    }
  } else {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_naive<Op, false, true, true>
                           : kernel_scan_naive<Op, false, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_naive<Op, false, false, true>
                           : kernel_scan_naive<Op, false, false, false>;
    }
  }

  const auto size = setup.size_outer * setup.size_inner;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, op, setup.size_scan,
                                 setup.size_inner);
}

constexpr Size_t NUM_ELEMENTS_PER_THREADS = 32;
constexpr Size_t NUM_THREADS_PER_BLOCK = NBLA_CUDA_NUM_THREADS;

template <class Op, bool exclusive, bool reverse>
__global__ void
kernel_scan_naive_inter_block_pre(Op op, const typename Op::IndexT size_outer,
                                  const typename Op::IndexT size_scan,
                                  const typename Op::IndexT size_inner,
                                  const typename Op::IndexT size_scan_buf) {
  using IndexT = typename Op::IndexT;
  // Grid-stride loop for outer and inner axis.
  for (IndexT i02 = blockIdx.y * blockDim.y + threadIdx.y;
       i02 < size_outer * size_inner; i02 += gridDim.y * blockDim.y) {
    const IndexT i0 = i02 / size_inner;
    const IndexT i2 = i02 % size_inner;
    // Grid-stride loop for scan axis.
    for (IndexT i1_buf = blockIdx.x; i1_buf < size_scan_buf;
         i1_buf += gridDim.x) {
      typename Op::StorageT acc = op.init();
      for (IndexT k = 0; k < NUM_ELEMENTS_PER_THREADS; k++) {
        const IndexT i1 = i1_buf * NUM_ELEMENTS_PER_THREADS +
                          (reverse ? NUM_ELEMENTS_PER_THREADS - k - 1 : k);
        const IndexT idx = (i0 * size_scan + i1) * size_inner + i2;
        if (i1 < size_scan) {
          if (exclusive) {
            op.store<false>(idx, acc);
            acc = op(acc, op.input[idx]);
          } else {
            acc = op(acc, op.input[idx]);
            op.store<false>(idx, acc);
          }
        }
      }
      const IndexT buf_idx = (i0 * size_scan_buf + i1_buf) * size_inner + i2;
      op.intermediate_store(buf_idx, acc);
    }
  }
}

template <class Op, bool accum>
__global__ void
kernel_scan_naive_inter_block_post(Op op, const typename Op::IndexT size_outer,
                                   const typename Op::IndexT size_scan,
                                   const typename Op::IndexT size_inner,
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

template <class Op>
void scan_naive_inter_block(const Context &ctx, Op op, const ScanSetup &setup) {
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
    auto kernel = kernel_scan_naive_inter_block_pre<Op, false /* exclusive */,
                                                    false /* reverse */>;
    if (setup.exclusive) {
      kernel = setup.reverse
                   ? kernel_scan_naive_inter_block_pre<Op, true, true>
                   : kernel_scan_naive_inter_block_pre<Op, true, false>;
    } else {
      kernel = setup.reverse
                   ? kernel_scan_naive_inter_block_pre<Op, false, true>
                   : kernel_scan_naive_inter_block_pre<Op, false, false>;
    }
    kernel<<<grid_dim, block_dim>>>(op_pre, setup.size_outer, setup.size_scan,
                                    setup.size_inner, size_scan_buf);
  }

  // Step 2
  ScanSetup setup_mid;
  setup_mid(v_pre_buf.shape(), 1 /* axis */, true /* exclusive */,
            setup.reverse, false);
  Variable v_mid_out({setup.size_outer, size_scan_buf, setup.size_inner});
  Op op_mid(v_pre_buf.get_data_pointer<Tcu>(ctx),
            v_mid_out.cast_data_and_get_pointer<Tcu>(ctx, true));
  scan(ctx, op_mid, setup_mid);

  // Step 3
  Op op_post(v_pre_out.get_data_pointer<Tcu>(ctx), op.output_);
  op_post.buf = v_mid_out.cast_data_and_get_pointer<Tcu>(ctx, false);
  {
    auto kernel = setup.accum ? kernel_scan_naive_inter_block_post<Op, true>
                              : kernel_scan_naive_inter_block_post<Op, false>;
    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan,
                                    setup.size_inner, size_scan_buf);
  }
}

constexpr Size_t NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK = 32;
constexpr Size_t NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER = 16;

template <class Op, bool exclusive, bool reverse, bool accum, int num_threads>
__device__ typename Op::StorageT
warp_scan(Op op, const typename Op::IndexT i0,
          const typename Op::IndexT i1_base,
          const typename Op::IndexT size_scan,
          const typename Op::StorageT &last_block_total) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  const auto tidx = threadIdx.x;
  const IndexT i1 = i1_base + tidx;

  StorageT v;
  if (i1 < size_scan) {
    const IndexT idx = i0 * size_scan + i1;
    if (tidx == 0 && !reverse) {
      v = op(op.input[idx], last_block_total);
    } else if (tidx == 31 && reverse) {
      v = op(op.input[idx], last_block_total);
    } else {
      v = op.input[idx];
    }
  } else {
    v = op.init();
  }

  if (reverse) {
    for (int i = 1; i <= 32; i *= 2) {
      const StorageT v_pair = warp::shuffle_down(v, i);
      if (tidx < 32 - i) {
        v = op(v, v_pair);
      }
    }
  } else {
    for (int i = 1; i <= 32; i *= 2) {
      const StorageT v_pair = warp::shuffle_up(v, i);
      if (tidx >= i) {
        v = op(v, v_pair);
      }
    }
  }

  if (i1 < size_scan) {
    const IndexT idx = i0 * size_scan + i1;
    if (exclusive) {
      if (reverse) {
        if (tidx == 31 || i1 == size_scan - 1) {
          op.store<accum>(idx, last_block_total);
        }
        if (tidx > 0 && i1 > 0) {
          op.store<accum>(idx - 1, v);
        }
      } else {
        if (tidx == 0) {
          op.store<accum>(idx, last_block_total);
        }
        if (tidx < 31 && i1 + 1 < size_scan) {
          op.store<accum>(idx + 1, v);
        }
      }
    } else {
      op.store<accum>(idx, v);
    }
  }

  if (reverse) {
    v = warp::shuffle(v, 0);
  } else {
    v = warp::shuffle(v, 31);
  }
  return v;
}

template <class Op, typename StorageT, bool exclusive, bool reverse, bool accum,
          int num_threads>
__device__ StorageT block_scan_parallel(Op op, const typename Op::IndexT i0,
                                        const typename Op::IndexT i1_base,
                                        const typename Op::IndexT size_scan,
                                        const StorageT &last_block_total) {
  using IndexT = typename Op::IndexT;

  const auto tidx = threadIdx.x;
  const auto tidy = threadIdx.y;

  // Load to shared memory
  __shared__ StorageT smem_[NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER]
                           [2 * num_threads];
  auto smem = smem_[tidy];

  const Size_t i1 = i1_base + tidx;
  if (i1 < size_scan) {
    const IndexT idx = i0 * size_scan + i1;
    if (tidx == 0 && !reverse) {
      smem[tidx] = op(last_block_total, op.input[idx]);
    } else {
      smem[tidx] = op.input[idx];
    }
  } else {
    smem[tidx] = op.init();
  }

  if (i1 + num_threads < size_scan) {
    const IndexT idx = i0 * size_scan + (i1 + num_threads);
    if (tidx + num_threads == 2 * num_threads - 1 && reverse) {
      smem[tidx + num_threads] = op(last_block_total, op.input[idx]);
    } else {
      smem[tidx + num_threads] = op.input[idx];
    }
  } else {
    smem[tidx + num_threads] = op.init();
  }
  __syncthreads();

  // Up-sweep phase
  for (auto p = 1; p <= num_threads; p <<= 1) {
    auto offset = (tidx * 2 + 1) * p - 1;
    if (offset + p < num_threads * 2) {
      if (reverse) {
        smem[num_threads * 2 - (offset + p) - 1] =
            op(smem[num_threads * 2 - offset - 1],
               smem[num_threads * 2 - (offset + p) - 1]);
      } else {
        smem[offset + p] = op(smem[offset], smem[offset + p]);
      }
    }
    __syncthreads();
  }

  // Down-sweep phase
  for (auto p = num_threads / 2; p >= 1; p >>= 1) {
    auto offset = 2 * (tidx + 1) * p - 1;
    if (offset + p < num_threads * 2) {
      if (reverse) {
        smem[num_threads * 2 - (offset + p) - 1] =
            op(smem[num_threads * 2 - offset - 1],
               smem[num_threads * 2 - (offset + p) - 1]);
      } else {
        smem[offset + p] = op(smem[offset], smem[offset + p]);
      }
    }
    __syncthreads();
  }

  // Store to global memory
  if (exclusive) {
    if (reverse) {
      if (i1 < size_scan) {
        const IndexT idx = i0 * size_scan + i1;
        op.store<accum>(idx, smem[tidx + 1]);
      }
      if (i1 + num_threads < size_scan) {
        const IndexT idx = i0 * size_scan + (i1 + num_threads);
        if (tidx + num_threads == 2 * num_threads - 1) {
          op.store<accum>(idx, last_block_total);
        } else {
          op.store<accum>(idx, smem[tidx + num_threads + 1]);
        }
      }
    } else {
      if (i1 < size_scan) {
        const IndexT idx = i0 * size_scan + i1;
        if (tidx == 0) {
          op.store<accum>(idx, last_block_total);
        } else {
          op.store<accum>(idx, smem[tidx - 1]);
        }
      }
      if (i1 + num_threads < size_scan) {
        const IndexT idx = i0 * size_scan + (i1 + num_threads);
        op.store<accum>(idx, smem[tidx + num_threads - 1]);
      }
    }
  } else {
    if (i1 < size_scan) {
      const IndexT idx = i0 * size_scan + i1;
      op.store<accum>(idx, smem[tidx]);
    }
    if (i1 + num_threads < size_scan) {
      const IndexT idx = i0 * size_scan + (i1 + num_threads);
      op.store<accum>(idx, smem[tidx + num_threads]);
    }
  }

  __syncthreads();
  const StorageT total_val = reverse ? smem[0] : smem[2 * num_threads - 1];
  return total_val;
}

template <class Op, int block_dim_x, int block_dim_y, bool exclusive,
          bool reverse, bool accum>
__global__ void kernel_scan_parallel(Op op,
                                     const typename Op::IndexT size_outer,
                                     const typename Op::IndexT size_scan) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

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

template <class Op>
void scan_parallel(const Context &ctx, Op op, const ScanSetup &setup) {
  using IndexT = typename Op::IndexT;

  constexpr IndexT block_dim_x = NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK;
  constexpr IndexT block_dim_y = NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER;

  auto kernel =
      kernel_scan_parallel<Op, block_dim_x, block_dim_y, false /* exclusive */,
                           false /* reverse */, false /* accum */>;

  if (setup.exclusive) {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  true, true, true>
                           : kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  true, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  true, false, true>
                           : kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  true, false, false>;
    }
  } else {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  false, true, true>
                           : kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  false, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  false, false, true>
                           : kernel_scan_parallel<Op, block_dim_x, block_dim_y,
                                                  false, false, false>;
    }
  }

  const dim3 grid_dim(
      std::min(NBLA_CEIL_SIZE_T_DIV((Size_t)setup.size_outer, block_dim_x),
               NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(block_dim_x, block_dim_y);

  kernel<<<grid_dim, block_dim>>>(op, setup.size_outer, setup.size_scan);
  NBLA_CUDA_KERNEL_CHECK();
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

template <class Op>
void scan_parallel_inter_block(const Context &ctx, Op op,
                               const ScanSetup &setup) {
  using Tcu = typename Op::Tcu;
  using StorageT = typename Op::StorageT;
  using IndexT = typename Op::IndexT;

  // Step 1
  constexpr IndexT block_dim_x = NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK;
  constexpr IndexT block_dim_y = NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER;
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
                                             false /* exclusive */,
                                             false /* reverse */>;
    if (setup.exclusive) {
      kernel =
          setup.reverse
              ? kernel_scan_parallel_inter_block_pre<Op, block_dim_x,
                                                     block_dim_y, true, true>
              : kernel_scan_parallel_inter_block_pre<Op, block_dim_x,
                                                     block_dim_y, true, false>;
    } else {
      kernel =
          setup.reverse
              ? kernel_scan_parallel_inter_block_pre<Op, block_dim_x,
                                                     block_dim_y, false, true>
              : kernel_scan_parallel_inter_block_pre<Op, block_dim_x,
                                                     block_dim_y, false, false>;
    }

    kernel<<<grid_dim, block_dim>>>(op_pre, setup.size_outer, setup.size_scan,
                                    size_scan_buf);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Step 2
  Variable mid_out(pre_buf.shape());
  Op op_mid(pre_buf.get_data_pointer<Tcu>(ctx),
            mid_out.cast_data_and_get_pointer<Tcu>(ctx, true));

  ScanSetup setup_mid;
  setup_mid(mid_out.shape(), 1 /* axis */, true /* exclusive */, setup.reverse,
            false /* accum */);
  scan(ctx, op_mid, setup_mid);

  // Step 3
  {
    Op op_post(pre_out.get_data_pointer<Tcu>(ctx), op.output_);
    op_post.buf = mid_out.cast_data_and_get_pointer<StorageT>(ctx, false);

    auto kernel =
        setup.accum ? kernel_scan_parallel_inter_block_post<Op, block_dim_x,
                                                            block_dim_y, true>
                    : kernel_scan_parallel_inter_block_post<Op, block_dim_x,
                                                            block_dim_y, false>;

    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan,
                                    size_scan_buf);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

template <class Op>
void scan(const Context &ctx, Op op, const ScanSetup &setup) {
  if (setup.size_inner == 1) {
    if (setup.size_scan < 128) {
      scan_parallel(ctx, op, setup);
    } else {
      scan_parallel_inter_block(ctx, op, setup);
    }
  } else {
    if (setup.size_scan < 128) {
      scan_naive(ctx, op, setup);
    } else {
      scan_naive_inter_block(ctx, op, setup);
    }
  }
}
}

#endif
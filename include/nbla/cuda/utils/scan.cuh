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
          op.store<accum>(idx, op.input[idx] + buf_val);
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
  op_post.buf = v_mid_out.cast_data_and_get_pointer<Tcu>(ctx, true);
  {
    auto kernel = setup.accum ? kernel_scan_naive_inter_block_post<Op, true>
                              : kernel_scan_naive_inter_block_post<Op, false>;
    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan,
                                    setup.size_inner, size_scan_buf);
  }
}

constexpr Size_t NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK = 16;
constexpr Size_t NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER = 32;

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

template <class Op, bool exclusive, bool reverse, bool accum>
__global__ void kernel_scan_parallel(Op op,
                                     const typename Op::IndexT size_outer,
                                     const typename Op::IndexT size_scan) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  for (IndexT i0 =
           blockIdx.x * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER + threadIdx.y;
       i0 < size_outer;
       i0 += gridDim.x * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER) {
    StorageT last_val = op.init();
    for (IndexT k = 0;
         k < NBLA_CEIL_SIZE_T_DIV(
                 size_scan, NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2);
         k++) {
      IndexT i1_base =
          (reverse ? (NBLA_CEIL_SIZE_T_DIV(
                          size_scan,
                          NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2) -
                      k - 1)
                   : k) *
          NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2;
      last_val =
          block_scan_parallel<Op, StorageT, exclusive, reverse, accum,
                              NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK>(
              op, i0, i1_base, size_scan, last_val);
    }
  }
}

template <class Op>
void scan_parallel(const Context &ctx, Op op, const ScanSetup &setup) {
  auto kernel = kernel_scan_parallel<Op, false /* exclusive */,
                                     false /* reverse */, false /* accum */>;

  if (setup.exclusive) {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_parallel<Op, true, true, true>
                           : kernel_scan_parallel<Op, true, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_parallel<Op, true, false, true>
                           : kernel_scan_parallel<Op, true, false, false>;
    }
  } else {
    if (setup.reverse) {
      kernel = setup.accum ? kernel_scan_parallel<Op, false, true, true>
                           : kernel_scan_parallel<Op, false, true, false>;
    } else {
      kernel = setup.accum ? kernel_scan_parallel<Op, false, false, true>
                           : kernel_scan_parallel<Op, false, false, false>;
    }
  }

  const dim3 grid_dim(std::min(
      NBLA_CEIL_SIZE_T_DIV(setup.size_outer,
                           NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK),
      NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK,
                       NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER);

  kernel<<<grid_dim, block_dim>>>(op, setup.size_outer, setup.size_scan);
  NBLA_CUDA_KERNEL_CHECK();
}

template <class Op, bool reverse>
__global__ void
kernel_scan_parallel_inter_block_pre(Op op,
                                     const typename Op::IndexT size_outer,
                                     const typename Op::IndexT size_scan) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Grid-stride loop for outer axis
  for (IndexT i0 =
           blockIdx.y * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER + threadIdx.y;
       i0 < size_outer;
       i0 += gridDim.y * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER) {
    // Grid-stride loop for scan axis
    for (IndexT i1_base =
             blockIdx.x * (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2);
         i1_base < size_scan;
         i1_base +=
         gridDim.x * (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2)) {
      const auto last_val =
          block_scan_parallel<Op, StorageT, false /* exclusive */, reverse,
                              false /* accum */,
                              NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK>(
              op, i0, i1_base, size_scan, op.init());
      const IndexT buf_idx =
          i0 * NBLA_CEIL_SIZE_T_DIV(
                   size_scan, NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2) +
          (i1_base / (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2));
      if (threadIdx.x == 0) {
        op.intermediate_store(buf_idx, last_val);
      }
    }
  }
}

template <class Op, bool exclusive, bool reverse, bool accum>
__global__ void
kernel_scan_parallel_inter_block_post(Op op,
                                      const typename Op::IndexT size_outer,
                                      const typename Op::IndexT size_scan) {
  using IndexT = typename Op::IndexT;
  using StorageT = typename Op::StorageT;

  // Grid-stride loop for outer axis
  for (IndexT i0 =
           blockIdx.y * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER + threadIdx.y;
       i0 < size_outer;
       i0 += gridDim.y * NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER) {
    // Grid-stride loop for scan axis
    for (IndexT i1_base =
             blockIdx.x * (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2);
         i1_base < size_scan;
         i1_base +=
         gridDim.x * (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2)) {
      const IndexT buf_idx =
          i0 * NBLA_CEIL_SIZE_T_DIV(
                   size_scan, NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2) +
          (i1_base / (NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2));

      const IndexT i1 = i1_base + threadIdx.x;
      const IndexT num_threads = NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK;
      if (exclusive) {
        if (reverse) {
          if (i1 < size_scan) {
            const IndexT idx = i0 * size_scan + i1;
            if (i1 > 0) {
              op.store<accum>(idx - 1, op.input[idx] + op.buf[buf_idx]);
            }
            if (i1 == size_scan - 1) {
              op.store<accum>(idx, op.init());
            }
          }
          if (i1 + num_threads < size_scan) {
            const IndexT idx = i0 * size_scan + (i1 + num_threads);
            static_assert(num_threads > 0);
            op.store<accum>(idx - 1, op.input[idx] + op.buf[buf_idx]);
            if (i1 + num_threads == size_scan - 1) {
              op.store<accum>(idx, op.init());
            }
          }
        } else {
          if (i1 < size_scan) {
            const IndexT idx = i0 * size_scan + i1;
            if (i1 + 1 < size_scan) {
              op.store<accum>(idx + 1, op.input[idx] + op.buf[buf_idx]);
            }
            if (i1 == 0) {
              op.store<accum>(idx, op.init());
            }
          }
          if (i1 + num_threads < size_scan) {
            const IndexT idx = i0 * size_scan + (i1 + num_threads);
            if (i1 + num_threads + 1 < size_scan) {
              op.store<accum>(idx + 1, op.input[idx] + op.buf[buf_idx]);
            }
          }
        }
      } else {
        if (i1 < size_scan) {
          const IndexT idx = i0 * size_scan + i1;
          op.store<accum>(idx, op.input[idx] + op.buf[buf_idx]);
        }
        if (i1 + num_threads < size_scan) {
          const IndexT idx = i0 * size_scan + (i1 + num_threads);
          op.store<accum>(idx, op.input[idx] + op.buf[buf_idx]);
        }
      }
    }
  }
}

template <class Op>
void scan_parallel_inter_block(const Context &ctx, Op op,
                               const ScanSetup &setup) {
  using Tcu = typename Op::Tcu;
  using StorageT = typename Op::StorageT;

  // TODO: Step 1
  Variable pre_buf(
      {setup.size_outer,
       NBLA_CEIL_SIZE_T_DIV(setup.size_scan,
                            NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2)});
  Variable pre_out({setup.size_outer, setup.size_scan});
  Op op_pre(op.input, pre_out.cast_data_and_get_pointer<Tcu>(ctx, true));
  op_pre.buf = pre_buf.cast_data_and_get_pointer<StorageT>(ctx, true);

  auto kernel = setup.reverse ? kernel_scan_parallel_inter_block_pre<Op, true>
                              : kernel_scan_parallel_inter_block_pre<Op, false>;

  const dim3 grid_dim(
      std::min(
          NBLA_CEIL_SIZE_T_DIV(setup.size_scan,
                               NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK * 2),
          NBLA_CUDA_SCAN_MAX_BLOCKS),
      std::min(NBLA_CEIL_SIZE_T_DIV(setup.size_outer,
                                    NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER),
               NBLA_CUDA_SCAN_MAX_BLOCKS));
  const dim3 block_dim(NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_PER_BLOCK,
                       NBLA_CUDA_PREFIX_SCAN_NUM_THREADS_OUTER);

  kernel<<<grid_dim, block_dim>>>(op_pre, setup.size_outer, setup.size_scan);
  NBLA_CUDA_KERNEL_CHECK();

  // TODO: Step 2
  Variable mid_out(pre_buf.shape());
  Op op_mid(pre_buf.get_data_pointer<Tcu>(ctx),
            mid_out.cast_data_and_get_pointer<Tcu>(ctx, true));

  ScanSetup setup_mid;
  setup_mid(mid_out.shape(), 1 /* axis */, true /* exclusive */, setup.reverse,
            false /* accum */);
  // scan_parallel(ctx, op_mid, setup_mid);
  scan(ctx, op_mid, setup_mid);

  // TODO: Step 3
  {
    Op op_post(pre_out.get_data_pointer<Tcu>(ctx), op.output_);
    op_post.buf = mid_out.cast_data_and_get_pointer<StorageT>(ctx, false);

    auto kernel = kernel_scan_parallel_inter_block_post<
        Op, false /* exclusive */, false /* reverse */, false /* accum */>;

    if (setup.exclusive) {
      if (setup.reverse) {
        kernel =
            setup.accum
                ? kernel_scan_parallel_inter_block_post<Op, true, true, true>
                : kernel_scan_parallel_inter_block_post<Op, true, true, false>;
      } else {
        kernel =
            setup.accum
                ? kernel_scan_parallel_inter_block_post<Op, true, false, true>
                : kernel_scan_parallel_inter_block_post<Op, true, false, false>;
      }
    } else {
      if (setup.reverse) {
        kernel =
            setup.accum
                ? kernel_scan_parallel_inter_block_post<Op, false, true, true>
                : kernel_scan_parallel_inter_block_post<Op, false, true, false>;
      } else {
        kernel =
            setup.accum
                ? kernel_scan_parallel_inter_block_post<Op, false, false, true>
                : kernel_scan_parallel_inter_block_post<Op, false, false,
                                                        false>;
      }
    }

    kernel<<<grid_dim, block_dim>>>(op_post, setup.size_outer, setup.size_scan);
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
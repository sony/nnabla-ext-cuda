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

// All of the following CUDA kernels and constants are copied from PyTorch (link
// below) and modified/optimized for NNabla.
// https://github.com/pytorch/pytorch/blob/32b37ba2462d9d87337a4fe332f95524a4c49777/aten/src/ATen/native/cuda/Normalization.cuh

#include <nbla/cuda/common.hpp>

namespace nbla {

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int SYNC_BN_MAX_BLOCK_SIZE = 256;
#else
constexpr int SYNC_BN_MAX_BLOCK_SIZE = 512;
#endif

constexpr unsigned SYNC_BN_MAX_GRID_SIZE = 65535u;

constexpr int SYNC_BN_ELEMENTS_PER_ITER =
    4; // enables concurrency within each thread to hide latency
constexpr int SYNC_BN_ELEMENTS_PER_THREAD = 16;
constexpr int SYNC_BN_OPTIMAL_TILE_W = 32;
constexpr int SYNC_BN_MAX_H_BLOCK = 128;

// Number of threads in a block given an input size up to SYNC_BN_MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = {16, 32, 64, 128, SYNC_BN_MAX_BLOCK_SIZE};
#else
  int threadSizes[5] = {32, 64, 128, 256, SYNC_BN_MAX_BLOCK_SIZE};
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return SYNC_BN_MAX_BLOCK_SIZE;
}

// returns 2**floor(log2(n))
static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) { return 31 - __clz(val); }

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask,
                                           int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

// Sum across all threads within a warp
template <typename T> static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(CUDA_WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, CUDA_WARP_SIZE);
  }
  return val;
}

struct InvStd {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / sqrt(var + epsilon);
    }
    return invstd;
  }
};

struct Var {
  template <typename T>
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

__host__ void flexible_launch_configs(const int reduction, const int stride,
                                      dim3 &block, dim3 &grid,
                                      const bool coop_flag = false) {
  int block_x = std::min(lastPow2(stride), SYNC_BN_OPTIMAL_TILE_W);
  int block_y = std::min(
      lastPow2(NBLA_CEIL_INT_DIV(reduction, SYNC_BN_ELEMENTS_PER_THREAD)),
      SYNC_BN_MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != SYNC_BN_MAX_BLOCK_SIZE) {
    block_x = std::min(lastPow2(stride), SYNC_BN_MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = NBLA_CEIL_INT_DIV(stride, block_x);
  int grid_y = std::min(
      NBLA_CEIL_INT_DIV(reduction, block_y * SYNC_BN_ELEMENTS_PER_THREAD),
      SYNC_BN_MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template <typename VarTransform, typename input_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const input_scalar_t *input, const stat_accscalar_t epsilon,
    stat_accscalar_t *save_mean, stat_accscalar_t *save_transformed_var,
    const Size_t size0, const Size_t size1, const Size_t size2) {

  __shared__ int shared_n[2 * 2 * CUDA_WARP_SIZE + CUDA_WARP_SIZE];

  int plane = blockIdx.x;
  Size_t N = size0 * size2;
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across
  // the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a
  // description.
  stat_accscalar_t *shared_avg_var =
      (stat_accscalar_t *)&shared_n[CUDA_WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (index_t batch = threadIdx.y; batch < size0; batch += blockDim.y) {
    // Unroll and prefetch for latency hiding optimization
    constexpr int B = 4;
    stat_accscalar_t reg[B];
    for (index_t x = threadIdx.x; x < size2; x += blockDim.x * B) {
      const int base_idx = (batch * size1 + plane) * size2 + x;
      if (x + blockDim.x * (B - 1) < size2) {
// No need to consider boundary condition inside the loop.

// Prefetch
#pragma unroll
        for (int k = 0; k < B; k++) {
          const int idx = base_idx + blockDim.x * k;
          reg[k] = input[idx];
        }
// Calculate mean and variance
#pragma unroll
        for (int k = 0; k < B; k++) {
          stat_accscalar_t d1 = reg[k] - avg;
          n++;
          avg += d1 / n;
          var_n += d1 * (reg[k] - avg);
        }
      } else {
// Prefetch
#pragma unroll
        for (int k = 0; k < B; k++) {
          const int idx = base_idx + blockDim.x * k;
          if (x + blockDim.x * k < size2) {
            reg[k] = input[idx];
          }
        }
// Calculate mean and variance
#pragma unroll
        for (int k = 0; k < B; k++) {
          if (x + blockDim.x * k < size2) {
            stat_accscalar_t d1 = reg[k] - avg;
            n++;
            avg += d1 / n;
            var_n += d1 * (reg[k] - avg);
          }
        }
      }
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(CUDA_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, CUDA_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, CUDA_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, CUDA_WARP_SIZE) +
             (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most CUDA_WARP_SIZE items left because
  // there are at most CUDA_WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % CUDA_WARP_SIZE == 0) {
    shared_n[tid / CUDA_WARP_SIZE] = n;
    shared_avg_var[tid / CUDA_WARP_SIZE * 2] = avg;
    shared_avg_var[tid / CUDA_WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < CUDA_WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / CUDA_WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y / CUDA_WARP_SIZE
               ? shared_avg_var[2 * tid]
               : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y / CUDA_WARP_SIZE
                 ? shared_avg_var[2 * tid + 1]
                 : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(CUDA_WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, CUDA_WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, CUDA_WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n + o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, CUDA_WARP_SIZE) +
             (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var != NULL) {
      save_transformed_var[plane] = VarTransform{}(var_n / N, epsilon);
    }
  }
}

template <typename T, typename C>
__device__ __forceinline__ void
welford_merge_element(C &count, T &mean, T &m2n, const C &count_new,
                      const T &mean_new, const T &m2n_new) {
  T factor = T(1.0) / ::max(1, (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

// merge mean/m2n among threadIdx.y within block
template <typename T, typename C>
__device__ __forceinline__ void
welford_merge_block_vertical(C &count, T &mean, T &m2n, C *shmem_count,
                             T *shmem_mean, T *shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

// welford kernel for c last tensor calculating
// mean/biased_variance/unbiased_variance
// original apex name: welford_kernel_c_last
template <typename VarTransform, typename scalar_t, typename accscalar_t,
          typename index_t, int PARALLEL_LOADS>
__global__ void batch_norm_collect_statistics_channels_last_kernel(
    const scalar_t *__restrict__ input, accscalar_t *__restrict__ out_mean,
    accscalar_t *__restrict__ out_invstd, volatile accscalar_t *staging_data,
    int *semaphores, const int reduction_size, const int stride,
    accscalar_t epsilon) {
  // hide latency with concurrency
  accscalar_t x_mean[PARALLEL_LOADS];
  accscalar_t m_2_n[PARALLEL_LOADS];
  int count[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = accscalar_t(0);
    m_2_n[i] = accscalar_t(0);
    count[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  index_t inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  index_t m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  index_t c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  index_t loop_count =
      1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  index_t address_base = m_offset * stride + c_offset;
  index_t address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_math[PARALLEL_LOADS];
    accscalar_t x_count_inv[PARALLEL_LOADS];
    accscalar_t is_valid[PARALLEL_LOADS];

// load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input[address_base];
        count[j]++;
        x_count_inv[j] = accscalar_t(1) / count[j];
        is_valid[j] = accscalar_t(1);
      } else {
        x_math[j] = accscalar_t(0);
        x_count_inv[j] = accscalar_t(0);
        is_valid[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

// calculate mean/m2n with welford
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      accscalar_t delta0 = x_math[j] - x_mean[j];
      x_mean[j] += delta0 * x_count_inv[j];
      accscalar_t delta1 = x_math[j] - x_mean[j];
      m_2_n[j] += delta0 * delta1 * is_valid[j];
    }
  }

// thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j], x_mean[j],
                          m_2_n[j]);
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  auto m2_th = m_2_n[0];
  auto count_th = count[0];

  // block-wise reduction with shared memory (since reduction cannot be done
  // within a warp)
  static __shared__ accscalar_t shmem_mean[SYNC_BN_MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_m2n[SYNC_BN_MAX_BLOCK_SIZE];
  static __shared__ int shmem_count[SYNC_BN_MAX_BLOCK_SIZE];

  welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count,
                               shmem_mean, shmem_m2n);

  if (gridDim.y > 1) {
    volatile accscalar_t *staging_mean = staging_data;
    volatile accscalar_t *staging_m2n = &staging_data[stride * gridDim.y];
    volatile int *staging_count =
        reinterpret_cast<volatile int *>(&staging_m2n[stride * gridDim.y]);

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
      staging_m2n[address_base] = m2_th;
      staging_count[address_base] = count_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y - 1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      count_th = 0;
      mean_th = accscalar_t(0.0);
      m2_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        int count_new = c_offset < stride ? staging_count[address_base] : 0;
        accscalar_t mean_new =
            c_offset < stride ? staging_mean[address_base] : accscalar_t(0.0);
        accscalar_t m2n_new =
            c_offset < stride ? staging_m2n[address_base] : accscalar_t(0.0);

        welford_merge_element(count_th, mean_th, m2_th, count_new, mean_new,
                              m2n_new);
      }

      welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count,
                                   shmem_mean, shmem_m2n);
      if (threadIdx.y == 0 && c_offset < stride) {
        out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
        out_invstd[c_offset] = VarTransform{}(m2_th / count_th, epsilon);
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      out_mean[c_offset] = static_cast<accscalar_t>(mean_th);
      out_invstd[c_offset] = VarTransform{}(m2_th / count_th, epsilon);
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void batch_norm_reduce_statistics_kernel(
    const accscalar_t *vec_mean, const accscalar_t *vec_invstd,
    accscalar_t *mean, accscalar_t *var, scalar_t *running_mean,
    scalar_t *running_var, const accscalar_t epsilon,
    const accscalar_t decay_rate, const scalar_t *counts,
    const int feature_size, const int world_size) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // first the reductions each thread does separately
  for (int i = bid * blockDim.x + tid; i < feature_size;
       i += gridDim.x * blockDim.x) {
    accscalar_t avg = 0;
    accscalar_t var_n = 0;
    index_t n = 0;
    for (int j = 0; j < world_size; j++) {
      scalar_t count = counts[j];
      accscalar_t m = vec_mean[j * feature_size + i];
      accscalar_t v = accscalar_t(1.0) / (vec_invstd[j * feature_size + i]);
      v = (v * v - epsilon) * count;
      accscalar_t factor = 1.0 / ((accscalar_t)n + count);
      var_n += v + (avg - m) * (avg - m) * n * count * factor;
      avg = n * factor * avg + count * factor * m;
      n += (index_t)count;
    }
    mean[i] = avg;
    var[i] = var_n / n;
    if (running_mean != NULL) {
      running_mean[i] = static_cast<scalar_t>(decay_rate * running_mean[i] +
                                              (1.0 - decay_rate) * avg);
    }
    accscalar_t unbiasedVar = var_n / (n - 1);
    if (running_var != NULL) {
      running_var[i] = static_cast<scalar_t>(decay_rate * running_var[i] +
                                             (1.0 - decay_rate) * unbiasedVar);
    }
  }
}

// template <typename input_scalar_t, typename stat_scalar_t, typename
// stat_accscalar_t, bool train, typename index_t>
template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const input_scalar_t *input, input_scalar_t *output,
    const stat_accscalar_t *mean_, const stat_accscalar_t *var,
    const stat_scalar_t *weight, const stat_scalar_t *bias,
    const stat_accscalar_t epsilon, const int size0, const int size1,
    const int size2) {

  index_t plane = blockIdx.x;

  if (plane >= size1) {
    return;
  }

  stat_accscalar_t gamma = weight ? static_cast<stat_accscalar_t>(weight[plane])
                                  : static_cast<stat_accscalar_t>(1);
  stat_accscalar_t beta = bias ? static_cast<stat_accscalar_t>(bias[plane])
                               : static_cast<stat_accscalar_t>(0);
  stat_accscalar_t mean = static_cast<stat_accscalar_t>(mean_[plane]);
  stat_accscalar_t invstd =
      static_cast<stat_accscalar_t>(1) /
      sqrt(static_cast<stat_accscalar_t>(var[plane]) + epsilon);

  index_t bs = size0;
  index_t fs = size2;

  index_t bstep = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs;
       batch += bstep) {
    auto o = &output[(batch * size1 + plane) * size2];
    auto i = &input[(batch * size1 + plane) * size2];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<input_scalar_t>(
          gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}

// elementwise BN kernel
// original apex name: batchnorm_forward_c_last_kernel
template <typename scalar_t, typename accscalar_t, typename layerscalar_t,
          typename index_t, int PARALLEL_LOADS>
__global__ void batch_norm_transform_input_channels_last_kernel(
    const scalar_t *__restrict__ input, const accscalar_t *__restrict__ mean,
    // const accscalar_t* __restrict__ inv_std,
    const accscalar_t *__restrict__ var,
    const layerscalar_t *__restrict__ weight, // sacle or gamma
    const layerscalar_t *__restrict__ shift,  // bias or beta
    scalar_t *__restrict__ out, const accscalar_t epsilon,
    const index_t reduction_size, const index_t stride) {
  // tensor dimension (m,c)
  // loop along m dimension
  index_t inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  index_t m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  index_t c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  auto m_c = mean[c_offset];
  auto inv_std_c = static_cast<accscalar_t>(1) / sqrt(var[c_offset] + epsilon);
  auto w_c = weight == nullptr ? accscalar_t(1.0)
                               : static_cast<accscalar_t>(weight[c_offset]);
  auto s_c = shift == nullptr ? accscalar_t(0.0)
                              : static_cast<accscalar_t>(shift[c_offset]);

  index_t loop_count =
      1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  index_t address_base = m_offset * stride + c_offset;
  index_t address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        auto tmp = w_c * (static_cast<accscalar_t>(input[address_base]) - m_c) *
                       inv_std_c +
                   s_c;
        out[address_base] = static_cast<scalar_t>(tmp);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

template <typename scalar_t, typename accscalar_t> struct GradOp {
  __device__ GradOp(accscalar_t m, const scalar_t *i, const scalar_t *g)
      : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ void operator()(accscalar_t *dy,
                                             accscalar_t *dy_xmu, int idx) {
    accscalar_t g = grad_output[idx];
    accscalar_t c = static_cast<accscalar_t>(input[idx]) - mean;
    *dy = g;
    *dy_xmu = g * c;
  }
  const accscalar_t mean;
  const scalar_t *input;
  const scalar_t *grad_output;
};

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffling reduction.
// First each warp (of CUDA_WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most CUDA_WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than CUDA_WARP_SIZE**2 threads.
template <typename Op, typename scalar_t, typename accscalar_t,
          typename index_t>
__device__ void reduce(Op op, scalar_t *sum_dy_o, accscalar_t *sum_dy_xmu_o,
                       const index_t plane, const Size_t size0,
                       const Size_t size1, const Size_t size2) {
  // first the reductions each thread does separately
  accscalar_t sum_dy = static_cast<accscalar_t>(0);
  accscalar_t sum_dy_xmu = static_cast<accscalar_t>(0);
  for (index_t batch = threadIdx.y; batch < size0; batch += blockDim.y) {
    // Prefetch and unrolling for latency hiding optimization.
    constexpr int B = 8;
    accscalar_t reg_dy[B];
    accscalar_t reg_dy_xmu[B];
    for (index_t x = threadIdx.x; x < size2; x += blockDim.x * B) {
      const index_t base_idx = (batch * size1 + plane) * size2 + x;
// Prefetch
#pragma unroll
      for (int k = 0; k < B; k++) {
        const index_t idx = base_idx + blockDim.x * k;
        if (x + blockDim.x * k < size2) {
          op(&reg_dy[k], &reg_dy_xmu[k], idx);
        }
      }
// Calculation
#pragma unroll
      for (int k = 0; k < B; k++) {
        if (x + blockDim.x * k < size2) {
          sum_dy += reg_dy[k];
          sum_dy_xmu += reg_dy_xmu[k];
        }
      }
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum_dy = warpSum(sum_dy);
  sum_dy_xmu = warpSum(sum_dy_xmu);

  // this writes each warps  item into shared memory
  // there are at most CUDA_WARP_SIZE items left because
  // there are at most CUDA_WARP_SIZE**2 threads at the beginning
  __shared__ accscalar_t shared_dy[CUDA_WARP_SIZE];
  __shared__ accscalar_t shared_dy_xmu[CUDA_WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % CUDA_WARP_SIZE == 0) {
    shared_dy[tid / CUDA_WARP_SIZE] = sum_dy;
    shared_dy_xmu[tid / CUDA_WARP_SIZE] = sum_dy_xmu;
  }
  if (tid >= blockDim.x * blockDim.y / CUDA_WARP_SIZE && tid < CUDA_WARP_SIZE) {
    // zero out the other entries in shared
    shared_dy[tid] = (scalar_t)0;
    shared_dy_xmu[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / CUDA_WARP_SIZE == 0) {
    sum_dy = warpSum(shared_dy[tid]);
    sum_dy_xmu = warpSum(shared_dy_xmu[tid]);
    if (tid == 0) {
      shared_dy[0] = sum_dy;
      shared_dy_xmu[0] = sum_dy_xmu;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  *sum_dy_o = shared_dy[0];
  *sum_dy_xmu_o = shared_dy_xmu[0];
}

template <typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_reduce_kernel(
    const input_scalar_t *input, const input_scalar_t *grad_output,
    const stat_accscalar_t *mean, const stat_accscalar_t *var,
    stat_accscalar_t *sum_dy_o, stat_accscalar_t *sum_dy_xmu_o,
    stat_scalar_t *grad_weight, stat_scalar_t *grad_bias,
    const stat_accscalar_t epsilon, const Size_t size0, const Size_t size1,
    const Size_t size2) {

  index_t plane = blockIdx.x;

  stat_accscalar_t r_mean = mean[plane];
  stat_accscalar_t factor =
      static_cast<stat_accscalar_t>(1) / sqrt(var[plane] + epsilon);

  GradOp<input_scalar_t, stat_accscalar_t> g(r_mean, input, grad_output);
  stat_scalar_t sum_dy;
  stat_accscalar_t sum_dy_xmu;

  reduce(g, &sum_dy, &sum_dy_xmu, plane, size0, size1, size2);

  if (threadIdx.x == 0) {
    if (grad_weight) {
      grad_weight[plane] = static_cast<stat_scalar_t>(sum_dy_xmu * factor);
    }
    if (grad_bias) {
      grad_bias[plane] = static_cast<stat_scalar_t>(sum_dy);
    }
    if (sum_dy_o) {
      sum_dy_o[plane] = static_cast<stat_accscalar_t>(sum_dy);
    }
    if (sum_dy_xmu_o) {
      sum_dy_xmu_o[plane] = static_cast<stat_accscalar_t>(sum_dy_xmu);
    }
  }
}

template <typename T>
__device__ __forceinline__ void
merge_block_vertical_backward(T &sum_dy, T &sum_dy_xmu, T *shmem_sum_dy,
                              T *shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];
    }
  }
}

// batchnorm backward kernel for c last tensor
// original apex name: reduce_bn_c_last_kernel
template <int PARALLEL_LOADS, typename scalar_t, typename accscalar_t,
          typename layerscalar_t, typename index_t>
__global__ void batch_norm_backward_reduce_channels_last_kernel(
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ grad_output,
    const accscalar_t *__restrict__ mean, const accscalar_t *__restrict__ var,
    accscalar_t *__restrict__ sum_dy_o, accscalar_t *__restrict__ sum_dy_xmu_o,
    layerscalar_t *__restrict__ grad_weight,
    layerscalar_t *__restrict__ grad_bias, volatile accscalar_t *staging_data,
    int *semaphores, const index_t reduction_size, const index_t stride,
    const accscalar_t epsilon) {

  // hide latency with concurrency
  accscalar_t sum_dy[PARALLEL_LOADS];
  accscalar_t sum_dy_xmu[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    sum_dy[i] = accscalar_t(0);
    sum_dy_xmu[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  index_t inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  index_t m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  index_t c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  index_t loop_count =
      1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  index_t address_base = m_offset * stride + c_offset;
  index_t address_increment = inner_loop_stride * stride;

  auto r_mean = mean[c_offset];
  // auto factor = inv_std[c_offset];
  auto factor = 1.0 / sqrt(var[c_offset] + epsilon);

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_input[PARALLEL_LOADS];
    accscalar_t x_grad_output[PARALLEL_LOADS];

// load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_input[j] = input[address_base];
        x_grad_output[j] = grad_output[address_base];
      } else {
        x_input[j] = accscalar_t(0);
        x_grad_output[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

// calculate sum_dy / sum_dy_xmu
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      sum_dy[j] += x_grad_output[j];
      sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
    }
  }

// thread reduction to accumulate sum_dy / sum_dy_xmu between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    sum_dy[0] += sum_dy[j];
    sum_dy_xmu[0] += sum_dy_xmu[j];
  }

  // release array of registers
  auto sum_dy_th = sum_dy[0];
  auto sum_dy_xmu_th = sum_dy_xmu[0];

  // block-wise reduction with shared memory (since reduction cannot be done
  // within a warp)
  static __shared__ accscalar_t shmem_sum_dy[SYNC_BN_MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_sum_dy_xmu[SYNC_BN_MAX_BLOCK_SIZE];

  merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy,
                                shmem_sum_dy_xmu);

  if (gridDim.y > 1) {
    volatile accscalar_t *staging_sum_dy = staging_data;
    volatile accscalar_t *staging_sum_dy_xmu =
        &staging_data[stride * gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_sum_dy[address_base] = sum_dy_th;
      staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y - 1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = accscalar_t(0.0);
      sum_dy_xmu_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        sum_dy_th += (c_offset < stride ? staging_sum_dy[address_base]
                                        : accscalar_t(0.0));
        sum_dy_xmu_th += (c_offset < stride ? staging_sum_dy_xmu[address_base]
                                            : accscalar_t(0.0));
      }

      merge_block_vertical_backward(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy,
                                    shmem_sum_dy_xmu);
      if (threadIdx.y == 0 && c_offset < stride) {
        if (grad_bias != nullptr) {
          grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
        }
        if (grad_weight != nullptr) {
          grad_weight[c_offset] =
              static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
        }
        sum_dy_o[c_offset] = sum_dy_th;
        sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      if (grad_bias != nullptr) {
        grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
      }
      if (grad_weight != nullptr) {
        grad_weight[c_offset] =
            static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
      }
      sum_dy_o[c_offset] = sum_dy_th;
      sum_dy_xmu_o[c_offset] = sum_dy_xmu_th;
    }
  }
}

template <bool accum, typename input_scalar_t, typename stat_scalar_t,
          typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_backward_elemt_kernel(
    const input_scalar_t *input, const input_scalar_t *grad_output,
    const stat_accscalar_t *mean, const stat_accscalar_t *var,
    const stat_accscalar_t *dmean, const stat_accscalar_t *dvar,
    const stat_scalar_t *weight, const stat_accscalar_t *sum_dy,
    const stat_accscalar_t *sum_dy_xmu, input_scalar_t *grad_input,
    const stat_accscalar_t epsilon, const int *__restrict__ numel,
    const int world_size, const index_t size0, const index_t size1,
    const index_t size2) {
  index_t plane = blockIdx.x;

  if (plane >= size1) {
    return;
  }

  // Inverted total reduction size.
  int64_t total_numel = 0;
  for (int i = 0; i < world_size; i++) {
    total_numel += numel[i];
  }
  const stat_accscalar_t norm_fct = static_cast<stat_accscalar_t>(1) /
                                    static_cast<stat_accscalar_t>(total_numel);

  // weight, mean and invstd
  const stat_accscalar_t w_c = weight[plane];
  const stat_accscalar_t m_c = mean[plane];
  const stat_accscalar_t invstd_c = 1.0 / sqrt(var[plane] + epsilon);

  // dmean and dvar
  stat_scalar_t dv = w_c * sum_dy_xmu[plane];
  dv = dv * (-0.5) * invstd_c * invstd_c * invstd_c + (dvar ? dvar[plane] : 0);
  stat_scalar_t dm = w_c * sum_dy[plane];
  dm = dm * -invstd_c + (dmean ? dmean[plane] : 0);

  // Pre-calculation of factors for optimization.
  const stat_scalar_t factor1 = w_c * invstd_c;
  const stat_scalar_t factor2 = dv * 2 * norm_fct;
  const stat_scalar_t factor3 = (dm - dv * 2 * m_c) * norm_fct;

  index_t bs = size0;
  index_t fs = size2;

  index_t bstep = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs;
       batch += bstep) {
    const int idx_offset = (batch * size1 + plane) * size2;
    auto *g_i = &grad_input[idx_offset];
    auto *g_o = &grad_output[idx_offset];
    auto *i = &input[idx_offset];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      const stat_scalar_t grad =
          g_o[feature] * factor1 + i[feature] * factor2 + factor3;
      if (accum) {
        g_i[feature] += grad;
      } else {
        g_i[feature] = grad;
      }
    }
  }
}

// elementwise BN kernel
// original apex name: batchnorm_backward_c_last_kernel
template <int PARALLEL_LOADS, bool accum, typename scalar_t,
          typename accscalar_t, typename layerscalar_t, typename index_t>
__global__ void batch_norm_backward_elemt_channels_last_kernel(
    const scalar_t *__restrict__ grad_output,
    const scalar_t *__restrict__ input, const accscalar_t *__restrict__ mean,
    // const accscalar_t* __restrict__ inv_std,
    const accscalar_t *__restrict__ var, const accscalar_t *__restrict__ dmean,
    const accscalar_t *__restrict__ dvar,
    const layerscalar_t *__restrict__ weight,
    const accscalar_t *__restrict__ sum_dy,
    const accscalar_t *__restrict__ sum_dy_xmu, const int *__restrict__ numel,
    scalar_t *__restrict__ grad_input, const int world_size,
    const index_t reduction_size, const index_t stride,
    const accscalar_t epsilon) {
  // tensor dimension (m,c)
  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  if (c_offset >= stride || m_offset >= reduction_size) {
    return;
  }

  // Inverted total reduction size.
  int64_t total_numel = 0;
  for (int i = 0; i < world_size; i++) {
    total_numel += numel[i];
  }
  auto norm_fct =
      static_cast<accscalar_t>(1) / static_cast<accscalar_t>(total_numel);

  // weight, mean and invstd
  const accscalar_t w_c = weight[c_offset];
  const accscalar_t m_c = mean[c_offset];
  const accscalar_t invstd_c = 1.0 / sqrt(var[c_offset] + epsilon);

  // dmean and dvar
  scalar_t dv = w_c * sum_dy_xmu[c_offset];
  dv = dv * (-0.5) * invstd_c * invstd_c * invstd_c +
       (dvar ? dvar[c_offset] : 0);
  scalar_t dm = w_c * sum_dy[c_offset];
  dm = dm * -invstd_c + (dmean ? dmean[c_offset] : 0);

  // Pre-calculation of factors for optimization.
  const scalar_t factor1 = w_c * invstd_c;
  const scalar_t factor2 = dv * 2 * norm_fct;
  const scalar_t factor3 = (dm - dv * 2 * m_c) * norm_fct;

  index_t loop_count =
      1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  index_t address_base = m_offset * stride + c_offset;
  index_t address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        const scalar_t grad = grad_output[address_base] * factor1 +
                              input[address_base] * factor2 + factor3;

        if (accum) {
          grad_input[address_base] += grad;
        } else {
          grad_input[address_base] = grad;
        }
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}
}

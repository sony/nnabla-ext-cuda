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
#include <nbla/cuda/function/group_normalization.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/cuda/utils/warp_reduce.cuh>

// For channel-last adaptor
#include <nbla/function/transpose.hpp>

namespace nbla {

// TODO:
inline void dump_buffer(const float *ptr, const int size, string message) {
  std::cout << message << ": ";
  for (int i = 0; i < size; i++) {
    std::cout << ptr[i] << ", ";
  }
  std::cout << std::endl;
}

inline void dump_data_buffer(Variable *var, const int size, string message) {
  cudaDeviceSynchronize();
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  const float *ptr = var->get_data_pointer<float>(cpu_ctx);
  dump_buffer(ptr, size, message);
}

inline void dump_grad_buffer(Variable *var, const int size, string message) {
  cudaDeviceSynchronize();
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  const float *ptr = var->get_grad_pointer<float>(cpu_ctx);
  dump_buffer(ptr, size, message);
}

template <typename T>
void GroupNormalizationCuda<T>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  GroupNormalization<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const auto x = inputs[0];
  const auto x_shape = x->shape();
  const auto ndim = x->ndim();
  const auto c = this->channel_axis_;
  channel_last_ = c == ndim - 1 && ndim != 2;
  channel_size_ = x_shape[c];

  if (channel_last_) {
    // Pre-transpose: [b, h, w, c] -> [b, c, h, w]
    vector<int> pre_transpose_shape;
    for (int i = 0; i < this->batch_axis_.size(); i++) {
      pre_transpose_shape.push_back(i);
    }
    pre_transpose_shape.push_back(c);
    for (int i = this->batch_axis_.size(); i < ndim - 1; i++) {
      pre_transpose_shape.push_back(i);
    }
    pre_transpose_ = create_Transpose(this->ctx_, pre_transpose_shape);
    pre_transpose_->setup({inputs[0]}, {&pre_adaptor_});

    // Post-transpose: [b, c, h, w] -> [b, h, w, c]
    vector<int> post_transpose_shape;
    for (int i = 0; i < this->batch_axis_.size(); i++) {
      post_transpose_shape.push_back(i);
    }
    for (int i = this->batch_axis_.size(); i < ndim - 1; i++) {
      post_transpose_shape.push_back(i + 1);
    }
    post_transpose_shape.push_back(this->batch_axis_.size());
    post_transpose_ = create_Transpose(this->ctx_, post_transpose_shape);
    post_adaptor_.reshape(pre_adaptor_.shape(), true);
    post_transpose_->setup({&post_adaptor_}, {outputs[0]});

    // Create stats shape
    const auto c_ = this->batch_axis_.size();
    Shape_t stats_shape;
    for (auto i = 0; i < c_; i++) {
      stats_shape.push_back(pre_adaptor_.shape()[i]);
    }
    stats_shape.push_back(this->num_groups_);
    stats_shape.push_back(pre_adaptor_.shape()[c_] / this->num_groups_);
    for (auto i = c + 1; i < ndim; i++) {
      stats_shape.push_back(pre_adaptor_.shape()[i]);
    }

    batch_size_ = pre_adaptor_.size() / pre_adaptor_.size(c_);
    reduce_size_ =
        pre_adaptor_.size(c_ + 1) * (channel_size_ / this->num_groups_);
    outer_size_ = pre_adaptor_.size() / reduce_size_;

    // TODO: output stat
    a_.reshape({batch_size_ * channel_size_}, true);
    b_.reshape({batch_size_ * channel_size_}, true);
    var_.reshape(stats_shape, true);
    mean_.reshape(stats_shape, true);

    sum_dy_.reshape({batch_size_ * channel_size_}, true);
    sum_dyx_.reshape({batch_size_ * channel_size_}, true);
    gamma_invstd_.reshape({batch_size_ * channel_size_}, true);
    factor1_.reshape({batch_size_ * this->num_groups_}, true);
    factor2_.reshape({batch_size_ * this->num_groups_}, true);
  } else {
    batch_size_ = x->size() / x->size(c);
    reduce_size_ = x->size(c + 1) * (channel_size_ / this->num_groups_);
    outer_size_ = x->size() / reduce_size_;

    // Create stats shape
    Shape_t stats_shape;
    for (auto i = 0; i < c; i++) {
      stats_shape.push_back(x_shape[i]);
    }
    stats_shape.push_back(this->num_groups_);
    stats_shape.push_back(x_shape[c] / this->num_groups_);
    for (auto i = c + 1; i < ndim; i++) {
      stats_shape.push_back(x_shape[i]);
    }

    // TODO: output stat
    a_.reshape({batch_size_ * channel_size_}, true);
    b_.reshape({batch_size_ * channel_size_}, true);
    var_.reshape(stats_shape, true);
    mean_.reshape(stats_shape, true);

    sum_dy_.reshape({batch_size_ * channel_size_}, true);
    sum_dyx_.reshape({batch_size_ * channel_size_}, true);
    gamma_invstd_.reshape({batch_size_ * channel_size_}, true);
    factor1_.reshape({batch_size_ * this->num_groups_}, true);
    factor2_.reshape({batch_size_ * this->num_groups_}, true);
  }
}

constexpr int GROUP_NORM_ELEMENTWISE_UNROLL_SIZE = 4;

template <typename index_t> struct WelfordType {
  float mean;
  float m2;
  index_t n;
};

template <typename index_t> class WelfordOp {
public:
  using storage_type = WelfordType<index_t>;

  __forceinline__ __device__ void init(storage_type &thread_data) {
    thread_data.mean = 0.0f;
    thread_data.m2 = 0.0f;
    thread_data.n = 0;
  }

  __forceinline__ __device__ void reduce(storage_type &to,
                                         const storage_type &from) {
    if (to.n == 0) {
      to = from;
      return;
    }
    if (from.n == 0) {
      return;
    }
    const index_t next_n = to.n + from.n;
    const float next_n_inv = 1.0f / next_n;
    const float dmean = from.mean - to.mean;
    const float next_mean = to.mean + dmean * from.n * next_n_inv;
    const float next_m2 =
        to.m2 + from.m2 + dmean * dmean * to.n * from.n * next_n_inv;
    to.mean = next_mean;
    to.m2 = next_m2;
    to.n = next_n;
  }

  __forceinline__ __device__ void reduce_one(storage_type &to, const float x) {
    const float dmean = x - to.mean;
    const float next_mean = to.mean + dmean / (to.n + 1);
    const float next_m2 = to.m2 + dmean * (x - next_mean);
    to.mean = next_mean;
    to.m2 = next_m2;
    to.n = to.n + 1;
  }
};

// Explicit template instantiation of shuffle_down for WelfordType
template <>
__forceinline__ __device__ WelfordType<int>
shuffle_down(WelfordType<int> val, int offset, int width) {
  WelfordType<int> buff;
  buff.mean = shuffle_down(val.mean, offset, width);
  buff.m2 = shuffle_down(val.m2, offset, width);
  buff.n = shuffle_down(val.n, offset, width);
  return buff;
}

template <>
__forceinline__ __device__ WelfordType<Size_t>
shuffle_down(WelfordType<Size_t> val, int offset, int width) {
  WelfordType<Size_t> buff;
  buff.mean = shuffle_down(val.mean, offset, width);
  buff.m2 = shuffle_down(val.m2, offset, width);
  buff.n = shuffle_down(val.n, offset, width);
  return buff;
}

template <typename T, typename index_t>
__global__ void group_norm_forward_mean_var(const index_t reduce_size,
                                            const T *x, T *mean, T *var) {

  const index_t bidx = blockIdx.x;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  WelfordOp<index_t> op;

  // Load and reduce
  WelfordType<index_t> val;
  op.init(val);
  for (index_t i = tidx; i < reduce_size; i += bdimx) {
    const index_t idx = bidx * reduce_size + i;
    op.reduce_one(val, float(x[idx]));
  }

  if (bdimx <= CUDA_WARP_SIZE) {
    // Warp reduce
    warpReduce(op, val);
  } else {
    // Block reduce
    blockReduce(op, val);
  }

  // Store
  if (threadIdx.x == 0) {
    mean[bidx] = val.mean;
    var[bidx] = val.m2 / reduce_size;
  }
}

template <typename T, typename index_t>
__global__ void group_norm_forward_normalization_factor(
    const index_t batch_size, const index_t channel_size, const int num_groups,
    const T *mean, const T *var, const T *beta, const T *gamma, T *a, T *b,
    const float eps) {
  // Calculate `a` and `b` of simplified normalization formula
  // as `y = a * x + b`.
  // Original formula is `y = gamma * (x - mean) / sqrt(var) + beta`.
  // Thus `a = gamma / sqrt(var), b = beta - gamma * mean / sqrt(var).`

  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size * channel_size) {
    const index_t stats_idx = idx / (channel_size / num_groups);
    const index_t param_idx = idx % channel_size;
    const T scale = gamma ? gamma[param_idx] : (T)1.0f;
    const T bias = beta ? beta[param_idx] : (T)0.0f;

    const T invstd = rsqrt(var[stats_idx] + eps);
    const T scale_invstd = scale * invstd;
    a[idx] = scale_invstd;
    b[idx] = bias - mean[stats_idx] * scale_invstd;
  }
}

template <typename T, typename index_t>
__global__ void
group_norm_forward_normalization(const index_t size, const index_t spatial_size,
                                 const T *x, const T *a, const T *b, T *y) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t ab_idx = idx / spatial_size;
      y[idx] = a[ab_idx] * x[idx] + b[ab_idx];
    }
  }
}

template <typename T, typename index_t>
__global__ void group_norm_backward_sum_dy_dyx(const index_t spatial_size,
                                               const T *x, const T *dy,
                                               T *sum_dy_out, T *sum_dyx_out) {

  const index_t bidx = blockIdx.x;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  // Load and reduce
  float sum_dy = 0.0f;
  float sum_dyx = 0.0f;
  for (index_t i = tidx; i < spatial_size; i += bdimx) {
    const index_t idx = bidx * spatial_size + i;
    sum_dy += static_cast<float>(dy[idx]);
    sum_dyx += static_cast<float>(dy[idx]) * static_cast<float>(x[idx]);
  }

  if (bdimx <= CUDA_WARP_SIZE) {
    sum_dy = warpReduceSum(sum_dy);
    sum_dyx = warpReduceSum(sum_dyx);
  } else {
    sum_dy = blockReduceSum(sum_dy);
    sum_dyx = blockReduceSum(sum_dyx);
  }

  // Store
  if (threadIdx.x == 0) {
    sum_dy_out[bidx] = sum_dy;
    sum_dyx_out[bidx] = sum_dyx;
  }
}

template <typename T, typename index_t>
__global__ void group_norm_backward_gamma_invstd(
    const index_t size, const index_t channel_size, const int num_groups,
    const T *gamma, const T *var, T *gamma_invstd, const float eps) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t stats_idx = idx / (channel_size / num_groups);
      const index_t param_idx = idx % channel_size;
      const T scale = gamma ? gamma[param_idx] : (T)1.0f;
      gamma_invstd[idx] = scale * rsqrt(var[stats_idx] + eps);
    }
  }
}

template <typename T, typename index_t>
__global__ void group_norm_backward_dx_factor(
    const index_t channel_size, const index_t spatial_size,
    const int num_groups, const T *mean, const T *var, const T *dmean,
    const T *dvar, const T *gamma, const T *sum_dy, const T *sum_dyx,
    T *factor1, T *factor2, const float eps) {
  const index_t bidx = blockIdx.x;
  const index_t bidy = blockIdx.y;
  const index_t tidx = threadIdx.x;
  const index_t bdimx = blockDim.x;

  const index_t chunk_size = channel_size / num_groups;

  // Load and reduce
  float sum_dy_gamma = 0.0f;
  float sum_dyx_gamma = 0.0f;
  for (index_t i = tidx; i < chunk_size; i += bdimx) {
    const index_t idx = (bidx * num_groups + bidy) * chunk_size + i;
    const index_t param_idx = bidy * chunk_size + i;
    const T scale = gamma ? gamma[param_idx] : (T)1.0f;
    sum_dy_gamma += static_cast<float>(sum_dy[idx] * scale);
    sum_dyx_gamma += static_cast<float>(sum_dyx[idx] * scale);
  }

  if (bdimx <= CUDA_WARP_SIZE) {
    sum_dy_gamma = warpReduceSum(sum_dy_gamma);
    sum_dyx_gamma = warpReduceSum(sum_dyx_gamma);
  } else {
    sum_dy_gamma = blockReduceSum(sum_dy_gamma);
    sum_dyx_gamma = blockReduceSum(sum_dyx_gamma);
  }

  // Store
  if (threadIdx.x == 0) {
    const float inv_reduce_size = 1.0f / (chunk_size * spatial_size);
    const index_t stats_idx = bidx * num_groups + bidy;
    const float invstd = rsqrt(var[stats_idx] + eps);
    // TODO:
    const float tmp = (sum_dy_gamma * mean[stats_idx] - sum_dyx_gamma) *
                          invstd * invstd * invstd * inv_reduce_size +
                      (dvar ? 2.0f * dvar[stats_idx] * inv_reduce_size : 0.0f);
    factor1[stats_idx] = tmp;
    factor2[stats_idx] = -tmp * mean[stats_idx] -
                         sum_dy_gamma * invstd * inv_reduce_size +
                         (dmean ? dmean[stats_idx] * inv_reduce_size : 0.0f);
  }
}

template <bool accum, typename T, typename index_t>
__global__ void
group_norm_backward_dx(const index_t size, const index_t channel_size,
                       const index_t spatial_size, const int num_groups,
                       const T *x, const T *dy, const T *gamma_invstd,
                       const T *factor1, const T *factor2, T *dx) {
  const index_t offset =
      blockIdx.x * (blockDim.x * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);
// TODO: Grid strided loop

#pragma unroll
  for (auto i = 0; i < GROUP_NORM_ELEMENTWISE_UNROLL_SIZE; i++) {
    const index_t idx = offset + threadIdx.x + i * blockDim.x;
    if (idx < size) {
      const index_t factor_idx =
          idx / (spatial_size * (channel_size / num_groups));
      const index_t param_idx = idx / (spatial_size);
      if (accum) {
        dx[idx] += gamma_invstd[param_idx] * dy[idx] +
                   factor1[factor_idx] * x[idx] + factor2[factor_idx];
      } else {
        dx[idx] = gamma_invstd[param_idx] * dy[idx] +
                  factor1[factor_idx] * x[idx] + factor2[factor_idx];
      }
    }
  }
}

template <bool beta_accum, bool gamma_accum, typename T, typename index_t>
__global__ void group_norm_backward_dbeta_dgamma(
    const index_t batch_size, const index_t channel_size, const int num_groups,
    const T *mean, const T *var, const T *sum_dy, const T *sum_dyx, T *dbeta,
    T *dgamma, const float eps) {
  const index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < channel_size) {
    const index_t chunk_size = channel_size / num_groups;

    float db = 0.0f;
    float dg = 0.0f;

    for (index_t n = 0; n < batch_size; n++) {
      const index_t param_idx = n * channel_size + idx;
      const index_t stats_idx = n * num_groups + idx / chunk_size;
      db += static_cast<float>(sum_dy[param_idx]);
      dg += (sum_dyx[param_idx] - sum_dy[param_idx] * mean[stats_idx]) *
            rsqrt(var[stats_idx] + eps);
    }

    if (dbeta) {
      if (beta_accum) {
        dbeta[idx] += db;
      } else {
        dbeta[idx] = db;
      }
    }
    if (dgamma) {
      if (gamma_accum) {
        dgamma[idx] += dg;
      } else {
        dgamma[idx] = dg;
      }
    }
  }
}

template <typename T>
void GroupNormalizationCuda<T>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  if (channel_last_) {
    pre_transpose_->forward({inputs[0]}, {&pre_adaptor_});

    auto gn_cf_in = inputs;
    gn_cf_in[0] = &pre_adaptor_;
    auto gn_cf_out = outputs;
    gn_cf_out[0] = &post_adaptor_;

    forward_channel_first(gn_cf_in, gn_cf_out);

    post_transpose_->forward({&post_adaptor_}, {outputs[0]});
  } else {
    forward_channel_first(inputs, outputs);
  }
}

template <typename T>
void GroupNormalizationCuda<T>::forward_channel_first(
    const Variables &inputs, const Variables &outputs) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate mean and variance.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    Tc *mean = v_mean->cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *var = v_var->cast_data_and_get_pointer<Tc>(this->ctx_);
    const int num_threads = reduce_size_ < 512 ? 32 : 512; // TODO:
    group_norm_forward_mean_var<<<outer_size_, num_threads>>>(reduce_size_, x,
                                                              mean, var);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate `a` and `b` for simplification of normalization formula
  // as `y = a * x + b`.
  {
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *beta = this->no_bias_
                         ? nullptr
                         : inputs[beta_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    Tc *a = a_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *b = b_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const auto grid =
        NBLA_CEIL_SIZE_T_DIV(batch_size_ * channel_size_, 256); // TODO:
    const auto block = 256;
    group_norm_forward_normalization_factor<<<grid, block>>>(
        batch_size_, channel_size_, this->num_groups_, mean, var, beta, gamma,
        a, b, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Normalization by `y = a * x + b`.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *a = a_.get_data_pointer<Tc>(this->ctx_);
    const Tc *b = b_.get_data_pointer<Tc>(this->ctx_);
    Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const Size_t num_threads = CUDA_WARP_SIZE * 2;

    const auto block = num_threads;
    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, num_threads * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);

    group_norm_forward_normalization<<<grid, block>>>(size, spatial_size, x, a,
                                                      b, y);
    NBLA_CUDA_KERNEL_CHECK();
  }
}

template <typename T>
void GroupNormalizationCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || (inputs.size() > 1 && propagate_down[1]) ||
        (inputs.size() > 2 && propagate_down[2]))) {
    return;
  }
  cuda_set_device(this->device_);

  if (channel_last_) {
    auto gn_cf_in = inputs;
    gn_cf_in[0] = &pre_adaptor_;
    auto gn_cf_out = outputs;
    gn_cf_out[0] = &post_adaptor_;

    post_transpose_->backward({&post_adaptor_}, {outputs[0]}, {true}, {false});

    auto gn_cf_accum = accum;
    gn_cf_accum[0] = false;
    backward_channel_first(gn_cf_in, gn_cf_out, propagate_down, gn_cf_accum);

    pre_transpose_->backward({inputs[0]}, {&pre_adaptor_}, {propagate_down[0]},
                             {accum[0]});
  } else {
    backward_channel_first(inputs, outputs, propagate_down, accum);
  }
}

template <typename T>
void GroupNormalizationCuda<T>::backward_channel_first(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  Variable *v_mean = &mean_;
  Variable *v_var = &var_;
  // Output mean and var when output_stats == true.
  if (outputs.size() == 3) {
    v_mean = outputs[1];
    v_var = outputs[2];
  }

  // Calculate sum of dy and dy*x for the following gradient calculation.
  {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    Tc *sum_dy = sum_dy_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *sum_dyx = sum_dyx_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const auto num_threads = spatial_size < 512 ? CUDA_WARP_SIZE : 512; // TODO:

    const auto grid = batch_size_ * channel_size_;
    const auto block = num_threads;

    group_norm_backward_sum_dy_dyx<<<grid, block>>>(spatial_size, x, dy, sum_dy,
                                                    sum_dyx);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate gamma / sqrt(var)
  {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    Tc *gamma_invstd = gamma_invstd_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = batch_size_ * channel_size_;
    const auto num_threads = CUDA_WARP_SIZE * 2;

    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, GROUP_NORM_ELEMENTWISE_UNROLL_SIZE * num_threads);
    const auto block = num_threads;

    group_norm_backward_gamma_invstd<<<grid, block>>>(
        size, channel_size_, this->num_groups_, gamma, var, gamma_invstd,
        this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate factor1 and factor2
  {
    const auto gamma_idx = this->no_bias_ ? 1 : 2;
    const Tc *gamma = this->no_scale_
                          ? nullptr
                          : inputs[gamma_idx]->get_data_pointer<Tc>(this->ctx_);
    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *dmean = outputs.size() == 3
                          ? v_mean->get_grad_pointer<Tc>(this->ctx_)
                          : nullptr;
    const Tc *dvar =
        outputs.size() == 3 ? v_var->get_grad_pointer<Tc>(this->ctx_) : nullptr;
    const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
    Tc *factor1 = factor1_.cast_data_and_get_pointer<Tc>(this->ctx_);
    Tc *factor2 = factor2_.cast_data_and_get_pointer<Tc>(this->ctx_);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const auto num_threads = CUDA_WARP_SIZE * 2;

    dim3 grid(batch_size_, this->num_groups_);
    dim3 block(num_threads);

    group_norm_backward_dx_factor<<<grid, block>>>(
        channel_size_, spatial_size, this->num_groups_, mean, var, dmean, dvar,
        gamma, sum_dy, sum_dyx, factor1, factor2, this->eps_);
    NBLA_CUDA_KERNEL_CHECK();
  }

  // Calculate dx by `dx = gamma_invstd * dy + factor1 * x + factor2`.
  if (propagate_down[0]) {
    const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
    const Tc *dy = outputs[0]->get_grad_pointer<Tc>(this->ctx_);
    const Tc *gamma_invstd = gamma_invstd_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor1 = factor1_.get_data_pointer<Tc>(this->ctx_);
    const Tc *factor2 = factor2_.get_data_pointer<Tc>(this->ctx_);
    Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, !accum[0]);

    const Size_t size = inputs[0]->size();
    const Size_t spatial_size = size / (batch_size_ * channel_size_);
    const Size_t num_threads = CUDA_WARP_SIZE * 2;

    const auto block = num_threads;
    const auto grid = NBLA_CEIL_SIZE_T_DIV(
        size, num_threads * GROUP_NORM_ELEMENTWISE_UNROLL_SIZE);

    if (accum[0]) {
      group_norm_backward_dx<true><<<grid, block>>>(
          size, channel_size_, spatial_size, this->num_groups_, x, dy,
          gamma_invstd, factor1, factor2, dx);
      NBLA_CUDA_KERNEL_CHECK();
      // std::cout << "debug 01" << std::endl;
    } else {
      group_norm_backward_dx<false><<<grid, block>>>(
          size, channel_size_, spatial_size, this->num_groups_, x, dy,
          gamma_invstd, factor1, factor2, dx);
      NBLA_CUDA_KERNEL_CHECK();
      // std::cout << "debug 02" << std::endl;
    }
  }

  if ((inputs.size() > 1 && propagate_down[1]) ||
      (inputs.size() > 2 && propagate_down[2])) {
    // TODO: optional beta, gamma
    const auto beta_idx = 1;
    const auto gamma_idx = this->no_bias_ ? 1 : 2;

    const Tc *mean = v_mean->get_data_pointer<Tc>(this->ctx_);
    const Tc *var = v_var->get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dy = sum_dy_.get_data_pointer<Tc>(this->ctx_);
    const Tc *sum_dyx = sum_dyx_.get_data_pointer<Tc>(this->ctx_);
    Tc *dbeta = !this->no_bias_ && propagate_down[beta_idx]
                    ? inputs[beta_idx]->cast_grad_and_get_pointer<Tc>(
                          this->ctx_, !accum[beta_idx])
                    : nullptr;
    Tc *dgamma = !this->no_scale_ && propagate_down[gamma_idx]
                     ? inputs[gamma_idx]->cast_grad_and_get_pointer<Tc>(
                           this->ctx_, !accum[gamma_idx])
                     : nullptr;

    const auto block = 256; // TODO:
    const auto grid = NBLA_CEIL_SIZE_T_DIV(channel_size_, 256);

    if (!this->no_bias_ && accum[beta_idx]) {
      if (!this->no_scale_ && accum[gamma_idx]) {
        group_norm_backward_dbeta_dgamma<true, true><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      } else {
        group_norm_backward_dbeta_dgamma<true, false><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      }
    } else {
      if (!this->no_scale_ && accum[gamma_idx]) {
        group_norm_backward_dbeta_dgamma<false, true><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      } else {
        group_norm_backward_dbeta_dgamma<false, false><<<grid, block>>>(
            batch_size_, channel_size_, this->num_groups_, mean, var, sum_dy,
            sum_dyx, dbeta, dgamma, this->eps_);
      }
    }
    NBLA_CUDA_KERNEL_CHECK();
  }
}
}

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

#include <nbla/cuda/function/kernel/sync_batch_normalization.cuh>

namespace nbla {

static bool can_use_int_as_index_t(const Size_t size0, const Size_t size1,
                                   const Size_t size2) {
  return size0 * size1 * size2 < std::numeric_limits<int>::max();
}

template <typename T>
void forward_collect_statistics(const Size_t size0, const Size_t size1,
                                const Size_t size2, Variable *x,
                                Variable *local_mean, Variable *local_invstd,
                                const float epsilon, Context &ctx) {
  using input_scalar_t = T;
  using stat_accscalar_t = typename CudaTypeForceFloat<T>::type;

  const auto *x_ptr = x->get_data_pointer<input_scalar_t>(ctx);
  auto *local_mean_ptr =
      local_mean->cast_data_and_get_pointer<stat_accscalar_t>(ctx);
  auto *local_invstd_ptr =
      local_invstd->cast_data_and_get_pointer<stat_accscalar_t>(ctx);

  dim3 blocks(size1);
  int tf = getNumThreads(size2);
  dim3 threads(tf, SYNC_BN_MAX_BLOCK_SIZE / tf);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_collect_statistics_kernel<InvStd, input_scalar_t,
                                         stat_accscalar_t, index_t>
        <<<blocks, threads>>>(x_ptr, epsilon, local_mean_ptr, local_invstd_ptr,
                              size0, size1, size2);
  } else {
    using index_t = Size_t;
    batch_norm_collect_statistics_kernel<InvStd, input_scalar_t,
                                         stat_accscalar_t, index_t>
        <<<blocks, threads>>>(x_ptr, epsilon, local_mean_ptr, local_invstd_ptr,
                              size0, size1, size2);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void forward_collect_statistics_channels_last(
    const Size_t size0, const Size_t size1, const Size_t size2, Variable *x,
    Variable *local_mean, Variable *local_invstd, Variable *staging_data,
    Variable *semaphores, const float epsilon, Context &ctx) {
  using scalar_t = T;
  using accscalar_t = typename CudaTypeForceFloat<T>::type;
  const Size_t reduction_size = size0;
  const Size_t stride = size1;

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  if (grid.y > 1) {
    staging_data->reshape({4 * stride * grid.y}, true);
    semaphores->reshape({grid.x}, true);
  }

  const auto *x_ptr = x->get_data_pointer<scalar_t>(ctx);
  auto *local_mean_ptr =
      local_mean->cast_data_and_get_pointer<accscalar_t>(ctx);
  auto *local_invstd_ptr =
      local_invstd->cast_data_and_get_pointer<accscalar_t>(ctx);

  auto *staging_data_ptr =
      grid.y > 1 ? staging_data->cast_data_and_get_pointer<accscalar_t>(ctx)
                 : nullptr;
  int *semaphores_ptr =
      grid.y > 1 ? semaphores->cast_data_and_get_pointer<int>(ctx) : nullptr;

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_collect_statistics_channels_last_kernel<
        InvStd, scalar_t, accscalar_t, index_t, SYNC_BN_ELEMENTS_PER_ITER>
        <<<grid, block>>>(x_ptr, local_mean_ptr, local_invstd_ptr,
                          staging_data_ptr, semaphores_ptr, reduction_size,
                          stride, epsilon);
  } else {
    using index_t = Size_t;
    batch_norm_collect_statistics_channels_last_kernel<
        InvStd, scalar_t, accscalar_t, index_t, SYNC_BN_ELEMENTS_PER_ITER>
        <<<grid, block>>>(x_ptr, local_mean_ptr, local_invstd_ptr,
                          staging_data_ptr, semaphores_ptr, reduction_size,
                          stride, epsilon);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void forward_reduce_statistics(const Size_t size0, const Size_t size1,
                               const Size_t size2, Variable *all_mean,
                               Variable *all_invstd, Variable *all_count,
                               Variable *global_mean, Variable *global_var,
                               Variable *r_mean, Variable *r_var,
                               const float epsilon, const float decay_rate,
                               Context &ctx, const int n_workers) {
  using scalar_t = T;
  using accscalar_t = typename CudaTypeForceFloat<T>::type;

  const accscalar_t *all_mean_ptr =
      all_mean->get_data_pointer<accscalar_t>(ctx);
  const accscalar_t *all_invstd_ptr =
      all_invstd->get_data_pointer<accscalar_t>(ctx);

  auto *global_mean_ptr =
      global_mean->cast_data_and_get_pointer<accscalar_t>(ctx);
  auto *global_var_ptr =
      global_var->cast_data_and_get_pointer<accscalar_t>(ctx);

  auto *r_mean_ptr =
      r_mean ? r_mean->cast_data_and_get_pointer<scalar_t>(ctx) : nullptr;
  auto *r_var_ptr =
      r_var ? r_var->cast_data_and_get_pointer<scalar_t>(ctx) : nullptr;

  const auto *all_count_ptr = all_count->get_data_pointer<scalar_t>(ctx);

  const int feature_size = size1;
  int block = getNumThreads(feature_size);
  int grid = std::max<int>(1, feature_size / block);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_reduce_statistics_kernel<scalar_t, accscalar_t, index_t>
        <<<grid, block>>>(all_mean_ptr, all_invstd_ptr, global_mean_ptr,
                          global_var_ptr, r_mean_ptr, r_var_ptr, epsilon,
                          decay_rate, all_count_ptr, feature_size, n_workers);
  } else {
    using index_t = Size_t;
    batch_norm_reduce_statistics_kernel<scalar_t, accscalar_t, index_t>
        <<<grid, block>>>(all_mean_ptr, all_invstd_ptr, global_mean_ptr,
                          global_var_ptr, r_mean_ptr, r_var_ptr, epsilon,
                          decay_rate, all_count_ptr, feature_size, n_workers);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void forward_normalization(const Size_t size0, const Size_t size1,
                           const Size_t size2, Variable *x, Variable *y,
                           Variable *global_mean, Variable *global_var,
                           Variable *beta, Variable *gamma, const float epsilon,
                           Context &ctx) {
  using input_scalar_t = T;
  using stat_scalar_t = T;
  using stat_accscalar_t = typename CudaTypeForceFloat<T>::type;

  const auto *x_ptr = x->get_data_pointer<input_scalar_t>(ctx);
  auto *y_ptr = y->cast_data_and_get_pointer<input_scalar_t>(ctx);
  const auto *global_mean_ptr =
      global_mean->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *global_var_ptr =
      global_var->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *weight_ptr = gamma->get_data_pointer<stat_scalar_t>(ctx);
  const auto *bias_ptr = beta->get_data_pointer<stat_scalar_t>(ctx);

  // (Following comments are quoted from PyTorch.)
  // The input_transform kernel is pointwise, but we need to balance reading
  // parameters (save_var/mean, weight/bias) - which we only do once and have a
  // for loop afterwards - with having many threads and blocks and good
  // occupancy. Quiet likely, we could go with even more blocks than 1024.
  // The various planes are independent, so we use blocks for them.
  const Size_t tf = std::max<int>(getNumThreads(size2 / 4),
                                  std::min<int>(getNumThreads(size1), 64));
  const Size_t tb = std::max<int>(64 / tf, 1);
  dim3 blocks_trans(size1,
                    std::max<int>(1, std::min<int>((256 * 1024) / size1,
                                                   (size0 + tb - 1) / tb)));
  blocks_trans.y = std::min(blocks_trans.y, SYNC_BN_MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t,
                                      stat_accscalar_t, index_t>
        <<<blocks_trans, threads_trans>>>(x_ptr, y_ptr, global_mean_ptr,
                                          global_var_ptr, weight_ptr, bias_ptr,
                                          epsilon, size0, size1, size2);
  } else {
    using index_t = Size_t;
    batch_norm_transform_input_kernel<input_scalar_t, stat_scalar_t,
                                      stat_accscalar_t, index_t>
        <<<blocks_trans, threads_trans>>>(x_ptr, y_ptr, global_mean_ptr,
                                          global_var_ptr, weight_ptr, bias_ptr,
                                          epsilon, size0, size1, size2);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void forward_normalization_channel_last(const Size_t size0, const Size_t size1,
                                        const Size_t size2, Variable *x,
                                        Variable *global_mean,
                                        Variable *global_var, Variable *beta,
                                        Variable *gamma, Variable *y,
                                        const float epsilon, Context &ctx) {
  using scalar_t = T;
  using layerscalar_t = T;
  using accscalar_t = typename CudaTypeForceFloat<T>::type;

  const Size_t reduction_size = size0;
  const Size_t stride = size1;

  const scalar_t *x_ptr = x->get_data_pointer<scalar_t>(ctx);

  const accscalar_t *global_mean_ptr =
      global_mean->get_data_pointer<accscalar_t>(ctx);
  const accscalar_t *global_var_ptr =
      global_var->get_data_pointer<accscalar_t>(ctx);

  const layerscalar_t *gamma_ptr = gamma->get_data_pointer<layerscalar_t>(ctx);
  const layerscalar_t *beta_ptr = beta->get_data_pointer<layerscalar_t>(ctx);

  scalar_t *y_ptr = y->cast_data_and_get_pointer<scalar_t>(ctx);

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_transform_input_channels_last_kernel<scalar_t, accscalar_t,
                                                    layerscalar_t, index_t,
                                                    SYNC_BN_ELEMENTS_PER_ITER>
        <<<grid, block>>>(x_ptr, global_mean_ptr, global_var_ptr, gamma_ptr,
                          beta_ptr, y_ptr, epsilon, reduction_size, stride);
  } else {
    using index_t = Size_t;
    batch_norm_transform_input_channels_last_kernel<scalar_t, accscalar_t,
                                                    layerscalar_t, index_t,
                                                    SYNC_BN_ELEMENTS_PER_ITER>
        <<<grid, block>>>(x_ptr, global_mean_ptr, global_var_ptr, gamma_ptr,
                          beta_ptr, y_ptr, epsilon, reduction_size, stride);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void backward_reduce(const Size_t size0, const Size_t size1, const Size_t size2,
                     Variable *x, Variable *y, Variable *global_mean,
                     Variable *global_var, Variable *sum_dy,
                     Variable *sum_dy_xmu, Variable *beta, Variable *gamma,
                     const float epsilon, Context &ctx) {
  using input_scalar_t = T;
  using stat_scalar_t = T;
  using stat_accscalar_t = typename CudaTypeForceFloat<T>::type;

  const auto *x_ptr = x->get_data_pointer<input_scalar_t>(ctx);
  const auto *dy_ptr = y->get_grad_pointer<input_scalar_t>(ctx);
  const auto *global_mean_ptr =
      global_mean->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *global_var_ptr =
      global_var->get_data_pointer<stat_accscalar_t>(ctx);
  auto *sum_dy_ptr =
      sum_dy->cast_data_and_get_pointer<stat_accscalar_t>(ctx, true);
  auto *sum_dy_xmu_ptr =
      sum_dy_xmu->cast_data_and_get_pointer<stat_accscalar_t>(ctx, true);
  auto *grad_weight_ptr =
      gamma->cast_data_and_get_pointer<stat_scalar_t>(ctx, true);
  auto *grad_bias_ptr =
      beta->cast_data_and_get_pointer<stat_scalar_t>(ctx, true);

  auto batch_size = size0;
  auto n_input = size1;
  auto feature_size = size2;

  int block_y = std::min<int>(lastPow2(batch_size),
                              SYNC_BN_MAX_BLOCK_SIZE / CUDA_WARP_SIZE);
  // We want block_x to be at least a warp width
  int block_x =
      std::min<int>(std::max<int>(getNumThreads(feature_size), CUDA_WARP_SIZE),
                    SYNC_BN_MAX_BLOCK_SIZE / block_y);
  const dim3 grid(n_input);
  const dim3 block(block_x, block_y);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_backward_reduce_kernel<input_scalar_t, stat_scalar_t,
                                      stat_accscalar_t, index_t>
        <<<grid, block>>>(x_ptr, dy_ptr, global_mean_ptr, global_var_ptr,
                          sum_dy_ptr, sum_dy_xmu_ptr, grad_weight_ptr,
                          grad_bias_ptr, epsilon, size0, size1, size2);
  } else {
    using index_t = Size_t;
    batch_norm_backward_reduce_kernel<input_scalar_t, stat_scalar_t,
                                      stat_accscalar_t, index_t>
        <<<grid, block>>>(x_ptr, dy_ptr, global_mean_ptr, global_var_ptr,
                          sum_dy_ptr, sum_dy_xmu_ptr, grad_weight_ptr,
                          grad_bias_ptr, epsilon, size0, size1, size2);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T>
void backward_reduce_channels_last(const Size_t size0, const Size_t size1,
                                   const Size_t size2, Variable *x, Variable *y,
                                   Variable *batch_mean, Variable *batch_var,
                                   Variable *sum_dy_o, Variable *sum_dy_xmu_o,
                                   Variable *beta, Variable *gamma,
                                   Variable *staging_data, Variable *semaphores,
                                   const float epsilon, Context &ctx) {
  using scalar_t = T;
  using layerscalar_t = T;
  using accscalar_t = typename CudaTypeForceFloat<T>::type;

  const Size_t reduction_size = size0;
  const Size_t stride = size1;

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  if (grid.y > 1) {
    staging_data->reshape({2 * stride * grid.y}, true);
    semaphores->reshape({grid.x}, true);
  }

  const auto *x_ptr = x->get_data_pointer<scalar_t>(ctx);
  const auto *dy_ptr = y->get_grad_pointer<scalar_t>(ctx);
  const auto *mean_ptr = batch_mean->get_data_pointer<accscalar_t>(ctx);
  const auto *var_ptr = batch_var->get_data_pointer<accscalar_t>(ctx);

  auto *sum_dy_o_ptr = sum_dy_o->cast_data_and_get_pointer<accscalar_t>(ctx);
  auto *sum_dy_xmu_o_ptr =
      sum_dy_xmu_o->cast_data_and_get_pointer<accscalar_t>(ctx);
  auto *grad_weight_ptr = gamma->cast_data_and_get_pointer<layerscalar_t>(ctx);
  auto *grad_bias_ptr = beta->cast_data_and_get_pointer<layerscalar_t>(ctx);

  auto *staging_data_ptr =
      staging_data->cast_data_and_get_pointer<accscalar_t>(ctx);
  int *semaphores_ptr = semaphores->cast_data_and_get_pointer<int>(ctx);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_backward_reduce_channels_last_kernel<SYNC_BN_ELEMENTS_PER_ITER,
                                                    scalar_t, accscalar_t,
                                                    layerscalar_t, index_t>
        <<<grid, block>>>(x_ptr, dy_ptr, mean_ptr, var_ptr, sum_dy_o_ptr,
                          sum_dy_xmu_o_ptr, grad_weight_ptr, grad_bias_ptr,
                          staging_data_ptr, semaphores_ptr, reduction_size,
                          stride, epsilon);
  } else {
    using index_t = Size_t;
    batch_norm_backward_reduce_channels_last_kernel<SYNC_BN_ELEMENTS_PER_ITER,
                                                    scalar_t, accscalar_t,
                                                    layerscalar_t, index_t>
        <<<grid, block>>>(x_ptr, dy_ptr, mean_ptr, var_ptr, sum_dy_o_ptr,
                          sum_dy_xmu_o_ptr, grad_weight_ptr, grad_bias_ptr,
                          staging_data_ptr, semaphores_ptr, reduction_size,
                          stride, epsilon);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T, bool accum>
void backward_dx_post(const Size_t size0, const Size_t size1,
                      const Size_t size2, Variable *x, Variable *y,
                      Variable *global_mean, Variable *global_var,
                      Variable *sum_dy, Variable *sum_dy_xmu, Variable *gamma,
                      Variable *all_count, const bool output_stat,
                      const float epsilon, Context &ctx) {
  using input_scalar_t = T;
  using stat_scalar_t = T;
  using stat_accscalar_t = typename CudaTypeForceFloat<T>::type;

  const auto *x_ptr = x->get_data_pointer<input_scalar_t>(ctx);
  const auto *dy_ptr = y->get_grad_pointer<input_scalar_t>(ctx);
  const auto *global_mean_ptr =
      global_mean->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *global_var_ptr =
      global_var->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *dmean_ptr =
      output_stat ? global_mean->get_grad_pointer<stat_accscalar_t>(ctx)
                  : nullptr;
  const auto *dvar_ptr =
      output_stat ? global_var->get_grad_pointer<stat_accscalar_t>(ctx)
                  : nullptr;
  const auto *weight_ptr = gamma->get_data_pointer<stat_scalar_t>(ctx);
  const auto *sum_dy_ptr = sum_dy->get_data_pointer<stat_accscalar_t>(ctx);
  const auto *sum_dy_xmu_ptr =
      sum_dy_xmu->get_data_pointer<stat_accscalar_t>(ctx);

  auto *dx_ptr = x->cast_grad_and_get_pointer<input_scalar_t>(ctx, false);

  const int *all_count_ptr = all_count->get_data_pointer<int>(ctx);
  const Size_t all_count_numel = all_count->size();

  const Size_t tf = std::max<int>(getNumThreads(size2 / 4),
                                  std::min<int>(getNumThreads(size2), 64));
  const Size_t tb = std::max<int>(64 / tf, 1);
  dim3 blocks_trans(size1,
                    std::max<int>(1, std::min<int>((256 * 1024) / size1,
                                                   (size0 + tb - 1) / tb)));
  blocks_trans.y = std::min(blocks_trans.y, SYNC_BN_MAX_GRID_SIZE);
  dim3 threads_trans(tf, tb);

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_backward_elemt_kernel<accum, input_scalar_t, stat_scalar_t,
                                     stat_accscalar_t, index_t>
        <<<blocks_trans, threads_trans>>>(
            x_ptr, dy_ptr, global_mean_ptr, global_var_ptr, dmean_ptr, dvar_ptr,
            weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, dx_ptr, epsilon,
            all_count_ptr, all_count_numel, size0, size1, size2);
  } else {
    using index_t = Size_t;
    batch_norm_backward_elemt_kernel<accum, input_scalar_t, stat_scalar_t,
                                     stat_accscalar_t, index_t>
        <<<blocks_trans, threads_trans>>>(
            x_ptr, dy_ptr, global_mean_ptr, global_var_ptr, dmean_ptr, dvar_ptr,
            weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, dx_ptr, epsilon,
            all_count_ptr, all_count_numel, size0, size1, size2);
  }
  NBLA_CUDA_KERNEL_CHECK();
}

template <typename T, bool accum>
void backward_dx_post_channels_last(const Size_t size0, const Size_t size1,
                                    const Size_t size2, Variable *y,
                                    Variable *x, Variable *batch_mean,
                                    Variable *batch_var, Variable *gamma,
                                    Variable *sum_dy_o, Variable *sum_dy_xmu_o,
                                    Variable *count, const bool output_stat,
                                    const float epsilon, Context &ctx) {
  using scalar_t = T;
  using layerscalar_t = T;
  using accscalar_t = typename CudaTypeForceFloat<T>::type;

  const Size_t reduction_size = size0;
  const Size_t stride = size1;

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  const auto *dy_ptr = y->get_grad_pointer<scalar_t>(ctx);
  const auto *x_ptr = x->get_data_pointer<scalar_t>(ctx);
  const auto *mean_ptr = batch_mean->get_data_pointer<accscalar_t>(ctx);
  const auto *var_ptr = batch_var->get_data_pointer<accscalar_t>(ctx);
  const auto *dmean_ptr =
      output_stat ? batch_mean->get_grad_pointer<accscalar_t>(ctx) : nullptr;
  const auto *dvar_ptr =
      output_stat ? batch_var->get_grad_pointer<accscalar_t>(ctx) : nullptr;

  const auto *weight_ptr = gamma->get_data_pointer<layerscalar_t>(ctx);

  const auto *sum_dy_ptr = sum_dy_o->get_data_pointer<accscalar_t>(ctx);
  const auto *sum_dy_xmu_ptr = sum_dy_xmu_o->get_data_pointer<accscalar_t>(ctx);

  const auto *numel_ptr = count->get_data_pointer<int>(ctx);

  auto *dx_ptr = x->cast_grad_and_get_pointer<scalar_t>(ctx);

  const Size_t world_size = count->size();

  if (can_use_int_as_index_t(size0, size1, size2)) {
    using index_t = int;
    batch_norm_backward_elemt_channels_last_kernel<SYNC_BN_ELEMENTS_PER_ITER,
                                                   accum, scalar_t, accscalar_t,
                                                   layerscalar_t, index_t>
        <<<grid, block>>>(dy_ptr, x_ptr, mean_ptr, var_ptr, dmean_ptr, dvar_ptr,
                          weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, numel_ptr,
                          dx_ptr, world_size, reduction_size, stride, epsilon);
  } else {
    using index_t = Size_t;
    batch_norm_backward_elemt_channels_last_kernel<SYNC_BN_ELEMENTS_PER_ITER,
                                                   accum, scalar_t, accscalar_t,
                                                   layerscalar_t, index_t>
        <<<grid, block>>>(dy_ptr, x_ptr, mean_ptr, var_ptr, dmean_ptr, dvar_ptr,
                          weight_ptr, sum_dy_ptr, sum_dy_xmu_ptr, numel_ptr,
                          dx_ptr, world_size, reduction_size, stride, epsilon);
  }
  NBLA_CUDA_KERNEL_CHECK();
}
} // namespace nbla

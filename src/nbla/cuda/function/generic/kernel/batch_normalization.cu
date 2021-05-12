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

#include <nbla/cuda/function/kernel/batch_normalization.cuh>

namespace nbla {

template <typename T>
void forward_batch(const int size0, const int size1, const int size2,
                   const float decay_rate, const float eps, const T *x,
                   const T *gamma, const T *beta, T *m, T *v, T *rm, T *rv,
                   T *y) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_mean_variance_kernel,
                                 /* Input */
                                 size1, size2, size0 * size2, size1 * size2,
                                 decay_rate, eps, x, gamma, beta,
                                 /* Output */
                                 m, v, rm, rv);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_gamma_beta_kernel,
                                 /* Input */
                                 size1 * size0 * size2, size0, size2,
                                 size0 * size2, size1 * size2, decay_rate, eps,
                                 x, m, v, rm, rv, gamma, beta,
                                 /* Output */
                                 y);
}

template <typename T>
void backward_batch_data(const int size0, const int size1, const int size2,
                         const float decay_rate, const float eps, const T *dy,
                         const T *m, const T *v, const T *x, const T *g,
                         const T *dm, const T *dv, T *dx, T *dmean, T *dvar) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_batch_data_mean_variance_kernel,
                                 /* Input */
                                 size1, size2, size0 * size2, size1 * size2,
                                 decay_rate, eps, dy, m, v, x, g, dm, dv,
                                 /* Output */
                                 dmean, dvar);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_batch_data_dx_kernel,
                                 /* Input */
                                 size1 * size0 * size2, size0, size1, size2,
                                 size0 * size2, size1 * size2, decay_rate, eps,
                                 dy, m, v, x, g, dm, dv, dmean, dvar,
                                 /* Output */
                                 dx);
}

//#define TEST_FEATURE_MEAN_VARIANCE_AXIS_REDUCTION_KERNEL
//#define TEST_FEATURE_MEAN_VARIANCE_KERNEL

template <typename T>
void forward_batch_parallel_reduction(
    const int size0, const int size1, const int size2, const int ndim,
    const int *axes, const int *x_strides, const int *x_shape,
    const int *y_strides, const int *y_shape, const float decay_rate,
    const float eps, const T *x, const T *gamma, const T *beta, T *x_trans,
    T *m, T *v, T *rm, T *rv, T *y, T *tmp_mean_buffer_per_block,
    T *tmp_variance_buffer_per_block, T *inv_sqrt_variance) {
  int N = size0 * size2;
  reduction_blocks(blocks, N);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(transpose_kernel, size1 * N, ndim, axes,
                                 x_strides, y_strides, y_shape, x, x_trans);
#ifdef TEST_FEATURE_MEAN_VARIANCE_AXIS_REDUCTION_KERNEL
  printf("TEST_FEATURE_MEAN_VARIANCE_AXIS_REDUCTION_KERNEL\n");
  mean_variance_with_axis_kernel<<<blocks, NBLA_CUDA_NUM_THREADS>>>(
      x_trans, tmp_mean_buffer_per_block, tmp_variance_buffer_per_block, m, v,
      N, blocks, size1);
#elif defined TEST_FEATURE_MEAN_VARIANCE_KERNEL
  printf("TEST_FEATURE_MEAN_VARIANCE_KERNEL\n");
  for (int i = 0; i < size1; ++i) {
    mean_variance_kernel<<<blocks, NBLA_CUDA_NUM_THREADS>>>(
        x_trans + i * N, tmp_mean_buffer_per_block,
        tmp_variance_buffer_per_block, m + i, v + i, N, blocks);
  }
#else
  blocks = min((N + NBLA_CUDA_NUM_THREADS - 1) / NBLA_CUDA_NUM_THREADS, 1024);
  for (int i = 0; i < size1; ++i) {
    forward_batch_kernel_mean_variance_preprocess<<<blocks,
                                                    NBLA_CUDA_NUM_THREADS>>>(
        /* Input */
        x_trans + i * N, N,
        /* Output */
        tmp_mean_buffer_per_block, tmp_variance_buffer_per_block);
    forward_batch_kernel_mean_variance_postprocess<<<1, 1024>>>(
        /* Input */
        tmp_mean_buffer_per_block, tmp_variance_buffer_per_block, blocks,
        decay_rate, 1. / N, (float)N / (N - 1),
        /* Output */
        m + i, v + i, rm ? rm + i : nullptr, rv ? rv + i : nullptr);
  }
#endif
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward_batch_kernel_gamma_beta_trans,
                                 /* Input */
                                 size1 * N, N, x_trans, gamma, beta, m, v,
                                 decay_rate, eps, ndim, axes, y_strides,
                                 x_strides, x_shape,
                                 /* Output */
                                 y, inv_sqrt_variance);
}

template <typename T>
void backward_batch_data_parallel_reduction(
    const int size0, const int size1, const int size2, const int ndim,
    const int *axes, const int *x_strides, const int *x_shape,
    const int *y_strides, const int *y_shape, const float decay_rate,
    const float eps, const T *dy, const T *m, const T *v, const T *x,
    const T *g, const T *dm, const T *dv, T *dx, T *tmp_mean_buffer_per_block,
    T *tmp_variance_buffer_per_block, T *tmp_t_buffer_per_block, T *dmean,
    T *dvar, T *t, T *inv_sqrt_variance, T *x_trans, T *dy_trans) {
  int N = size0 * size2;
  int shape_size = size1 * N;
  int blocks =
      min((N + NBLA_CUDA_NUM_THREADS - 1) / NBLA_CUDA_NUM_THREADS, 1024);
  for (int i = 0; i < size1; i++) {
    backward_batch_data_kernel_mean_variance_preprocess<<<
        blocks, NBLA_CUDA_NUM_THREADS>>>(
        /* Input */
        N, dy_trans + i * N, x_trans + i * N, g ? g + i : nullptr, m + i,
        /* Output */
        tmp_mean_buffer_per_block, tmp_variance_buffer_per_block,
        tmp_t_buffer_per_block);
    backward_batch_data_kernel_mean_variance_postprocess<<<1, 1024>>>(
        /* Input */
        tmp_mean_buffer_per_block, tmp_variance_buffer_per_block,
        tmp_t_buffer_per_block, blocks, 1. / N, v + i, dm, dv, eps, N,
        inv_sqrt_variance + i, i,
        /* Output */
        dmean + i, dvar + i, t + i);
  }
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(backward_batch_data_kernel_gamma_beta_trans,
                                 /* Input */
                                 shape_size, 1. / N, dy_trans, x_trans, g, v, m,
                                 dmean, dvar, ndim, axes, y_strides, x_strides,
                                 x_shape, inv_sqrt_variance,
                                 /* Output */
                                 dx);
}

template <typename T>
void backward_batch_gamma_beta_parallel_reduction(
    const int size0, const int size1, const int size2, const T *dy_trans,
    const T *m, const T *v, const T *x_trans, float eps, T *db, T *dg,
    T *gamma_reduction_space, T *beta_reduction_space, T *inv_sqrt_variance) {
  int N = size0 * size2;
  int blocks =
      min((N + NBLA_CUDA_NUM_THREADS - 1) / NBLA_CUDA_NUM_THREADS, 1024);

  for (int i = 0; i < size1; i++) {
    backward_batch_kernel_gamma_beta_preprocess<<<blocks,
                                                  NBLA_CUDA_NUM_THREADS>>>(
        /* Input */
        N, dy_trans + i * N, x_trans + i * N, m + i,
        /* Output */
        gamma_reduction_space, beta_reduction_space, inv_sqrt_variance + i);
    backward_batch_kernel_gamma_beta_postprocess<<<1, 1024>>>(
        /* Input */
        gamma_reduction_space, beta_reduction_space, blocks,
        /* Output */
        dg ? dg + i : nullptr, db ? db + i : nullptr);
  }
}
}

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

#include <nbla/cuda/function/kernel/cumprod.cuh>
#include <nbla/cuda/utils/scan_ops/prod.cuh>
#include <nbla/cuda/utils/scan_ops/sum.cuh>

namespace nbla {

/**
 * @brief Launch naive cumprod backward kernel.
 */
template <typename Tcu, typename IndexT, bool accum, bool exclusive,
          bool reverse>
void cumprod_backward_naive(const Context &ctx, const Tcu *g_y, const Tcu *x,
                            Tcu *g_x, const ScanSetup &setup) {
  using AccumType = typename CudaTypeForceFloat<Tcu>::type;

  // `masked_cumprod` is a cumulative prod of `x` but treating the first zero
  // element as `1` on each `axis`.
  Variable v_masked_cumprod({setup.size_input});
  AccumType *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<AccumType>(ctx, true);

  auto kernel = kernel_cumprod_backward<Tcu, AccumType, IndexT, accum,
                                        exclusive, reverse>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, setup.size_outer * setup.size_inner,
                                 setup.size_scan, setup.size_inner, x, g_y,
                                 masked_cumprod, g_x);
}

/**
 * @brief Launch composited cumprod backward kernels.
 *
 * The algorithm used here is exactly same as C++ implement but divided into
 * multiple kernel calls.
 * See details about the algorithm in
 * `nnabla/src/nbla/function/generic/cumprod.cpp`.
 */
template <typename Tcu, typename IndexT, bool accum, bool exclusive,
          bool reverse>
void cumprod_backward(const Context &ctx, const Tcu *g_y, const Tcu *x,
                      Tcu *g_x, const ScanSetup &setup) {
  using AccumType = typename CudaTypeForceFloat<Tcu>::type;

  // Step 1: Normal cumprod
  Variable v_cumprod({setup.size_input});
  Tcu *cumprod = v_cumprod.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_cumprod = setup;
  device_cumprod(ctx, x, cumprod, setup_cumprod, false /* accum */);

  // Step 2: Find first 0 index
  Variable v_first_zero_index({setup.size_outer, setup.size_inner});
  IndexT *first_zero_index =
      v_first_zero_index.cast_data_and_get_pointer<IndexT>(ctx, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_fill_size_scan<IndexT>,
                                 setup.size_outer * setup.size_inner,
                                 setup.size_scan, first_zero_index);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_first_zero_index<Tcu, IndexT, reverse>), setup.size_input,
      setup.size_scan, setup.size_inner, x, first_zero_index);

  // Step 3: Mask input
  Variable v_masked_input({setup.size_input});
  Tcu *masked_input = v_masked_input.cast_data_and_get_pointer<Tcu>(ctx, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      (kernel_mask_input<Tcu, IndexT, reverse>), setup.size_input,
      setup.size_scan, setup.size_inner, x, first_zero_index, masked_input);

  // Step 4: Masked cumprod
  Variable v_masked_cumprod({setup.size_input});
  Tcu *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_masked_cumprod = setup;
  device_cumprod(ctx, masked_input, masked_cumprod, setup_masked_cumprod,
                 false /* accum */);

  // Step 5: Prod dy to cumprod and masked_cumprod
  Variable v_cumprod_dy({setup.size_input}),
      v_masked_cumprod_dy({setup.size_input});
  Tcu *cumprod_dy = v_cumprod_dy.cast_data_and_get_pointer<Tcu>(ctx, true);
  Tcu *masked_cumprod_dy =
      v_masked_cumprod_dy.cast_data_and_get_pointer<Tcu>(ctx, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_prod_dy<Tcu, IndexT>),
                                 setup.size_input, cumprod, masked_cumprod, g_y,
                                 cumprod_dy, masked_cumprod_dy);

  // Step 6: Reversed-cumsum of cumprod_dy
  Variable v_cumsum({setup.size_input});
  Tcu *cumsum = v_cumsum.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_cumsum = setup;
  setup_cumsum.reverse = !setup.reverse;
  device_cumsum(ctx, cumprod_dy, cumsum, setup_cumsum, false /* accum */);

  // Step 7: Reversed-cumsum of masked_cumprod_dy
  Variable v_masked_cumsum({setup.size_input});
  Tcu *masked_cumsum =
      v_masked_cumsum.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_masked_cumsum = setup;
  setup_masked_cumsum.reverse = !setup.reverse;
  device_cumsum(ctx, masked_cumprod_dy, masked_cumsum, setup_masked_cumsum,
                false /* accum */);

  // Step 8: Calculate g_x
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_dx<Tcu, IndexT, reverse, accum>),
                                 setup.size_input, setup.size_scan,
                                 setup.size_inner, x, cumsum, masked_cumsum,
                                 first_zero_index, g_x);
}

template <typename Tcu, typename IndexT, bool accum, bool exclusive,
          bool reverse>
void device_cumprod_backward_impl(const Context &ctx, const Tcu *g_y,
                                  const Tcu *x, Tcu *g_x,
                                  const ScanSetup &setup) {
  // There are two backward implementation. One is using a single naive kernel
  // and the other is composite of multiple kernels.
  // The former is optimized for the scanning to non-contiguous memory direction
  // and small scan size with enough outer size.
  // The latter can perform inter-block scan and it is suitable for large scan
  // size.
  // The switching condition bellow is determined heauristically.
  const bool use_naive_kernel = setup.size_inner != 1 &&
                                setup.size_outer * setup.size_inner >= 1024 &&
                                setup.size_scan <= 128;
  if (use_naive_kernel) {
    cumprod_backward_naive<Tcu, IndexT, accum, exclusive, reverse>(ctx, g_y, x,
                                                                   g_x, setup);
  } else {
    cumprod_backward<Tcu, IndexT, accum, exclusive, reverse>(ctx, g_y, x, g_x,
                                                             setup);
  }
}

// The following `dispatch_*` functions are only for the purpose of dealing with
// template arguments of `device_cumprod_backward_impl`.

template <typename Tcu, typename IndexT, bool accum, bool exclusive>
void dispatch_device_cumprod_backward_reverse(const Context &ctx,
                                              const Tcu *g_y, const Tcu *x,
                                              Tcu *g_x,
                                              const ScanSetup &setup) {
  if (setup.reverse) {
    device_cumprod_backward_impl<Tcu, IndexT, accum, exclusive, true>(
        ctx, g_y, x, g_x, setup);
  } else {
    device_cumprod_backward_impl<Tcu, IndexT, accum, exclusive, false>(
        ctx, g_y, x, g_x, setup);
  }
}

template <typename Tcu, typename IndexT, bool accum>
void dispatch_device_cumprod_backward_exclusive(const Context &ctx,
                                                const Tcu *g_y, const Tcu *x,
                                                Tcu *g_x,
                                                const ScanSetup &setup) {
  if (setup.exclusive) {
    dispatch_device_cumprod_backward_reverse<Tcu, IndexT, accum, true>(
        ctx, g_y, x, g_x, setup);
  } else {
    dispatch_device_cumprod_backward_reverse<Tcu, IndexT, accum, false>(
        ctx, g_y, x, g_x, setup);
  }
}

template <typename Tcu, typename IndexT>
void dispatch_device_cumprod_backward_accum(const Context &ctx, const Tcu *g_y,
                                            const Tcu *x, Tcu *g_x,
                                            const ScanSetup &setup,
                                            const bool accum) {
  if (accum) {
    dispatch_device_cumprod_backward_exclusive<Tcu, IndexT, true>(ctx, g_y, x,
                                                                  g_x, setup);
  } else {
    dispatch_device_cumprod_backward_exclusive<Tcu, IndexT, false>(ctx, g_y, x,
                                                                   g_x, setup);
  }
}

/**
 * @brief An interface for cumprod backward.
 */
template <typename Tcu>
void device_cumprod_backward(const Context &ctx, const Tcu *g_y, const Tcu *x,
                             Tcu *g_x, const ScanSetup &setup,
                             const bool accum) {
  if (setup.require_64bit_index) {
    dispatch_device_cumprod_backward_accum<Tcu, Size_t>(ctx, g_y, x, g_x, setup,
                                                        accum);
  } else {
    dispatch_device_cumprod_backward_accum<Tcu, int32_t>(ctx, g_y, x, g_x,
                                                         setup, accum);
  }
}
}

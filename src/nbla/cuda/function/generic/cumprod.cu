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
#include <nbla/cuda/function/cumprod.hpp>
#include <nbla/cuda/utils/atomic_min.cuh>
#include <nbla/cuda/utils/scan_ops/prod.cuh>
#include <nbla/cuda/utils/scan_ops/sum.cuh>
#include <nbla/variable.hpp>

#include <memory>

namespace nbla {

template <typename T>
void CumProdCuda<T>::setup_impl(const Variables &inputs,
                                const Variables &outputs) {
  CumProd<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  scan_setup_forward_(inputs[0]->shape(), this->axis_, this->exclusive_,
                      this->reverse_, false /* accum */);
  scan_setup_backward_ = scan_setup_forward_;
}

template <typename T>
void CumProdCuda<T>::forward_impl(const Variables &inputs,
                                  const Variables &outputs) {
  cuda_set_device(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  device_cumprod(this->ctx_, x, y, scan_setup_forward_);
}

template <typename T, typename AccumType, bool exclusive, bool reverse,
          bool accum>
__global__ void kernel_cumprod_backward(const int size0x2_, const int size1_,
                                        const int size2_, const T *x,
                                        const T *g_y, AccumType *masked_cumprod,
                                        T *g_x) {
  NBLA_CUDA_KERNEL_LOOP(idx, size0x2_) {
    const int i0 = idx / size2_;
    const int i2 = idx % size2_;

    const int offset = i0 * size1_ * size2_ + i2;

    // Create masked_cumprod
    int first_zero_pos = size1_;
    AccumType prod = (AccumType)1;
    for (int k = 0; k < size1_; k++) {
      const int i1 = reverse ? size1_ - k - 1 : k;
      int idx = i1 * size2_ + offset;
      if (x[idx] == (T)0 && first_zero_pos == size1_) {
        first_zero_pos = k;
        // prod *= (AccumType)1;
      } else {
        prod *= x[idx];
      }
      masked_cumprod[k * size2_ + offset] = prod;
    }

    // Calculate gradient
    AccumType sum = 0;
    for (int k = size1_ - 1; k >= 0; k--) {
      const int i1 = reverse ? size1_ - k - 1 : k;
      int idx = i1 * size2_ + offset;

      if (!exclusive) {
        sum += masked_cumprod[k * size2_ + offset] * g_y[idx];
      }

      T grad;
      if (k == first_zero_pos) {
        grad = (T)sum;
        sum = 0;
      } else if (k > first_zero_pos) {
        grad = (T)0;
      } else {
        grad = (T)sum / x[idx];
      }
      g_x[idx] = grad + (accum ? g_x[idx] : (T)0);

      if (exclusive && k != 0) {
        sum += masked_cumprod[(k - 1) * size2_ + offset] * g_y[idx];
      }
    }
  }
}

template <typename Tcu>
void cumprod_backward_naive(const Context &ctx, const Tcu *g_y, const Tcu *x,
                            Tcu *g_x, const ScanSetup &setup) {
  using AccumType = typename CudaTypeForceFloat<Tcu>::type;

  // `masked_cumprod` is a cumulative prod of `x` but treating the first zero
  // element as `1` on each `axis`.
  Variable v_masked_cumprod({setup.size_input});
  AccumType *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<AccumType>(ctx, true);

  auto kernel = kernel_cumprod_backward<Tcu, AccumType, true /* exclusive */,
                                        true /* reverse */, true /* accum */>;
  if (setup.exclusive) {
    if (setup.reverse) {
      kernel = setup.accum
                   ? kernel_cumprod_backward<Tcu, AccumType, true, true, true>
                   : kernel_cumprod_backward<Tcu, AccumType, true, true, false>;
    } else {
      kernel =
          setup.accum
              ? kernel_cumprod_backward<Tcu, AccumType, true, false, true>
              : kernel_cumprod_backward<Tcu, AccumType, true, false, false>;
    }
  } else {
    if (setup.reverse) {
      kernel =
          setup.accum
              ? kernel_cumprod_backward<Tcu, AccumType, false, true, true>
              : kernel_cumprod_backward<Tcu, AccumType, false, true, false>;
    } else {
      kernel =
          setup.accum
              ? kernel_cumprod_backward<Tcu, AccumType, false, false, true>
              : kernel_cumprod_backward<Tcu, AccumType, false, false, false>;
    }
  }

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, setup.size_outer * setup.size_inner,
                                 setup.size_scan, setup.size_inner, x, g_y,
                                 masked_cumprod, g_x);
}

__global__ void kernel_fill_size_scan(const int size_outer, const int size_scan,
                                      int *first_zero_index) {
  NBLA_CUDA_KERNEL_LOOP(i0, size_outer) { first_zero_index[i0] = size_scan; }
}

template <typename T, bool reverse>
__global__ void kernel_first_zero_index(const int size_input,
                                        const int size_scan, const T *x,
                                        int *first_zero_index) {
  NBLA_CUDA_KERNEL_LOOP(idx, size_input) {
    const int i0 = idx / size_scan;
    const int i1 = idx % size_scan;
    const int k = reverse ? size_scan - i1 - 1 : i1;
    if (x[idx] == (T)0) {
      atomic_min(first_zero_index + i0, k);
    }
  }
}

template <typename T, bool reverse>
__global__ void kernel_mask_input(const int size_input, const int size_scan,
                                  const T *x, const int *first_zero_index,
                                  T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size_input) {
    const int i0 = idx / size_scan;
    const int i1 = idx % size_scan;
    int z_pos = first_zero_index[i0];
    const int k = reverse ? size_scan - i1 - 1 : i1;
    if (k == z_pos) {
      y[i0 * size_scan + i1] = (T)1;
    } else {
      y[i0 * size_scan + i1] = x[i0 * size_scan + i1];
    }
  }
}

template <typename T>
__global__ void kernel_prod_dy(const int size, const T *cumprod,
                               const T *masked_cumprod, const T *dy,
                               T *out_cumprod, T *out_masked_cumprod) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    out_cumprod[idx] = cumprod[idx] * dy[idx];
    out_masked_cumprod[idx] = masked_cumprod[idx] * dy[idx];
  }
}

template <typename T, bool reverse, bool accum>
__global__ void kernel_dx(const int size_input, const int size_scan, const T *x,
                          const T *cumsum, const T *masked_cumsum,
                          const int *first_zero_index, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size_input) {
    const int i0 = idx / size_scan;
    const int i1 = idx % size_scan;
    int z_pos = first_zero_index[i0];
    const int k = reverse ? size_scan - i1 - 1 : i1;
    T grad;
    if (k < z_pos) {
      grad = cumsum[i0 * size_scan + i1] / x[i0 * size_scan + i1];
    }
    if (k == z_pos) {
      grad = masked_cumsum[i0 * size_scan + i1];
    }
    if (k > z_pos) {
      grad = (T)0;
    }
    dx[i0 * size_scan + i1] = grad + (accum ? dx[i0 * size_scan + i1] : (T)0);
  }
}

template <typename Tcu>
void cumprod_backward_parallel(const Context &ctx, const Tcu *g_y, const Tcu *x,
                               Tcu *g_x, const ScanSetup &setup) {
  using AccumType = typename CudaTypeForceFloat<Tcu>::type;

  // Step 1: Normal cumprod
  Variable v_cumprod({setup.size_input});
  Tcu *cumprod = v_cumprod.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_cumprod = setup;
  setup_cumprod.accum = false;
  device_cumprod(ctx, x, cumprod, setup_cumprod);

  // Step 2: Find first 0 index
  Variable v_first_zero_index({setup.size_outer});
  int *first_zero_index =
      v_first_zero_index.cast_data_and_get_pointer<int>(ctx, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_fill_size_scan, setup.size_outer,
                                 setup.size_scan, first_zero_index);
  {
    auto kernel = setup.reverse ? kernel_first_zero_index<Tcu, true>
                                : kernel_first_zero_index<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, setup.size_input, setup.size_scan, x,
                                   first_zero_index);
  }

  // Step 3: Mask input
  Variable v_masked_input({setup.size_input});
  Tcu *masked_input = v_masked_input.cast_data_and_get_pointer<Tcu>(ctx, true);
  {
    auto kernel = setup.reverse ? kernel_mask_input<Tcu, true>
                                : kernel_mask_input<Tcu, false>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, setup.size_input, setup.size_scan, x,
                                   first_zero_index, masked_input);
  }

  // Step 4: Masked cumprod
  Variable v_masked_cumprod({setup.size_input});
  Tcu *masked_cumprod =
      v_masked_cumprod.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_masked_cumprod = setup;
  setup_masked_cumprod.accum = false;
  device_cumprod(ctx, masked_input, masked_cumprod, setup_masked_cumprod);

  // Step 5: Prod dy to cumprod and masked_cumprod
  Variable v_cumprod_dy({setup.size_input}),
      v_masked_cumprod_dy({setup.size_input});
  Tcu *cumprod_dy = v_cumprod_dy.cast_data_and_get_pointer<Tcu>(ctx, true);
  Tcu *masked_cumprod_dy =
      v_masked_cumprod_dy.cast_data_and_get_pointer<Tcu>(ctx, true);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_prod_dy, setup.size_input, cumprod,
                                 masked_cumprod, g_y, cumprod_dy,
                                 masked_cumprod_dy);

  // Step 6: Reverse cumsum of cumprod_dy
  Variable v_cumsum({setup.size_input});
  Tcu *cumsum = v_cumsum.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_cumsum = setup;
  setup_cumsum.accum = false;
  setup_cumsum.reverse = !setup.reverse;
  device_cumsum(ctx, cumprod_dy, cumsum, setup_cumsum);

  // Step 7: Reverse cumsum of masked_cumprod_dy
  Variable v_masked_cumsum({setup.size_input});
  Tcu *masked_cumsum =
      v_masked_cumsum.cast_data_and_get_pointer<Tcu>(ctx, true);
  auto setup_masked_cumsum = setup;
  setup_masked_cumsum.accum = false;
  setup_masked_cumsum.reverse = !setup.reverse;
  device_cumsum(ctx, masked_cumprod_dy, masked_cumsum, setup_masked_cumsum);

  // Step 8: Calculate g_x
  {
    auto kernel = kernel_dx<Tcu, false /* reverse */, false /* accum */>;
    if (setup.reverse) {
      kernel = setup.accum ? kernel_dx<Tcu, true, true>
                           : kernel_dx<Tcu, true, false>;
    } else {
      kernel = setup.accum ? kernel_dx<Tcu, false, true>
                           : kernel_dx<Tcu, false, false>;
    }
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, setup.size_input, setup.size_scan, x,
                                   cumsum, masked_cumsum, first_zero_index,
                                   g_x);
  }
}

template <typename T>
void CumProdCuda<T>::backward_impl(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);

  scan_setup_backward_.accum = accum[0];

  if (scan_setup_backward_.size_inner == 1) {
    cumprod_backward_parallel(this->ctx_, g_y, x, g_x, scan_setup_backward_);
  } else {
    cumprod_backward_naive(this->ctx_, g_y, x, g_x, scan_setup_backward_);
  }
}
}
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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/math.hpp>

namespace nbla {

// Sets head pointers of matrices in mini-batch.
template <typename T>
__global__ void kernel_set_batch_pointers(int batchSize, int n, const T **ptr,
                                          const T *head) {
  NBLA_CUDA_KERNEL_LOOP(idx, batchSize) { ptr[idx] = head + idx * n * n; }
}

// A macro that creates an array of pointers of matrices.
#define NBLA_GET_BATCH_POINTERS(PTR, NAME, BATCH, CONST)                       \
  CudaCachedArray list_##PTR(sizeof(Tcu *) * BATCH, dtypes::BYTE, ctx);        \
  CONST Tcu **dev_list_##NAME =                                                \
      reinterpret_cast<CONST Tcu **>(list_##PTR.pointer<void>());              \
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_set_batch_pointers, BATCH, dim,        \
                                 (const T **)dev_list_##NAME, (const T *)PTR)

template <typename T, bool with_abs_log>
__global__ void kernel_compute_det(int batchSize, int n, T *y, const T *lu,
                                   int *pivot) {
  NBLA_CUDA_KERNEL_LOOP(idx, batchSize) {
    y[idx] = 1.0;
    int offset = idx * n * n;
    int parity = 0;
    // determinant is a product of diagonal entries of lu factorization
    for (int i = 0; i < n; ++i) {
      y[idx] *= lu[offset + i * n + i];
      if (pivot[idx * n + i] != (i + 1))
        ++parity;
    }
    // flip sign dependently on pivot array that is equal to its index
    T sign = 1.0 - 2.0 * (T)(parity % 2);
    y[idx] *= sign;
    // post operation
    if (with_abs_log) {
      y[idx] = log(abs(y[idx]));
    }
  }
}

// ----------------------------------------------------------------------
// With cublas<t>getrfBatched
// ----------------------------------------------------------------------
template <typename T, typename Tcu, bool with_abs_log>
void batch_det_forward(const Context &ctx, int device, const Variables &inputs,
                       const Variables &outputs, int dim, int batch_size) {
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(ctx);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(ctx, true);

  shared_ptr<CudaCachedArray> pivot =
      make_shared<CudaCachedArray>(dim * batch_size, dtypes::INT, ctx);
  pivot->zero();

  shared_ptr<CudaCachedArray> info =
      make_shared<CudaCachedArray>(batch_size, dtypes::INT, ctx);
  info->zero();

  shared_ptr<CudaCachedArray> lu =
      make_shared<CudaCachedArray>(inputs[0]->size(), get_dtype<Tcu>(), ctx);
  lu->copy_from(inputs[0]->data()->cast(get_dtype<Tcu>(), ctx, false));

  Tcu *lu_ptr = lu->pointer<Tcu>();
  NBLA_GET_BATCH_POINTERS(lu_ptr, lu, batch_size, ); // dev_list_lu

  // LU factorization
  cuda_getrf_batched<Tcu>(device, dim, dev_list_lu, pivot->pointer<int>(),
                          info->pointer<int>(), batch_size);

  auto kernel = kernel_compute_det<Tcu, with_abs_log>;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, batch_size, dim, y, lu_ptr,
                                 pivot->pointer<int>());
}
}

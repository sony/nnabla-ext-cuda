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

#ifndef __NBLA_CUDA_FUNCTION_UTILS_FFT_HPP__
#define __NBLA_CUDA_FUNCTION_UTILS_FFT_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <nbla/cuda/common.hpp>

#include <memory>
#include <string>
#include <unordered_map>

#include <cufft.h>
#include <cufftXt.h>

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;
using std::pair;

static const char *cufftGetErrorString(cufftResult error) {
  switch (error) {

  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:
    return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INCOMPLETE_PARAMETER_LIST:
    return "CUFFT_INCOMPLETE_PARAMETER_LIST";
  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";
  case CUFFT_PARSE_ERROR:
    return "CUFFT_PARSE_ERROR";
  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_LICENSE_ERROR:
    return "CUFFT_LICENSE_ERROR";
  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}

#define NBLA_CUFFT_CHECK(EXPRESSION)                                           \
  do {                                                                         \
    cufftResult_t ret = EXPRESSION;                                            \
    if (ret != CUFFT_SUCCESS) {                                                \
      NBLA_ERROR(error_code::target_specific,                                  \
                 "`" #EXPRESSION "` failed with %s.",                          \
                 cufftGetErrorString(ret));                                    \
    }                                                                          \
  } while (0)

template <typename T>
__global__ void kernel_add_cufft_result(const Size_t size, const T *dx_tmp,
                                        T *dx) {
  NBLA_CUDA_KERNEL_LOOP(s, size) { dx[s] += dx_tmp[s]; }
}

template <typename Tc>
__global__ void kernel_normalize_cufft_result(const int size, const float scale,
                                              Tc *x) {
  NBLA_CUDA_KERNEL_LOOP(s, size) { x[s] *= scale; }
}

template <typename T>
__global__ void kernel_fill_cufft_3d_result_with_conjugate_symmetry(
    const Size_t size, const Size_t s0, const Size_t s1, const Size_t s2,
    T *data) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    // All indices
    Size_t b_idx = idx / (s0 * s1 * s2 * 2);
    Size_t x_idx = idx / (s1 * s2 * 2) - s0 * b_idx;
    Size_t y_idx = idx / (s2 * 2) - (s0 * s1) * b_idx - s1 * x_idx;
    Size_t z_idx =
        idx / 2 - (s0 * s1 * s2) * b_idx - (s1 * s2) * x_idx - s2 * y_idx;

    // Address idx == 0
    x_idx = (x_idx == 0) ? 0 : s0 - x_idx;
    y_idx = (y_idx == 0) ? 0 : s1 - y_idx;
    z_idx = (z_idx == 0) ? 0 : s2 - z_idx;

    // fill with the complex conjugate
    Size_t m_idx = (idx / (s2 * 2)) * (s2 * 2) + s2;
    if (m_idx < idx) {
      Size_t idx_cs = (s0 * s1 * s2 * 2) * b_idx + (s1 * s2 * 2) * x_idx +
                      (s2 * 2) * y_idx + 2 * z_idx;
      if (idx % 2 == 0) {
        data[idx] = data[idx_cs];
      } else {
        data[idx] = -data[idx_cs + 1];
      }
    }
  }
}

template <typename Tcu>
void fill_cufft_result_3d_with_conjugate_symmetry(const Size_t size,
                                                  const Size_t s0,
                                                  const Size_t s1,
                                                  const Size_t s2, Tcu *data) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_fill_cufft_3d_result_with_conjugate_symmetry, size, s0, s1, s2,
      data);
}

template <typename T>
__global__ void kernel_fill_cufft_2d_result_with_conjugate_symmetry(
    const Size_t size, const Size_t s0, const Size_t s1, T *data) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    // All indices
    Size_t b_idx = idx / (s0 * s1 * 2);
    Size_t x_idx = idx / (s1 * 2) - s0 * b_idx;
    Size_t y_idx = idx / 2 - (s0 * s1) * b_idx - s1 * x_idx;

    // Address idx == 0
    x_idx = (x_idx == 0) ? 0 : s0 - x_idx;
    y_idx = (y_idx == 0) ? 0 : s1 - y_idx;

    // fill with the complex conjugate
    Size_t m_idx = (idx / (s1 * 2)) * (s1 * 2) + s1;
    if (m_idx < idx) {
      Size_t idx_cs = (s0 * s1 * 2) * b_idx + (s1 * 2) * x_idx + 2 * y_idx;
      if (idx % 2 == 0) {
        data[idx] = data[idx_cs];
      } else {
        data[idx] = -data[idx_cs + 1];
      }
    }
  }
}

template <typename Tcu>
void fill_cufft_result_2d_with_conjugate_symmetry(const Size_t size,
                                                  const Size_t s0,
                                                  const Size_t s1, Tcu *data) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_fill_cufft_2d_result_with_conjugate_symmetry, size, s0, s1, data);
}

template <typename T>
__global__ void
kernel_fill_cufft_1d_result_with_conjugate_symmetry(const Size_t size,
                                                    const Size_t s0, T *data) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    // All indices
    Size_t b_idx = idx / (s0 * 2);
    Size_t x_idx = idx / 2 - s0 * b_idx;

    // Address idx == 0
    x_idx = (x_idx == 0) ? 0 : s0 - x_idx;

    // fill with the complex conjugate
    Size_t m_idx = (idx / (s0 * 2)) * (s0 * 2) + s0;
    if (m_idx < idx) {
      Size_t idx_cs = (s0 * 2) * b_idx + 2 * x_idx;
      if (idx % 2 == 0) {
        data[idx] = data[idx_cs];
      } else {
        data[idx] = -data[idx_cs + 1];
      }
    }
  }
}

template <typename Tcu>
void fill_cufft_result_1d_with_conjugate_symmetry(const Size_t size,
                                                  const Size_t s0, Tcu *data) {
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
      kernel_fill_cufft_1d_result_with_conjugate_symmetry, size, s0, data);
}

template <typename Tc> cudaDataType_t get_cufft_dtype(bool complex);

template <> inline cudaDataType_t get_cufft_dtype<Half>(bool complex) {
  return complex ? CUDA_C_16F : CUDA_R_16F;
}
template <> inline cudaDataType_t get_cufft_dtype<HalfCuda>(bool complex) {
  return complex ? CUDA_C_16F : CUDA_R_16F;
}
template <> inline cudaDataType_t get_cufft_dtype<float>(bool complex) {
  return complex ? CUDA_C_32F : CUDA_R_32F;
}

/*
 Execute cuFFT. This is the unscaled or unnormalized FFT.
 Note: the normalization is taken care by a user of this function.
 */
template <typename Tcu>
void exec_cufft(const Context ctx, const Tcu *input_ptr, Tcu *output_ptr,
                Shape_t ishape, Shape_t oshape, cufftHandle &plan,
                bool complex_input, bool complex_output, int direction,
                std::vector<long long int> n, long long int signal_ndim) {
  // Check arguments
  NBLA_CHECK(complex_input || complex_output, error_code::value,
             "Real to Real transform does not exist.");
  // Check number of dimension of the signal
  NBLA_CHECK(signal_ndim <= 3, error_code::value,
             "Number of signal dimensions must be less than or equal to 3.");
  // Check complex input/output cases
  if (complex_input) {
    NBLA_CHECK(
        ishape.back() == 2, error_code::value,
        "The number of elements in the last dimension of the input must be 2.");
    NBLA_CHECK(
        (ishape.size() >= signal_ndim + 1), error_code::value,
        "Number of dimensions of the input must be greater than or equal to\n"
        "`signal_ndim` + 1\n"
        "at least (Signal_0, 2) shape. \n"
        "ishape.size() = %d, `signal_ndim` + 1 = %d\n",
        ishape.size(), signal_ndim + 1);
  } else { // real input
    NBLA_CHECK(
        (ishape.size() >= signal_ndim), error_code::value,
        "Number of dimensions of the input must be greater than or equal to\n"
        "`signal_ndim`\n"
        "at least (Signal_0) shape.\n"
        "ishape.size() = %d, `signal_ndim` = %d\n",
        ishape.size(), signal_ndim);
  }
  if (complex_output) {
    NBLA_CHECK(oshape.back() == 2, error_code::value,
               "The number of elements in the last dimension of the output "
               "must be 2.");
    NBLA_CHECK(
        (oshape.size() >= signal_ndim + 1), error_code::value,
        "Number of dimensions of the output must be greater than or equal to\n"
        "`signal_ndim` + 1\n"
        "at least (Signal_0, 2) shape.\n"
        "oshape.size() = %d, `signal_ndim` + 1 = %d\n",
        oshape.size(), signal_ndim + 1);
  } else { // real output
    NBLA_CHECK(
        (oshape.size() >= signal_ndim), error_code::value,
        "Number of dimensions of the output must be greater than or equal to\n"
        "`signal_ndim`\n"
        "oshape.size() = %d, `signal_ndim` = %d\n",
        oshape.size(), signal_ndim);
  }

  // rank: number of dimensions for the input signal
  int rank = signal_ndim;

  Size_t base_axis_input = complex_input ? ishape.size() - 1 - signal_ndim
                                         : ishape.size() - signal_ndim;

  // inembed: storage dimensions for the input signal
  std::vector<long long int> inembed;
  for (int i = 0; i < rank; i++) {
    // TODO: address istride for the future
    inembed.push_back(ishape[base_axis_input + i]);
  }

  // istride: input stride
  long long int istride = 1;

  // idist: distance between two consecutive input signals
  long long int idist = 1;
  for (int i = 0; i < rank; i++) {
    idist *= ishape[base_axis_input + i];
  }

  // input_type: input type
  cudaDataType_t input_type = get_cufft_dtype<Tcu>(complex_input);

  Size_t base_axis_output = complex_output ? oshape.size() - 1 - signal_ndim
                                           : oshape.size() - signal_ndim;

  // onembed: storage dimensions for the output signal
  std::vector<long long int> onembed;
  for (int i = 0; i < rank; i++) {
    // TODO: address ostride for the future
    onembed.push_back(oshape[base_axis_output + i]);
  }

  // ostride: output stride
  long long int ostride = 1;

  // odist: distance between two consecutive output signals
  long long int odist = 1;
  for (int i = 0; i < rank; i++) {
    odist *= oshape[base_axis_output + i];
  }

  // output_type: output type
  cudaDataType_t output_type = get_cufft_dtype<Tcu>(complex_output);

  // batch: batch size
  long long int batch = 1;
  for (int i = 0; i < base_axis_input; ++i) {
    batch *= ishape[i];
  }
  inembed[0] = batch; // just in case
  onembed[0] = batch; // just in case

  // work size
  size_t work_size = 0;

  // execution_type: execution type
  cudaDataType_t execution_type = get_cufft_dtype<Tcu>(true);

  // Make plan
  NBLA_CUFFT_CHECK(
      cufftSetAutoAllocation(plan, false)); // use nnabla's memory allocation
  NBLA_CUFFT_CHECK(
      cufftXtMakePlanMany(plan, rank, n.data(), inembed.data(), istride, idist,
                          input_type, onembed.data(), ostride, odist,
                          output_type, batch, &work_size, execution_type));
  shared_ptr<CudaCachedArray> arr_buff(
      new CudaCachedArray(work_size, get_dtype<unsigned char>(), ctx));
  unsigned char *buff = arr_buff->pointer<unsigned char>();
  NBLA_CUFFT_CHECK(cufftSetWorkArea(plan, (void *)buff));

  // Execute FFT
  NBLA_CUFFT_CHECK(
      cufftXtExec(plan, (void *)input_ptr, (void *)output_ptr, direction));
}
}
#endif

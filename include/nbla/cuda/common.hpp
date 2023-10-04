// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

/** Utilities for CUDA
 */
#ifndef __NBLA_CUDA_COMMON_HPP__
#define __NBLA_CUDA_COMMON_HPP__

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cutensor.h>
#if CUDA_VERSION >= 8000
#include <library_types.h>
#endif

#include <nbla/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla/exception.hpp>

#include <nbla/cuda/half.hpp>

#include <map>

namespace nbla {

#define STRINGIFY(str) #str
#if CUDA_VERSION >= 11010
#ifndef _MSC_VER
#define NBLA_DIAG_SUPPRESS(warn) _Pragma(STRINGIFY(nv_diag_suppress warn))
#define NBLA_DIAG_DEFAULT(warn) _Pragma(STRINGIFY(nv_diag_default warn))
#else
#define NBLA_DIAG_SUPPRESS(warn)
#define NBLA_DIAG_DEFAULT(warn)
#endif
#define inline_qualifier_ignored 20050
#else
#ifndef _MSC_VER
#define NBLA_DIAG_SUPPRESS(warn) _Pragma(STRINGIFY(diag_suppress warn))
#define NBLA_DIAG_DEFAULT(warn) _Pragma(STRINGIFY(diag_default warn))
#else
#define NBLA_DIAG_SUPPRESS(warn)
#define NBLA_DIAG_DEFAULT(warn)
#endif
#define inline_qualifier_ignored 3095
#endif

namespace cuda {
typedef int Index_t;
}

using std::map;

/** Check Kernel Execution*/
#define NBLA_CUDA_KERNEL_CHECK() NBLA_CUDA_CHECK(cudaGetLastError())

/**
Check CUDA error for synchronous call
cudaGetLastError is used to clear previous error happening at "condition".
*/
#define NBLA_CUDA_CHECK(condition)                                             \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      cudaGetLastError();                                                      \
      NBLA_ERROR(error_code::target_specific, "(%s) failed with \"%s\" (%s).", \
                 #condition, cudaGetErrorString(error),                        \
                 cudaGetErrorName(error));                                     \
    }                                                                          \
  }

#define NBLA_CUDA_FORCE_ASSERT(condition)                                      \
  {                                                                            \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      cudaGetLastError();                                                      \
      NBLA_FORCE_ASSERT(error != cudaSuccess, "(%s) failed with \"%s\" (%s).", \
                        #condition, cudaGetErrorString(error),                 \
                        cudaGetErrorName(error))                               \
    }                                                                          \
  }

/**
Check CUDA driver error.
*/
#define NBLA_CUDA_DRIVER_CHECK(condition)                                      \
  {                                                                            \
    CUresult status = (condition);                                             \
    if (status != CUDA_SUCCESS) {                                              \
      const char *err_name, *err_str;                                          \
      cuGetErrorName(status, &err_name);                                       \
      cuGetErrorString(status, &err_str);                                      \
      NBLA_ERROR(error_code::target_specific, "(%s) failed with \"%s\" (%s).", \
                 #condition, err_str, err_name);                               \
    }                                                                          \
  }

#define NBLA_CUDA_DRIVER_FORCE_ASSERT(condition)                               \
  {                                                                            \
    CUresult status = (condition);                                             \
    if (status != CUDA_SUCCESS) {                                              \
      const char *err_name, *err_str;                                          \
      cuGetErrorName(status, &err_name);                                       \
      cuGetErrorString(status, &err_str);                                      \
      NBLA_FORCE_ASSERT(status != CUDA_SUCCESS,                                \
                        "(%s) failed with \"%s\" (%s).", #condition, err_str,  \
                        err_name);                                             \
    }                                                                          \
  }

inline string cublas_status_to_string(cublasStatus_t status) {
#define CASE_CUBLAS_STATUS(NAME)                                               \
  case CUBLAS_STATUS_##NAME:                                                   \
    return #NAME;

  switch (status) {
    CASE_CUBLAS_STATUS(SUCCESS);
    CASE_CUBLAS_STATUS(NOT_INITIALIZED);
    CASE_CUBLAS_STATUS(ALLOC_FAILED);
    CASE_CUBLAS_STATUS(INVALID_VALUE);
    CASE_CUBLAS_STATUS(ARCH_MISMATCH);
    CASE_CUBLAS_STATUS(MAPPING_ERROR);
    CASE_CUBLAS_STATUS(EXECUTION_FAILED);
    CASE_CUBLAS_STATUS(INTERNAL_ERROR);
#if CUDA_VERSION >= 6000
    CASE_CUBLAS_STATUS(NOT_SUPPORTED);
#endif
#if CUDA_VERSION >= 6050
    CASE_CUBLAS_STATUS(LICENSE_ERROR);
#endif
  }
  return "UNKNOWN";
#undef CASE_CUBLAS_STATUS
}

/**
 */
#define NBLA_CUBLAS_CHECK(condition)                                           \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    cudaGetLastError();                                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      NBLA_ERROR(                                                              \
          error_code::target_specific,                                         \
          string("CUBLAS_STATUS_") + cublas_status_to_string(status) +         \
              string(                                                          \
                  " occured in `" #condition                                   \
                  "`. Please see CUBLAS API documentation for the cause."));   \
    }                                                                          \
  }

#define NBLA_CUBLAS_FORCE_ASSERT(condition)                                    \
  {                                                                            \
    cublasStatus_t status = condition;                                         \
    cudaGetLastError();                                                        \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      NBLA_FORCE_ASSERT(                                                       \
          status != CUBLAS_STATUS_SUCCESS,                                     \
          string("CUBLAS_STATUS_") + cublas_status_to_string(status) +         \
              string(                                                          \
                  " occured in `" #condition                                   \
                  "`. Please see CUBLAS API documentation for the cause."));   \
    }                                                                          \
  }

inline string cusolver_status_to_string(cusolverStatus_t status) {
#define CASE_CUSOLVER_STATUS(NAME)                                             \
  case CUSOLVER_STATUS_##NAME:                                                 \
    return #NAME;

  switch (status) {
    CASE_CUSOLVER_STATUS(SUCCESS);
    CASE_CUSOLVER_STATUS(NOT_INITIALIZED);
    CASE_CUSOLVER_STATUS(ALLOC_FAILED);
    CASE_CUSOLVER_STATUS(INVALID_VALUE);
    CASE_CUSOLVER_STATUS(ARCH_MISMATCH);
    CASE_CUSOLVER_STATUS(MAPPING_ERROR);
    CASE_CUSOLVER_STATUS(EXECUTION_FAILED);
    CASE_CUSOLVER_STATUS(INTERNAL_ERROR);
    CASE_CUSOLVER_STATUS(MATRIX_TYPE_NOT_SUPPORTED);
    CASE_CUSOLVER_STATUS(NOT_SUPPORTED);
    CASE_CUSOLVER_STATUS(ZERO_PIVOT);
    CASE_CUSOLVER_STATUS(INVALID_LICENSE);
#if CUDA_VERSION >= 10020
    CASE_CUSOLVER_STATUS(IRS_PARAMS_NOT_INITIALIZED);
    CASE_CUSOLVER_STATUS(IRS_PARAMS_INVALID);
    CASE_CUSOLVER_STATUS(IRS_INTERNAL_ERROR);
    CASE_CUSOLVER_STATUS(IRS_NOT_SUPPORTED);
    CASE_CUSOLVER_STATUS(IRS_OUT_OF_RANGE);
    CASE_CUSOLVER_STATUS(IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES);
    CASE_CUSOLVER_STATUS(IRS_INFOS_NOT_INITIALIZED);
#endif
#if CUDA_VERSION >= 11000
    CASE_CUSOLVER_STATUS(IRS_PARAMS_INVALID_PREC);
    CASE_CUSOLVER_STATUS(IRS_PARAMS_INVALID_REFINE);
    CASE_CUSOLVER_STATUS(IRS_PARAMS_INVALID_MAXITER);
    CASE_CUSOLVER_STATUS(IRS_INFOS_NOT_DESTROYED);
    CASE_CUSOLVER_STATUS(IRS_MATRIX_SINGULAR);
    CASE_CUSOLVER_STATUS(INVALID_WORKSPACE);
#endif
  }
  return "UNKNOWN";
#undef CASE_CUSOLVER_STATUS
}

#define NBLA_CUSOLVER_CHECK(condition)                                         \
  {                                                                            \
    cusolverStatus_t status = condition;                                       \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      NBLA_ERROR(                                                              \
          error_code::target_specific,                                         \
          string("CUSOLVER_STATUS_") + cusolver_status_to_string(status) +     \
              string(                                                          \
                  " occured in `" #condition                                   \
                  "`. Please see CUSOLVER API documentation for the cause.")); \
    }                                                                          \
  }

#define NBLA_CUSOLVER_FORCE_ASSERT(condition)                                  \
  {                                                                            \
    cusolverStatus_t status = condition;                                       \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      NBLA_FORCE_ASSERT(                                                       \
          status != CUSOLVER_STATUS_SUCCESS,                                   \
          string("CUSOLVER_STATUS_") + cusolver_status_to_string(status) +     \
              string(                                                          \
                  " occured in `" #condition                                   \
                  "`. Please see CUSOLVER API documentation for the cause.")); \
    }                                                                          \
  }

inline string curand_status_to_string(curandStatus_t status) {
#define CASE_CURAND_STATUS(NAME)                                               \
  case CURAND_STATUS_##NAME:                                                   \
    return #NAME;

  switch (status) {
    CASE_CURAND_STATUS(SUCCESS);
    CASE_CURAND_STATUS(VERSION_MISMATCH);
    CASE_CURAND_STATUS(NOT_INITIALIZED);
    CASE_CURAND_STATUS(ALLOCATION_FAILED);
    CASE_CURAND_STATUS(TYPE_ERROR);
    CASE_CURAND_STATUS(OUT_OF_RANGE);
    CASE_CURAND_STATUS(LENGTH_NOT_MULTIPLE);
    CASE_CURAND_STATUS(DOUBLE_PRECISION_REQUIRED);
    CASE_CURAND_STATUS(LAUNCH_FAILURE);
    CASE_CURAND_STATUS(PREEXISTING_FAILURE);
    CASE_CURAND_STATUS(INITIALIZATION_FAILED);
    CASE_CURAND_STATUS(ARCH_MISMATCH);
    CASE_CURAND_STATUS(INTERNAL_ERROR);
  }
  return "UNKNOWN";
#undef CASE_CURAND_STATUS
}

#define NBLA_CURAND_CHECK(condition)                                           \
  {                                                                            \
    curandStatus_t status = condition;                                         \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      NBLA_ERROR(                                                              \
          error_code::target_specific,                                         \
          string("CUBLAS_STATUS_") + curand_status_to_string(status) +         \
              string(                                                          \
                  " occured in `" #condition                                   \
                  "`. Please see CUSOLVER API documentation for the cause.")); \
    }                                                                          \
  }

#define NBLA_CUTENSOR_CHECK(condition)                                         \
  {                                                                            \
    cutensorStatus_t status = condition;                                       \
    NBLA_CHECK(status == CUTENSOR_STATUS_SUCCESS, error_code::target_specific, \
               cutensorGetErrorString(status));                                \
  };

/** Data type */
#if CUDA_VERSION >= 8000
template <typename T> struct cuda_data_type;
#define CUDA_TYPE_T(TYPE, ENUM)                                                \
  template <> struct cuda_data_type<TYPE> {                                    \
    static cudaDataType_t type() { return CUDA_##ENUM; }                       \
  }
CUDA_TYPE_T(double, R_64F);
CUDA_TYPE_T(float, R_32F);
CUDA_TYPE_T(half, R_16F);
CUDA_TYPE_T(Half, R_16F);
CUDA_TYPE_T(HalfCuda, R_16F);
CUDA_TYPE_T(uint8_t, R_8U);
CUDA_TYPE_T(int8_t, R_8I);
#undef CUDA_TYPE_T

#else // CUDA_VERSION >= 8000
template <typename T> struct cuda_data_type;
#define CUBLAS_TYPE_T(TYPE, UTYPE)                                             \
  template <> struct cuda_data_type<TYPE> {                                    \
    static cublasDataType_t type() { return CUBLAS_DATA_##UTYPE; }             \
  }
CUBLAS_TYPE_T(double, DOUBLE);
CUBLAS_TYPE_T(float, FLOAT);
CUBLAS_TYPE_T(half, HALF);
CUBLAS_TYPE_T(Half, HALF);
CUBLAS_TYPE_T(HalfCuda, HALF);
#undef CUBLAS_TYPE_T
#endif

enum {
  CUDA_WARP_SIZE = 32,
  CUDA_WARP_MASK = 0x1f,
  CUDA_WARP_BITS = 5,
};

/** ceil(N/D) where N and D are integers */
#define NBLA_CEIL_INT_DIV(N, D)                                                \
  ((static_cast<int>(N) + static_cast<int>(D) - 1) / static_cast<int>(D))

/** ceil(N/D) where N and D are nbla::Size)t integers */
#define NBLA_CEIL_SIZE_T_DIV(N, D)                                             \
  ((static_cast<Size_t>(N) + static_cast<Size_t>(D) - 1) /                     \
   static_cast<Size_t>(D))

/** Default num threads */
#define NBLA_CUDA_NUM_THREADS 512

/** Max number of blocks per dimension*/
#define NBLA_CUDA_MAX_BLOCKS 65536

/** Block size */
#define NBLA_CUDA_GET_BLOCKS(num) NBLA_CEIL_INT_DIV(num, NBLA_CUDA_NUM_THREADS)

/** Block size with nbla::Size_t */
#define NBLA_CUDA_GET_BLOCKS_SIZE_T(num)                                       \
  NBLA_CEIL_SIZE_T_DIV(num, NBLA_CUDA_NUM_THREADS)

/** Get an appropriate block size given a size of elements.

    The kernel is assumed to contain a grid-strided loop.
 */
inline int cuda_get_blocks_by_size(int size) {
  if (size == 0)
    return 0;
  const int blocks = NBLA_CUDA_GET_BLOCKS(size);
  const int inkernel_loop = NBLA_CEIL_INT_DIV(blocks, NBLA_CUDA_MAX_BLOCKS);
  const int total_blocks = NBLA_CEIL_INT_DIV(blocks, inkernel_loop);
  return total_blocks;
}

/** Get an appropriate block size given a size of elements with nbla::Size_t.

    The kernel is assumed to contain a grid-strided loop.
*/
inline Size_t cuda_get_blocks_by_size_with_size_t(const Size_t size) {
  if (size == 0)
    return 0;
  const Size_t blocks = NBLA_CUDA_GET_BLOCKS_SIZE_T(size);
  const Size_t inkernel_loop =
      NBLA_CEIL_SIZE_T_DIV(blocks, NBLA_CUDA_MAX_BLOCKS);
  const Size_t total_blocks = NBLA_CEIL_SIZE_T_DIV(blocks, inkernel_loop);
  return total_blocks;
}

/** Launch simple kernel */
#define NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, size, ...)                      \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), NBLA_CUDA_NUM_THREADS>>>(        \
        (size), __VA_ARGS__);                                                  \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Launch simple kernel */
#define NBLA_CUDA_LAUNCH_KERNEL_SIMPLE_SIZE_T(kernel, size, ...)               \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size_with_size_t(size),                      \
               NBLA_CUDA_NUM_THREADS>>>((size), __VA_ARGS__);                  \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Launch simple kernel */
#define NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel, stream, size, ...)           \
  {                                                                            \
    (kernel)<<<cuda_get_blocks_by_size(size), NBLA_CUDA_NUM_THREADS, 0,        \
               (stream)>>>((size), __VA_ARGS__);                               \
    NBLA_CUDA_KERNEL_CHECK();                                                  \
  }

/** Cuda grid-strided loop */
#define NBLA_CUDA_KERNEL_LOOP(idx, num)                                        \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (num);           \
       idx += blockDim.x * gridDim.x)

/** Cuda grid-strided loop */
#define NBLA_CUDA_KERNEL_LOOP_SIZE_T(idx, num)                                 \
  for (Size_t idx =                                                            \
           static_cast<Size_t>(blockIdx.x) * static_cast<Size_t>(blockDim.x) + \
           static_cast<Size_t>(threadIdx.x);                                   \
       idx < (num); idx += static_cast<Size_t>(blockDim.x) *                   \
                           static_cast<Size_t>(gridDim.x))

/** Instantiate template CUDA functions */
#define NBLA_INSTANTIATE_CUDA_FUNCS(type, classname)                           \
  template void classname<type>::forward_impl(const Variables &inputs,         \
                                              const Variables &outputs);       \
  template void classname<type>::backward_impl(                                \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down);

/** CUDA device setter
@return index of device before change
*/
int cuda_set_device(int device);
/** Get current CUDA device.
@return index of device
*/
int cuda_get_device();

/** Get free and total device memory size.
 */
NBLA_CUDA_API vector<size_t> cuda_mem_get_info();

/** Get device properties of current CUDA device.

    Note that using `cuda_get_current_device_properties`
    is extremely slower than `cuda_get_current_device_attribute`,
    since some props require PCIe reads to query.
    Keep in mind that sometime using this function could lead to huge
    slowdowns in your implementation.
 */
[[deprecated("Use SingletonManager::get<Cuda>()->get_device_properties() "
             "instead.")]] cudaDeviceProp
cuda_get_current_device_properties();

int cuda_get_current_device_attribute(cudaDeviceAttr attr);
} // namespace nbla
#endif

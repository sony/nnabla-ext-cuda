// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cublas.hpp>

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace nbla {

#if CUDA_VERSION >= 9000
// Get preferred cuBLAS GEMM algorithm. Only if data type is half, the use of
// Tensor Core is allowed.
inline cublasGemmAlgo_t infer_gemm_algo_by_type(cudaDataType_t dt) {
  return dt == CUDA_R_16F ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
}
#endif

// ----------------------------------------------------------------------
// Gemm
// ----------------------------------------------------------------------
template <>
void cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t op_x,
                         cublasOperation_t op_y, int m, int n, int k,
                         float alpha, const double *x, int lda, const double *y,
                         int ldb, float beta, double *z, int ldc) {
  double a = alpha;
  double b = beta;
  NBLA_CUBLAS_CHECK(
      cublasDgemm(handle, op_x, op_y, m, n, k, &a, x, lda, y, ldb, &b, z, ldc));
}

template <>
void cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t op_x,
                        cublasOperation_t op_y, int m, int n, int k,
                        float alpha, const float *x, int lda, const float *y,
                        int ldb, float beta, float *z, int ldc) {
  float a = alpha;
  float b = beta;
  auto dt = cuda_data_type<float>::type();
  /* NOTE: cublasSgemmEx has been added since CUDA 7.5 */
  NBLA_CUBLAS_CHECK(cublasSgemmEx(handle, op_x, op_y, m, n, k, &a, x, dt, lda,
                                  y, dt, ldb, &b, z, dt, ldc));
}

template <>
void cublas_gemm<half>(cublasHandle_t handle, cublasOperation_t op_x,
                       cublasOperation_t op_y, int m, int n, int k, float alpha,
                       const half *x, int lda, const half *y, int ldb,
                       float beta, half *z, int ldc) {
  float a = alpha;
  float b = beta;
  auto dt = cuda_data_type<half>::type();
#if CUDA_VERSION >= 9000
  // cublasGemmEx can only be used on architectures later than Kepler (>=5).
  cudaDeviceProp prop = cuda_get_current_device_properties();
  if (prop.major >= 5) {
    auto ct = cuda_data_type<typename CudaTypeForceFloat<half>::type>::type();
    NBLA_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    NBLA_CUBLAS_CHECK(cublasGemmEx(handle, op_x, op_y, m, n, k, &a, x, dt, lda,
                                   y, dt, ldb, &b, z, dt, ldc, ct,
                                   infer_gemm_algo_by_type(dt)));
    NBLA_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    NBLA_CUBLAS_CHECK(cublasSgemmEx(handle, op_x, op_y, m, n, k, &a, x, dt, lda,
                                    y, dt, ldb, &b, z, dt, ldc));
  }
#else // CUDA_VERSION < 9000
  NBLA_CUBLAS_CHECK(cublasSgemmEx(handle, op_x, op_y, m, n, k, &a, x, dt, lda,
                                  y, dt, ldb, &b, z, dt, ldc));
#endif
}

// ----------------------------------------------------------------------
// Gemv
// ----------------------------------------------------------------------
#define DEF_CUBLAS_GEMV(TYPE, PREFIX)                                          \
  template <>                                                                  \
  void cublas_gemv<TYPE>(cublasHandle_t handle, cublasOperation_t trans,       \
                         int m, int n, float alpha, const TYPE *A, int lda,    \
                         const TYPE *x, int incx, float beta, TYPE *y,         \
                         int incy) {                                           \
    typedef typename CudaTypeForceFloat<TYPE>::type Tf;                        \
    Tf a = alpha;                                                              \
    Tf b = beta;                                                               \
    NBLA_CUBLAS_CHECK(cublas##PREFIX##gemv(handle, trans, m, n, &a, A, lda, x, \
                                           incx, &b, y, incy));                \
  }
// NOTE: We don't instantiate half because gemv is not provided in cuBLAS.
// We use gemm instead of gemv as defined in math.hpp
DEF_CUBLAS_GEMV(float, S);
DEF_CUBLAS_GEMV(double, D);

// ----------------------------------------------------------------------
// Dot
// ----------------------------------------------------------------------
#if CUDA_VERSION >= 8000
template <typename T>
void cublas_dot(cublasHandle_t handle, int n, const T *x, int incx, const T *y,
                int incy, T *out) {
  cudaDataType_t dt = cuda_data_type<T>::type();
  cudaDataType_t ct =
      cuda_data_type<typename CudaTypeForceFloat<T>::type>::type();
  NBLA_CUBLAS_CHECK(
      cublasDotEx(handle, n, x, dt, incx, y, dt, incy, out, dt, ct));
}
#define DEF_CUBLAS_DOT(TYPE)                                                   \
  template void cublas_dot(cublasHandle_t handle, int n, const TYPE *x,        \
                           int incx, const TYPE *y, int incy, TYPE *out)
DEF_CUBLAS_DOT(half);
DEF_CUBLAS_DOT(float);
DEF_CUBLAS_DOT(double);
#else // CUDA_VERSION < 8000
template <>
void cublas_dot<double>(cublasHandle_t handle, int n, const double *x, int incx,
                        const double *y, int incy, double *out) {
  NBLA_CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, out));
}
template <>
void cublas_dot<float>(cublasHandle_t handle, int n, const float *x, int incx,
                       const float *y, int incy, float *out) {
  NBLA_CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, out));
}
template <>
void cublas_dot<half>(cublasHandle_t handle, int n, const half *x, int incx,
                      const half *y, int incy, half *out) {
  NBLA_CHECK(incx == 1 && incy == 1, error_code::value,
             "cublas_dot with half precision in CUDA<8 only accepts both "
             "incx and incy == 1.");
  // Because no hdot support when CUDA<8 falling back to GEMM.
  cublas_gemm<half>(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, 1, x, 1, y, n, 0,
                    out, 1);
}
#endif

// ----------------------------------------------------------------------
// Gemm batched
// ----------------------------------------------------------------------
#if CUDA_VERSION >= 9010
template <typename T>
void cublas_gemm_batched(cublasHandle_t handle, cublasOperation_t op_x,
                         cublasOperation_t op_y, int m, int n, int k,
                         float alpha, const T **x, int lda, const T **y,
                         int ldb, float beta, T **z, int ldc, int batch_count) {
  typedef typename CudaTypeForceFloat<float>::type Tf;
  Tf a = alpha;
  Tf b = beta;
  cudaDataType_t dt = cuda_data_type<T>::type();
  cudaDataType_t ct =
      cuda_data_type<typename CudaTypeForceFloat<T>::type>::type();
  NBLA_CUBLAS_CHECK(
      cublasGemmBatchedEx(handle, op_x, op_y, m, n, k, &a, (const void **)x, dt,
                          lda, (const void **)y, dt, ldb, &b, (void **)z, dt,
                          ldc, batch_count, ct, infer_gemm_algo_by_type(dt)));
}
#define DEF_CUBLAS_GEMM_BATCHED(TYPE)                                          \
  template void cublas_gemm_batched(                                           \
      cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,   \
      int m, int n, int k, float alpha, const TYPE **x, int lda,               \
      const TYPE **y, int ldb, float beta, TYPE **z, int ldc, int batch_count)
DEF_CUBLAS_GEMM_BATCHED(half);
DEF_CUBLAS_GEMM_BATCHED(float);
DEF_CUBLAS_GEMM_BATCHED(double);
#else  // CUDA_VERSION < 9010
template <>
void cublas_gemm_batched<double>(cublasHandle_t handle, cublasOperation_t op_x,
                                 cublasOperation_t op_y, int m, int n, int k,
                                 float alpha, const double **x, int lda,
                                 const double **y, int ldb, float beta,
                                 double **z, int ldc, int batch_count) {
  double a = alpha;
  double b = beta;
  NBLA_CUBLAS_CHECK(cublasDgemmBatched(handle, op_x, op_y, m, n, k, &a, x, lda,
                                       y, ldb, &b, z, ldc, batch_count));
}
template <>
void cublas_gemm_batched<float>(cublasHandle_t handle, cublasOperation_t op_x,
                                cublasOperation_t op_y, int m, int n, int k,
                                float alpha, const float **x, int lda,
                                const float **y, int ldb, float beta, float **z,
                                int ldc, int batch_count) {
  float a = alpha;
  float b = beta;
  NBLA_CUBLAS_CHECK(cublasSgemmBatched(handle, op_x, op_y, m, n, k, &a, x, lda,
                                       y, ldb, &b, z, ldc, batch_count));
}
template <>
void cublas_gemm_batched<half>(cublasHandle_t handle, cublasOperation_t op_x,
                               cublasOperation_t op_y, int m, int n, int k,
                               float alpha, const half **x, int lda,
                               const half **y, int ldb, float beta, half **z,
                               int ldc, int batch_count) {

  // No HgemmBatched with fp32 computation. Falling back to multiple calls of
  // Gemm.
  for (int b = 0; b < batch_count; b++) {
    auto x_ = x[b];
    auto y_ = y[b];
    auto z_ = z[b];
    cublas_gemm<half>(handle, op_x, op_y, m, n, k, alpha, x_, lda, y_, ldb,
                      beta, z_, ldc);
  }
}
#endif // CUDA_VERSION < 9010

// ----------------------------------------------------------------------
// Gemm strided batched
// ----------------------------------------------------------------------
#if CUDA_VERSION >= 8000
template <>
void cublas_gemm_strided_batched<double>(
    cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,
    int m, int n, int k, float alpha, const double *x, int lda, int stride_a,
    const double *y, int ldb, int stride_b, float beta, double *z, int ldc,
    int stride_c, int batch_count) {
  double a = alpha;
  double b = beta;
  NBLA_CUBLAS_CHECK(cublasDgemmStridedBatched(
      handle, op_x, op_y, m, n, k, &a, x, lda, stride_a, y, ldb, stride_b, &b,
      z, ldc, stride_c, batch_count));
}
template <>
void cublas_gemm_strided_batched<float>(
    cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,
    int m, int n, int k, float alpha, const float *x, int lda, int stride_a,
    const float *y, int ldb, int stride_b, float beta, float *z, int ldc,
    int stride_c, int batch_count) {
  float a = alpha;
  float b = beta;
  NBLA_CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, op_x, op_y, m, n, k, &a, x, lda, stride_a, y, ldb, stride_b, &b,
      z, ldc, stride_c, batch_count));
}

#if CUDA_VERSION >= 9010
template <typename T, int batch_chunk>
void cublas_gemm_strided_batched_chunk(cublasHandle_t handle,
                                       cublasOperation_t op_x,
                                       cublasOperation_t op_y, int m, int n,
                                       int k, float alpha, const T *x, int lda,
                                       int stride_a, const T *y, int ldb,
                                       int stride_b, float beta, T *z, int ldc,
                                       int stride_c, int batch_count) {
  typedef typename CudaTypeForceFloat<T>::type Tf;
  Tf a = alpha;
  Tf b = beta;
  cudaDataType_t dt = cuda_data_type<T>::type();
  cudaDataType_t ct =
      cuda_data_type<typename CudaTypeForceFloat<T>::type>::type();
  int chunks = NBLA_CEIL_INT_DIV(batch_count, batch_chunk);
  for (int i = 0; i < chunks; i++) {
    const T *x_i = x + i * batch_chunk * stride_a;
    const T *y_i = y + i * batch_chunk * stride_b;
    T *z_i = z + i * batch_chunk * stride_c;
    int batch_count_chunk =
        std::min(batch_chunk, batch_count - i * batch_chunk);
    NBLA_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, op_x, op_y, m, n, k, &a, x_i, dt, lda, stride_a, y_i, dt, ldb,
        stride_b, &b, z_i, dt, ldc, stride_c, batch_count_chunk, ct,
        infer_gemm_algo_by_type(dt)));
  }
}
#endif // CUDA_VERSION >= 9010

template <>
void cublas_gemm_strided_batched<half>(
    cublasHandle_t handle, cublasOperation_t op_x, cublasOperation_t op_y,
    int m, int n, int k, float alpha, const half *x, int lda, int stride_a,
    const half *y, int ldb, int stride_b, float beta, half *z, int ldc,
    int stride_c, int batch_count) {
#if CUDA_VERSION >= 9010
  cudaDeviceProp prop = cuda_get_current_device_properties();
  if (prop.major >= 5) {
    constexpr int batch_chunk = 1 << 15;
    if (batch_count > batch_chunk) {
      // Seems like cublasGemmStridedBatchedEx does not allow a large batch
      // count.
      // We confirmed that CUDA 9.1 and 9.2.
      // If batch_count > 1<<15, we apply batched gemm for each chunk.
      // TODO: Check a behavior of the newer versions.
      cublas_gemm_strided_batched_chunk<half, batch_chunk>(
          handle, op_x, op_y, m, n, k, alpha, x, lda, stride_a, y, ldb,
          stride_b, beta, z, ldc, stride_c, batch_count);
      return;
    }
    float a = alpha;
    float b = beta;
    cudaDataType_t dt = cuda_data_type<half>::type();
    cudaDataType_t ct = cuda_data_type<float>::type();
    NBLA_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    NBLA_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, op_x, op_y, m, n, k, &a, x, dt, lda, stride_a, y, dt, ldb,
        stride_b, &b, z, dt, ldc, stride_c, batch_count, ct,
        infer_gemm_algo_by_type(dt)));
    NBLA_CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    return;
  }
#endif // CUDA_VERSION >= 9010
  // No HgemmStridedBatched with fp32 computation. Falling back to multiple
  // calls of Gemm.
  for (int b = 0; b < batch_count; b++) {
    auto x_ = x + b * stride_a;
    auto y_ = y + b * stride_b;
    auto z_ = z + b * stride_c;
    cublas_gemm<half>(handle, op_x, op_y, m, n, k, alpha, x_, lda, y_, ldb,
                      beta, z_, ldc);
  }
}
#endif // CUDA_VERSION >= 8000

// ----------------------------------------------------------------------
// Getrf batched
// ----------------------------------------------------------------------
template <>
void cublas_getrf_batched<double>(cublasHandle_t handle, int n, double **x,
                                  int lda, int *pivot, int *info,
                                  int batchSize) {
  NBLA_CUBLAS_CHECK(
      cublasDgetrfBatched(handle, n, x, lda, pivot, info, batchSize));
}
template <>
void cublas_getrf_batched<float>(cublasHandle_t handle, int n, float **x,
                                 int lda, int *pivot, int *info,
                                 int batchSize) {
  NBLA_CUBLAS_CHECK(
      cublasSgetrfBatched(handle, n, x, lda, pivot, info, batchSize));
}

// ----------------------------------------------------------------------
// Getri batched
// ----------------------------------------------------------------------
template <>
void cublas_getri_batched<double>(cublasHandle_t handle, int n,
                                  const double **x, int lda, int *pivot,
                                  double **y, int ldc, int *info,
                                  int batchSize) {
  NBLA_CUBLAS_CHECK(
      cublasDgetriBatched(handle, n, x, lda, pivot, y, ldc, info, batchSize));
}
template <>
void cublas_getri_batched<float>(cublasHandle_t handle, int n, const float **x,
                                 int lda, int *pivot, float **y, int ldc,
                                 int *info, int batchSize) {
  NBLA_CUBLAS_CHECK(
      cublasSgetriBatched(handle, n, x, lda, pivot, y, ldc, info, batchSize));
}
}

#include "gtest/gtest.h"
#include <cudnn.h>
#include <exception>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

namespace nbla {
cudnnStatus_t FakeCudnnFunc(void) { return CUDNN_STATUS_ALLOC_FAILED; }

cublasStatus_t FakeCublasFunc(void) { return CUBLAS_STATUS_ALLOC_FAILED; }

cusolverStatus_t FakeCusolverFunc(void) { return CUSOLVER_STATUS_ALLOC_FAILED; }

curandStatus_t FakeCurandFunc(void) { return CURAND_STATUS_ALLOCATION_FAILED; }

TEST(NBLA_CUDNN_CHECK_Test, CheckCudnnFailed) {
  try {
    NBLA_CUDNN_CHECK(FakeCudnnFunc());
    EXPECT_TRUE(false);
  } catch (const std::exception &e) {
    std::cout << "Caught exception \"" << e.what() << "\"\n";
  }
}

TEST(NBLA_CUBLAS_CHECK_Test, CheckCublasFailed) {
  try {
    NBLA_CUBLAS_CHECK(FakeCublasFunc());
    EXPECT_TRUE(false);
  } catch (const std::exception &e) {
    std::cout << "Caught exception \"" << e.what() << "\"\n";
  }
}

TEST(NBLA_CUSOLVER_CHECK_Test, CheckCusolverFailed) {
  try {
    NBLA_CUSOLVER_CHECK(FakeCusolverFunc());
    EXPECT_TRUE(false);
  } catch (const std::exception &e) {
    std::cout << "Caught exception \"" << e.what() << "\"\n";
  }
}

TEST(NBLA_CURAND_CHECK_Test, CheckCurandFailed) {
  try {
    NBLA_CURAND_CHECK(FakeCurandFunc());
    EXPECT_TRUE(false);
  } catch (const std::exception &e) {
    std::cout << "Caught exception \"" << e.what() << "\"\n";
  }
}
} // namespace nbla

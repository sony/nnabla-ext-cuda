# Find cUDNN, NVIDIA Deep Learning core computation library for CUDA.
# Returns:
#   CUDNN_INCLUDE_DIRS
#   CUDNN_LIBRARIES

include(FindPackageHandleStandardArgs)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    PATHS
    $ENV{CUDNN_PATH}/include
    $ENV{CUDA_PATH}/include
    /usr/local/cuda/include
    /usr/include
    )

find_library(CUDNN_LIBRARY cudnn
    PATHS
    $ENV{CUDNN_PATH}/lib64
    $ENV{CUDA_PATH}/lib/x64
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
    )

find_package_handle_standard_args(
    cuDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

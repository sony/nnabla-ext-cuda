set(CUDA_SELECT_NVCC_ARCH_ARG "Auto" CACHE STRING "Select NVCC Arch. Options: Common | All | LIST(ARCH_AND_PTX ...). See select_compute_arch.cmake for details.")

find_package(CUDA REQUIRED)
message("-- CUDA--")
message("Version: " ${CUDA_VERSION})
message("Runtime: " ${CUDA_CUDART_LIBRARY})
message("CUBLAS: " ${CUDA_CUBLAS_LIBRARIES})
message("CURAND: " ${CUDA_curand_LIBRARY})
message("CUFFT: " ${CUDA_CUFFT_LIBRARIES})
set(NBLA_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
set(NBLA_CUDA_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
list(APPEND NBLA_CUDA_LINKER_LIBS
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  )
  
# Distributed Training  
if(DEFINED ENV{NCCL_HOME})
	list(APPEND NBLA_CUDA_INCLUDE_DIRS 
		$ENV{NCCL_HOME}/build/include)
	list(APPEND NBLA_CUDA_INCLUDE_DIRS 
		$ENV{MPI_HOME}/include)
	list(APPEND NBLA_CUDA_LINKER_LIBS
		$ENV{NCCL_HOME}/build/lib/libnccl.so)
endif()

####################################################################################################
# cuDNN

find_package(cuDNN REQUIRED)
# Get CuDNN version
set(file ${PROJECT_BINARY_DIR}/detect_cudnn_version.cpp)
file(WRITE ${file} ""
  "#include <cudnn.h>\n"
  "#include <cstdio>\n"
  "int main()\n"
  "{\n"
  "  std::printf(\"%d\", CUDNN_VERSION / 100);\n"
  "  return 0;\n"
  "}\n")
try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
  LINK_LIBRARIES ${CUDA_LIBRARIES}
  RUN_OUTPUT_VARIABLE CUDNN_VERSION)
add_definitions(-DWITH_CUDNN)
message("cuDNN-libs: " ${CUDNN_LIBRARIES})
message("cuDNN-includes: " ${CUDNN_INCLUDE_DIRS})
message("cuDNN version: " ${CUDNN_VERSION})
list(APPEND NBLA_CUDA_INCLUDE_DIRS ${CUDNN_INCLUDE_DIRS})
list(APPEND NBLA_CUDA_LINKER_LIBS ${CUDNN_LIBRARIES})

include_directories(${NBLA_CUDA_INCLUDE_DIRS})
message("CUDA libs: ${NBLA_CUDA_LINKER_LIBS}")
message("CUDA includes: ${NBLA_CUDA_INCLUDE_DIRS}")
include(${CMAKE_SOURCE_DIR}/build-tools/cmake/select_compute_arch.cmake)
CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS ${CUDA_SELECT_NVCC_ARCH_ARG})
message("Arch: ${ARCH_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${ARCH_FLAGS}")

# cuDNN code will link into single library with CUDA code.
# To prevent user confusion, library filename will change that cudnn in included or not.
set(NBLA_CUDA_LIBRARY_NAME nnabla_cuda)

file(GLOB CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ./*.cpp array/*.cpp cudnn/*.cpp)

if(MSVC)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler /W0")
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; --default-stream per-thread")
endif()

file(GLOB CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */*.cu cudnn/*/*.cu)

# Distributed Training
if(NOT DEFINED ENV{NCCL_HOME})
	list(REMOVE_ITEM CU_SOURCES communicator/data_parallel_communicator.cu)
	list(REMOVE_ITEM CU_SOURCES communicator/multi_process_data_parallel_communicator.cu)
endif()

cuda_compile(CU_OBJECTS
  ${CU_SOURCES}
  )

add_library(${NBLA_CUDA_LIBRARY_NAME} SHARED ${CPP_SOURCES} ${CU_OBJECTS})

target_link_libraries(${NBLA_CUDA_LIBRARY_NAME} ${NBLA_LINKER_LIBS} ${NBLA_CUDA_LINKER_LIBS} ${NBLA_CPU_LIBRARY_FILE})
set_property(TARGET ${NBLA_CUDA_LIBRARY_NAME} PROPERTY CXX_STANDARD 11)

set(NBLA_CUDA_INCLUDE_DIRS ${NBLA_CUDA_INCLUDE_DIRS} PARENT_SCOPE)
set(NBLA_CUDA_LINKER_LIBS ${NBLA_CUDA_LINKER_LIBS} PARENT_SCOPE)
set(NBLA_CUDA_LIBRARY_NAME ${NBLA_CUDA_LIBRARY_NAME} PARENT_SCOPE)
set(NBLA_CUDNN_VERSION ${CUDNN_VERSION} PARENT_SCOPE)

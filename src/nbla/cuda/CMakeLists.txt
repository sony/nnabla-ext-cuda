# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(NBLA_CUDA_LIBRARY_NAME nnabla_cuda${EXT_CUDA_LIB_NAME_SUFFIX})
set(CUDA_SELECT_NVCC_ARCH_ARG "Auto" CACHE STRING "Select NVCC Arch. Options: Common | All | LIST(ARCH_AND_PTX ...). See select_compute_arch.cmake for details.")

find_package(CUDA REQUIRED)
message("-- CUDA--")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})
message("CUBLAS: " ${CUDA_CUBLAS_LIBRARIES})
message("CURAND: " ${CUDA_curand_LIBRARY})
message("CUFFT: " ${CUDA_CUFFT_LIBRARIES})
set(NBLA_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
list(APPEND NBLA_CUDA_LINKER_LIBS
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDA_CUFFT_LIBRARIES}
  )
  
# Distributed Training
option(WITH_NCCL "Use nccl for distributed training" OFF)
if(WITH_NCCL)
	add_definitions(-DFEATURE_DIST_TRAIN)
	find_package(NCCL REQUIRED)
	find_package(MPI REQUIRED)
	list(APPEND NBLA_CUDA_INCLUDE_DIRS 
		${NCCL_INCLUDE_DIR}
	        ${MPI_INCLUDE_PATH}
	  )
	list(APPEND NBLA_CUDA_LINKER_LIBS
		${NCCL_LIBRARIES}
	        ${MPI_LIBRARIES}
	  )
endif()

# nvprof support
option(WITH_NVTX "Build with nvtx API for nvprof support" OFF)
if(WITH_NVTX)
    find_library(CUDA_NVTX_LIBRARY nvToolsExt
            HINTS
            "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
            "${CUDA_TOOLKIT_ROOT_DIR}/lib"
            "${CUDA_TOOLKIT_ROOT_DIR}"
            ENV CUDA_PATH
            ENV CUDA_LIB_PATH)
    if (NOT CUDA_NVTX_LIBRARY)
        message(WARNING "nvToolsExt-libs is not found. Build without nvToolsExt.")
    else()
        list(APPEND NBLA_CUDA_LINKER_LIBS ${CUDA_NVTX_LIBRARY})
        add_definitions(-DWITH_NVTX)
        message("NVToolsExt: " ${CUDA_NVTX_LIBRARY})
    endif()
endif()


####################################################################################################
# cuDNN

find_package(cuDNN REQUIRED)
# Get CuDNN version

foreach(cudnn_include_dir ${CUDNN_INCLUDE_DIRS})
  set(cudnn_version_file OFF)
  if (EXISTS ${cudnn_include_dir}/cudnn_version.h)
    set(cudnn_version_file "${cudnn_include_dir}/cudnn_version.h")
  elseif(EXISTS ${cudnn_include_dir}/cudnn.h)
    set(cudnn_version_file "${cudnn_include_dir}/cudnn.h")
  endif()

  if (cudnn_version_file)
    file(STRINGS ${cudnn_version_file} cudnn_defines)
    string(REGEX REPLACE [[^.*CUDNN_MAJOR +([0-9]+).*$]] [[\1]] cudnn_major ${cudnn_defines})
    string(REGEX REPLACE [[^.*CUDNN_MINOR +([0-9]+).*$]] [[\1]] cudnn_minor ${cudnn_defines})
    string(REGEX REPLACE [[^.*CUDNN_PATCHLEVEL +([0-9]+).*$]] [[\1]] cudnn_patchlevel ${cudnn_defines})
  endif()
endforeach()
set(CUDNN_VERSION "${cudnn_major}.${cudnn_minor}.${cudnn_patchlevel}")

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

file(GLOB CPP_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  ./*.cpp
  communicator/*.cpp
  memory/*.cpp
  array/*.cpp
  cudnn/*.cpp
  utils/*.cpp)

if(MSVC)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler /W0")
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
  # Do not use `--default-stream per-thread` since some kernel calls are executed strangely in blocking streams other than default stream
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; --default-stream per-thread")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};")
endif()

file(GLOB CU_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */*.cu cudnn/*/*.cu)

# Distributed Training
if(NOT WITH_NCCL)
	list(REMOVE_ITEM CU_SOURCES communicator/data_parallel_communicator.cu)
	list(REMOVE_ITEM CU_SOURCES communicator/multi_process_data_parallel_communicator.cu)
endif()

cuda_compile(CU_OBJECTS
  ${CU_SOURCES}
  )

add_library(${NBLA_CUDA_LIBRARY_NAME} SHARED ${CPP_SOURCES} ${CU_OBJECTS})

target_link_libraries(${NBLA_CUDA_LIBRARY_NAME} ${NBLA_LINKER_LIBS} ${NBLA_CUDA_LINKER_LIBS} ${CPPLIB_LIBRARY})
set_property(TARGET ${NBLA_CUDA_LIBRARY_NAME} PROPERTY CXX_STANDARD 11)

# Install library nnabla and include files.
install(TARGETS ${NBLA_CUDA_LIBRARY_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nbla DESTINATION include)
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/NOTICE.md
  DESTINATION doc)
install(FILES ${PROJECT_SOURCE_DIR}/third_party/LICENSES.md DESTINATION doc/third_party)
include(CPack)

set(NBLA_CUDA_INCLUDE_DIRS ${NBLA_CUDA_INCLUDE_DIRS} PARENT_SCOPE)
set(NBLA_CUDA_LINKER_LIBS ${NBLA_CUDA_LINKER_LIBS} PARENT_SCOPE)
set(NBLA_CUDA_LIBRARY_NAME ${NBLA_CUDA_LIBRARY_NAME} PARENT_SCOPE)

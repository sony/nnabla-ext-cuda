# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

get_filename_component(CPPLIB_DIR ${CPPLIB_LIBRARY} DIRECTORY)
link_directories(${CPPLIB_DIR})

if (WIN32)
  # Use simulated cxxabi.h since it is not appeared in MSVC. 
  include_directories(${PROJECT_SOURCE_DIR}/include/msvc)
endif()

if (NNABLA_UTILS_WITH_HDF5)
  add_definitions(-DNBLA_UTILS_WITH_HDF5)
endif()

if (NNABLA_UTILS_WITH_HDF5)
if (NNABLA_UTILS_STATIC_LINK_DEPS)
    set(HDF5_USE_STATIC_LIBRARIES 1)
endif()
find_package(HDF5 REQUIRED COMPONENTS C HL)
endif()
if (NNABLA_UTILS_STATIC_LINK_DEPS)
if (NNABLA_UTILS_WITH_HDF5)
    set(HDF5_LIBRARIES ${HDF5_LIBRARIES} ${CMAKE_DL_LIBS})
endif()
endif()

if (NNABLA_UTILS_WITH_HDF5)
include_directories(${HDF5_INCLUDE_DIRS})
endif()

include_directories(${NNABLA_DIR}/src/nbla_cli)
add_definitions(-DWITH_CUDA)
add_executable(nbla_cuda nbla_cuda.cpp)

target_link_libraries(nbla_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${NBLA_CLI_LIBRARY_NAME} ${HDF5_LIBRARIES})
set_property(TARGET nbla_cuda PROPERTY CXX_STANDARD 14)

install(TARGETS nbla_cuda RUNTIME DESTINATION bin)

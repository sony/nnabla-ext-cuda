# Copyright 2022 Sony Group Corporation.
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

cmake_minimum_required(VERSION 3.16)
project(semantic)

find_package(OpenCV REQUIRED)

include_directories("../../../../nnabla/include")

find_library(NBLA_LIBRARY_NAME nnabla)
# find_library(NBLA_LIBRARY_NAME nnabla HINTS ~/.local/lib/python3.9/site-packages/nnabla/bin)
find_library(NBLA_UTILS_LIBRARY_NAME nnabla_utils)
# find_library(NBLA_UTILS_LIBRARY_NAME nnabla_utils HINTS ~/.local/lib/python3.9/site-packages/nnabla/bin)
find_library(CURL curl)

option(TIMING "Timeing" ON)
if(TIMING)
  add_definitions(-DTIMING=1)
endif()
option(WITH_CUDA "Infer with cuda" OFF)
if(WITH_CUDA)
  add_definitions(-DWITH_CUDA=1)
  include_directories("../../../include")
  find_library(NBLA_CUDA_LIBRARY_NAME nnabla_cuda)
  # find_library(NBLA_CUDA_LIBRARY_NAME nnabla_cuda-110_8 HINTS ~/.local/lib/python3.9/site-packages/nnabla_ext/cuda)
endif()

add_executable(semantic semantic.cpp)
target_link_libraries(semantic ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${OpenCV_LIBS} ${CURL})
set_property(TARGET semantic PROPERTY CXX_STANDARD 14)




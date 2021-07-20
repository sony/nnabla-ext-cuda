# Copyright 2019,2020,2021 Sony Corporation.
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

add_executable(train_lenet_classifier_cuda train_lenet_classifier_cuda.cpp)
add_executable(train_resnet_classifier_cuda train_resnet_classifier_cuda.cpp)
add_executable(train_vae_cuda train_vae_cuda.cpp)
add_executable(train_dcgan_cuda train_dcgan_cuda.cpp)
add_executable(train_siamese_cuda train_siamese_cuda.cpp)
add_executable(train_vat_cuda train_vat_cuda.cpp)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${NNABLA_DIR}/examples/cpp/mnist_collection)
target_link_libraries(train_lenet_classifier_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_lenet_classifier_cuda PROPERTY CXX_STANDARD 11)
target_link_libraries(train_resnet_classifier_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_resnet_classifier_cuda PROPERTY CXX_STANDARD 11)
target_link_libraries(train_vae_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_vae_cuda PROPERTY CXX_STANDARD 11)
target_link_libraries(train_dcgan_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_dcgan_cuda PROPERTY CXX_STANDARD 11)
target_link_libraries(train_siamese_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_siamese_cuda PROPERTY CXX_STANDARD 11)
target_link_libraries(train_vat_cuda ${NBLA_LIBRARY_NAME} ${NBLA_UTILS_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME} ${ZLIB_LIBRARIES})
set_property(TARGET train_vat_cuda PROPERTY CXX_STANDARD 11)

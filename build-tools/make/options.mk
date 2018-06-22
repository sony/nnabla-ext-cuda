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

########################################################################################################################
# Suppress most of make message.
.SILENT:
NNABLA_EXT_CUDA_OPTIONS_INCLUDED = True

export CUDA_VERSION_MAJOR ?= 9
DOCKER_RUN_OPTS += -e CUDA_VERSION_MAJOR=$(CUDA_VERSION_MAJOR)

export CUDA_VERSION_MINOR ?= 2
DOCKER_RUN_OPTS += -e CUDA_VERSION_MINOR=$(CUDA_VERSION_MINOR)

export CUDNN_VERSION ?= 7
DOCKER_RUN_OPTS += -e CUDNN_VERSION=$(CUDNN_VERSION)

export WHL_NO_PREFIX ?= False
DOCKER_RUN_OPTS += -e WHL_NO_PREFIX=$(WHL_NO_PREFIX)

ifndef NNABLA_OPTIONS_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/options.mk
endif

MAKE_MANYLINUX_WHEEL ?= OFF
DOCKER_RUN_OPTS += -e MAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL)

export BUILD_EXT_CUDA_DIRECTORY_CPPLIB ?= $(NNABLA_EXT_CUDA_DIRECTORY)/build$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_EXT_CUDA_DIRECTORY_CPPLIB=$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)

export BUILD_EXT_CUDA_DIRECTORY_WHEEL ?= $(NNABLA_EXT_CUDA_DIRECTORY)/build_wheel$(BUILD_EXT_CUDA_DIRECTORY_WHEEL_SUFFIX)
DOCKER_RUN_OPTS += -e BUILD_EXT_CUDA_DIRECTORY_WHEEL=$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)

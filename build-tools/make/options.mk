# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2022 Sony Corporation.
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
ifndef NNABLA_EXT_CUDA_OPTIONS_INCLUDED

  NNABLA_EXT_CUDA_OPTIONS_INCLUDED = True

  export CUDA_VERSION_MAJOR ?= 10
  DOCKER_RUN_OPTS += -e CUDA_VERSION_MAJOR=$(CUDA_VERSION_MAJOR)

  export CUDA_VERSION_MINOR ?= 2
  DOCKER_RUN_OPTS += -e CUDA_VERSION_MINOR=$(CUDA_VERSION_MINOR)

  export CUDNN_VERSION ?= 8
  DOCKER_RUN_OPTS += -e CUDNN_VERSION=$(CUDNN_VERSION)

  export WHL_NO_CUDA_SUFFIX ?= False
  DOCKER_RUN_OPTS += -e WHL_NO_CUDA_SUFFIX=$(WHL_NO_CUDA_SUFFIX)

  ifndef NNABLA_OPTIONS_INCLUDED
    include $(NNABLA_DIRECTORY)/build-tools/make/options.mk
  endif

  ifndef OMPI_VERSION
    NOT_EMPYT=ON
    OMPI_VERSION=3.1.6
  endif

  ifdef MULTIGPU
    ifneq (, $(OMPI_VERSION))
        export OMPI_SUFFIX ?= _mpi$(shell echo $(OMPI_VERSION) | sed 's/^\([0-9]*\)\.\([0-9]*\)\.\([0-9]\)*.*$$/\1_\2_\3/g')
    else
      ifneq (, $(shell which ompi_info))
        export OMPI_SUFFIX ?= $(shell ompi_info |grep Open.MPI: |sed 's/^ *Open.MPI: \([0-9]*\.[0-9]*\)\.[0-9]*/_mpi\1/g')
      else
        export OMPI_SUFFIX ?= _mpi
      endif
    endif
    export MULTIGPU_SUFFIX ?= _nccl2$(OMPI_SUFFIX)
    export WITH_NCCL ?= ON
  else
    export WITH_NCCL ?= OFF
  endif

  DOCKER_RUN_OPTS += -e OMPI_VERSION=$(OMPI_VERSION)
  DOCKER_RUN_OPTS += -e MULTIGPU_SUFFIX=$(MULTIGPU_SUFFIX)
  DOCKER_RUN_OPTS += -e WITH_NCCL=$(WITH_NCCL)

  DOCKER_RUN_OPTS += -e NOT_EMPYT=$(NOT_EMPYT)
  UBUNTU_VERSION ?= 16
  DOCKER_RUN_OPTS += -e UBUNTU_VERSION=$(UBUNTU_VERSION)

  export BUILD_EXT_CUDA_DIRECTORY_CPPLIB ?= $(NNABLA_EXT_CUDA_DIRECTORY)/build$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_SUFFIX)
  DOCKER_RUN_OPTS += -e BUILD_EXT_CUDA_DIRECTORY_CPPLIB=$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)

  export BUILD_EXT_CUDA_DIRECTORY_WHEEL ?= $(NNABLA_EXT_CUDA_DIRECTORY)/build_wheel$(BUILD_EXT_CUDA_DIRECTORY_WHEEL_SUFFIX)
  DOCKER_RUN_OPTS += -e BUILD_EXT_CUDA_DIRECTORY_WHEEL=$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)

  export EXT_CUDA_LIB_NAME_SUFFIX ?= $(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)_$(CUDNN_VERSION)$(SUFFIX)
  DOCKER_RUN_OPTS += -e EXT_CUDA_LIB_NAME_SUFFIX=$(EXT_CUDA_LIB_NAME_SUFFIX)

  CU_MINOR=$(shell echo $(word 1, $(subst ., ,${CUDA_VERSION_MINOR})));
  TARGZ_CUDA_VERSION=`echo $(CUDA_VERSION_MAJOR)$(CU_MINOR)`

  ifeq ($(PYTHON_VERSION_MINOR), 7)
    TARGZ_PYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR)m
  else
    TARGZ_PYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR)
  endif
endif


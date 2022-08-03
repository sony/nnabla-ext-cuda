# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

########################################################################################################################
# Settings

NNABLA_DIRECTORY ?= $(shell cd ../nnabla && pwd)
DOCKER_RUN_OPTS += -e NNABLA_DIRECTORY=$(NNABLA_DIRECTORY)

NNABLA_EXT_CUDA_DIRECTORY ?= $(shell pwd)
DOCKER_RUN_OPTS += -e NNABLA_EXT_CUDA_DIRECTORY=$(NNABLA_EXT_CUDA_DIRECTORY)
DOCKER_RUN_OPTS += -e CMAKE_OPTS=$(CMAKE_OPTS)
DOCKER_RUN_OPTS += -e INCLUDE_CUDA_CUDNN_LIB_IN_WHL=$(INCLUDE_CUDA_CUDNN_LIB_IN_WHL)

include $(NNABLA_EXT_CUDA_DIRECTORY)/build-tools/make/options.mk
ifndef NNABLA_BUILD_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/build.mk
endif

ifndef NNABLA_BUILD_WITH_DOCKER_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/build-with-docker.mk
endif

NVIDIA_DOCKER_WRAPPER=$(NNABLA_EXT_CUDA_DIRECTORY)/build-tools/scripts/nvidia-docker.sh

CU_MINOR = $(shell echo $(word 1, $(subst ., ,${CUDA_VERSION_MINOR})))
CUDA_SUFFIX = $(CUDA_VERSION_MAJOR)$(CU_MINOR)-cudnn$(CUDNN_VERSION)

DOCKERFILE_OMPI_SUFFIX_NNABLA_EXT_CUDA=$(shell [ -n "$(OMPI_SUFFIX)" ] && echo -mpi)
DOCKERFILE_PATH_NNABLA_EXT_CUDA=$(NNABLA_EXT_CUDA_DIRECTORY)/docker/development/Dockerfile.build$(DOCKERFILE_OMPI_SUFFIX_NNABLA_EXT_CUDA)$(ARCH_SUFFIX)
DOCKER_IMAGE_ID_BUILD_NNABLA_EXT_CUDA = $(shell md5sum $(DOCKERFILE_PATH_NNABLA_EXT_CUDA) |cut -d \  -f 1)
DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA ?= $(DOCKER_IMAGE_NAME_BASE)-build-cuda$(CUDA_SUFFIX)$(OMPI_SUFFIX)$(ARCH_SUFFIX):$(DOCKER_IMAGE_ID_BUILD_NNABLA_EXT_CUDA)
DOCKER_IMAGE_NNABLA_EXT_CUDA ?= $(DOCKER_IMAGE_NAME_BASE)-nnabla-ext-cuda$(CUDA_SUFFIX)$(OMPI_SUFFIX)$(ARCH_SUFFIX)

########################################################################################################################
# Docker image

.PHONY: docker_image_build_cuda
docker_image_build_cuda:
	docker pull nvidia/cuda$(ARCH_SUFFIX):$(CUDA_VERSION_MAJOR).$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION)-devel-centos7
	if ! docker image inspect $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) >/dev/null 2>/dev/null; then \
		echo "Building: $(DOCKERFILE_PATH_NNABLA_EXT_CUDA)"; \
		(cd $(NNABLA_EXT_CUDA_DIRECTORY) && docker build $(DOCKER_BUILD_ARGS)\
			--build-arg CUDA_VERSION_MAJOR=$(CUDA_VERSION_MAJOR) \
			--build-arg CUDA_VERSION_MINOR=$(CUDA_VERSION_MINOR) \
			--build-arg CUDNN_VERSION=$(CUDNN_VERSION) \
			--build-arg ARCH_SUFFIX=$(ARCH_SUFFIX) \
			--build-arg MPIVER=$(OMPI_VERSION) \
			--build-arg PYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
			--build-arg PYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
			-t $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) \
			-f docker/development/Dockerfile.build-mpi$(ARCH_SUFFIX) \
			.); \
	fi

##############################################################################
# Auto Format

.PHONY: bwd-nnabla-ext-cuda-auto-format
bwd-nnabla-ext-cuda-auto-format: docker_image_auto_format
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_AUTO_FORMAT) make -f build-tools/make/build.mk nnabla-ext-cuda-auto-format


##############################################################################
# Check copyright

.PHONY: bwd-nnabla-ext-cuda-check-copyright
bwd-nnabla-ext-cuda-check-copyright: docker_image_auto_format
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) -v $$(pwd)/..:$$(pwd)/.. $(DOCKER_IMAGE_AUTO_FORMAT) make -f build-tools/make/build.mk nnabla-ext-cuda-check-copyright


########################################################################################################################
# Build and test

NNABLA_DIRECTORY_ABSOLUTE = $(shell cd $(NNABLA_DIRECTORY) && pwd)
DOCKER_RUN_OPTS += -v $(NNABLA_DIRECTORY_ABSOLUTE):$(NNABLA_DIRECTORY_ABSOLUTE)

NNABLA_EXT_CUDA_DIRECTORY_ABSOLUTE = $(shell cd $(NNABLA_EXT_CUDA_DIRECTORY) && pwd)
DOCKER_RUN_OPTS += -v $(NNABLA_EXT_CUDA_DIRECTORY_ABSOLUTE):$(NNABLA_EXT_CUDA_DIRECTORY_ABSOLUTE)

.PHONY: bwd-nnabla-ext-cuda-cpplib
bwd-nnabla-ext-cuda-cpplib: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk nnabla-ext-cuda-cpplib

.PHONY: bwd-nnabla-ext-cuda-wheel
bwd-nnabla-ext-cuda-wheel: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk MAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) nnabla-ext-cuda-wheel-local

.PHONY: bwd-nnabla-ext-cuda-test
bwd-nnabla-ext-cuda-test: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& ${NVIDIA_DOCKER_WRAPPER} run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk nnabla-ext-cuda-test-local

.PHONY: bwd-nnabla-ext-cuda-multi-gpu-test
bwd-nnabla-ext-cuda-multi-gpu-test: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& ${NVIDIA_DOCKER_WRAPPER} run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk nnabla-ext-cuda-multi-gpu-test-local

.PHONY: bwd-nnabla-ext-cuda-shell
bwd-nnabla-ext-cuda-shell: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& ${NVIDIA_DOCKER_WRAPPER} run $(DOCKER_RUN_OPTS) -it --rm ${DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA} make nnabla-ext-cuda-shell

########################################################################################################################
# Docker image with current nnabla
OMPI_BUILD_FLAGS_V1=""
OMPI_BUILD_FLAGS_V2="--enable-orterun-prefix-by-default --with-sge --enable-mpi-thread-multiple "
OMPI_BUILD_FLAGS_V3="--enable-orterun-prefix-by-default --with-sge "
OMPI_BUILD_FLAGS_V4="--enable-orterun-prefix-by-default --with-sge "

.PHONY: docker_image_nnabla_ext_cuda
docker_image_nnabla_ext_cuda:
	BASE=nvidia/cuda$(ARCH_SUFFIX):$(CUDA_VERSION_MAJOR).$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION)-runtime-ubuntu18.04 \
	&& docker pull $${BASE} \
	&& cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& if [ "$(OMPI_SUFFIX)" = "" ]; then \
		cp docker/runtime/Dockerfile.runtime$(ARCH_SUFFIX) Dockerfile; \
	   else \
		cp docker/runtime/Dockerfile.runtime-mpi$(ARCH_SUFFIX) Dockerfile; \
	   fi \
	&& cp $(BUILD_DIRECTORY_WHEEL)/dist/*.whl $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/dist/ \
	&& docker build $(DOCKER_BUILD_ARGS) \
		--build-arg BASE=$${BASE} \
		--build-arg MPIVER=$(OMPI_VERSION) \
		--build-arg OMPI_BUILD_FLAGS=${OMPI_BUILD_FLAGS_V$(firstword $(subst ., ,$(OMPI_VERSION)))} \
		--build-arg CUDA_VERSION_MAJOR=$(CUDA_VERSION_MAJOR) \
		--build-arg CUDA_VERSION_MINOR=$(CUDA_VERSION_MINOR) \
		--build-arg WHL_PATH=$$(echo build_wheel$(BUILD_EXT_CUDA_DIRECTORY_WHEEL_SUFFIX)/dist) \
		-t $(DOCKER_IMAGE_NNABLA_EXT_CUDA) . \
	&& ls $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/dist/* | grep -v nnabla_ext_cuda | xargs rm -f \
	&& rm -f Dockerfile

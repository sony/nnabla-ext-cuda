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

########################################################################################################################
# Settings

NNABLA_DIRECTORY ?= $(shell cd ../nnabla && pwd)
DOCKER_RUN_OPTS += -e NNABLA_DIRECTORY=$(NNABLA_DIRECTORY)

NNABLA_EXT_CUDA_DIRECTORY ?= $(shell pwd)
DOCKER_RUN_OPTS += -e NNABLA_EXT_CUDA_DIRECTORY=$(NNABLA_EXT_CUDA_DIRECTORY)
DOCKER_RUN_OPTS += -e CMAKE_OPTS=$(CMAKE_OPTS)

include $(NNABLA_EXT_CUDA_DIRECTORY)/build-tools/make/options.mk
ifndef NNABLA_BUILD_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/build.mk
endif

ifndef NNABLA_BUILD_WITH_DOCKER_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/build-with-docker.mk
endif

MAKE_MANYLINUX_WHEEL=ON

DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA ?= $(DOCKER_IMAGE_NAME_BASE)-build-cuda$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION)
DOCKER_IMAGE_NNABLA_EXT_CUDA ?= $(DOCKER_IMAGE_NAME_BASE)-nnabla-ext--cuda$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION)

########################################################################################################################
# Docker image

DOCKERFILE_NAME_SUFFIX := py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)
DOCKERFILE_NAME_SUFFIX := $(DOCKERFILE_NAME_SUFFIX)-cuda$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)
DOCKERFILE_NAME_SUFFIX := $(DOCKERFILE_NAME_SUFFIX)-cudnn$(CUDNN_VERSION)
.PHONY: docker_image_build_cuda
docker_image_build_cuda:
	docker pull nvidia/cuda:$(CUDA_VERSION_MAJOR).$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION)-devel-centos6
	@cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& python docker/development/generate_dockerfile.py $(DOCKERFILE_NAME_SUFFIX) \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) \
		-f docker/development/Dockerfile.build.$(DOCKERFILE_NAME_SUFFIX) .

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
	&& docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk nnabla-ext-cuda-wheel-local

.PHONY: bwd-nnabla-ext-cuda-test
bwd-nnabla-ext-cuda-test: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& nvidia-docker run $(DOCKER_RUN_OPTS) $(DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA) make -f build-tools/make/build.mk nnabla-ext-cuda-test-local

.PHONY: bwd-nnabla-ext-cuda-shell
bwd-nnabla-ext-cuda-shell: docker_image_build_cuda
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& nvidia-docker run $(DOCKER_RUN_OPTS) -it --rm ${DOCKER_IMAGE_BUILD_NNABLA_EXT_CUDA} make nnabla-ext-cuda-shell

########################################################################################################################
# Docker image with current nnabla
.PHONY: docker_image_nnabla_ext_cuda
docker_image_nnabla_ext_cuda: bwd-nnabla-cpplib bwd-nnabla-wheel bwd-nnabla-ext-cuda-cpplib bwd-nnabla-ext-cuda-wheel
	docker pull ubuntu:16.04
	cd $(NNABLA_EXT_CUDA_DIRECTORY) \
	&& cp docker/development/Dockerfile.build.py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)-cuda$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)-cudnn$(CUDNN_VERSION) Dockerfile \
	&& cp $(shell echo $(NNABLA_DIRECTORY)/build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) . \
	&& echo ADD $(shell basename $(NNABLA_DIRECTORY)/build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) /tmp/ >>Dockerfile \
	&& echo RUN pip install /tmp/$(shell basename $(NNABLA_DIRECTORY)/build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) >>Dockerfile \
	&& echo ADD $(shell echo build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)_$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)_$(CUDNN_VERSION)/dist/*.whl) /tmp/ >>Dockerfile \
	&& echo RUN pip install /tmp/$(shell basename build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)_$(CUDA_VERSION_MAJOR)$(CUDA_VERSION_MINOR)_$(CUDNN_VERSION)/dist/*.whl) >>Dockerfile \
	&& docker build $(DOCKER_BUILD_ARGS) -t $(DOCKER_IMAGE_NNABLA_EXT_CUDA) . \
	&& rm -f $(shell basename $(NNABLA_DIRECTORY)/build_wheel_py$(PYTHON_VERSION_MAJOR)$(PYTHON_VERSION_MINOR)/dist/*.whl) \
	&& rm -f Dockerfile

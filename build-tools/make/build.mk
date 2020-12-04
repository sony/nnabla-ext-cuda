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
NNABLA_EXT_CUDA_DIRECTORY ?= $(shell pwd)

include $(NNABLA_EXT_CUDA_DIRECTORY)/build-tools/make/options.mk

ifndef NNABLA_BUILD_INCLUDED
  include $(NNABLA_DIRECTORY)/build-tools/make/build.mk
endif

CUDA_ROOT ?= /usr/local/cuda

########################################################################################################################
# cleaning
.PHONY: nnabla-ext-cuda-clean
nnabla-ext-cuda-clean:
	@git clean -fdX

.PHONY: nnabla-ext-cuda-clean-all
nnabla-ext-cuda-clean-all:
	@git clean -fdx
##############################################################################
# Auto Format
.PHONY: nnabla-auto-format
nnabla-ext-cuda-auto-format:
	cd $(NNABLA_EXT_CUDA_DIRECTORY) && \
	python3 $(NNABLA_DIRECTORY)/build-tools/auto_format . --exclude \
		'\./src/nbla/cuda(/cudnn)?/(function|solver)/\w+\.cu' \
		'\./src/nbla/cuda(/cudnn)?/init.cpp' \
		'\./python/src/nnabla_ext/(cuda|cudnn)/.*.(cpp|hpp|h|c)'

########################################################################################################################
# cpplib
.PHONY: nnabla-ext-cuda-cpplib
nnabla-ext-cuda-cpplib:
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)
	cd $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB) \
	&& cmake \
		-D CUDA_SELECT_NVCC_ARCH_ARG:STRING="Common" \
		-DBUILD_CPP_LIB=ON \
		-DBUILD_CPP_UTILS=ON \
		-DBUILD_PYTHON_PACKAGE=OFF \
		-DBUILD_TEST=ON \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla$(LIB_NAME_SUFFIX).so \
	        -DEXT_CUDA_LIB_NAME_SUFFIX=$(EXT_CUDA_LIB_NAME_SUFFIX) \
	        -DLIB_NAME_SUFFIX=$(LIB_NAME_SUFFIX) \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DMULTIGPU_SUFFIX=$(MULTIGPU_SUFFIX) \
		-DWITH_NCCL=$(WITH_NCCL) \
		-DCMAKE_LIBRARY_PATH=$(CUDA_ROOT)/lib64/stubs \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY)
	$(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB) -j$(PARALLEL_BUILD_NUM)
	@cd $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB) && cpack -G ZIP

########################################################################################################################
# wheel
.PHONY: nnabla-ext-cuda-wheel
nnabla-ext-cuda-wheel:
	$(call with-venv, \
		$(NNABLA_EXT_CUDA_DIRECTORY), \
		$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/env, \
		-f build-tools/make/build.mk, \
		nnabla-ext-cuda-wheel-local)

.PHONY: nnabla-ext-cuda-wheel-local
nnabla-ext-cuda-wheel-local: nnabla-install \
		$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla$(LIB_NAME_SUFFIX).so \
		$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)/lib/libnnabla_cuda$(EXT_CUDA_LIB_NAME_SUFFIX).so
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& cmake \
		-DBUILD_CPP_LIB=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		-DCPPLIB_BUILD_DIR=$(BUILD_DIRECTORY_CPPLIB) \
		-DCPPLIB_CUDA_LIBRARY=$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)/lib/libnnabla_cuda$(EXT_CUDA_LIB_NAME_SUFFIX).so \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla$(LIB_NAME_SUFFIX).so \
	        -DEXT_CUDA_LIB_NAME_SUFFIX=$(EXT_CUDA_LIB_NAME_SUFFIX) \
	        -DLIB_NAME_SUFFIX=$(LIB_NAME_SUFFIX) \
		-DMAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DMULTIGPU_SUFFIX=$(MULTIGPU_SUFFIX) \
		-DWITH_NCCL=$(WITH_NCCL) \
                -DWHEEL_SUFFIX=$(WHEEL_SUFFIX) \
		-DCMAKE_LIBRARY_PATH=$(CUDA_ROOT)/lib64/stubs \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY) \
	&& $(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) wheel

.PHONY: nnabla-ext-cuda-install
nnabla-ext-cuda-install:
	pip install --force-reinstall --no-deps $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/dist/*-$(INSTALL_WHEEL_ARCH)*.whl

########################################################################################################################
# Shell (for rapid development)
.PHONY: nnabla-ext-cuda-shell
nnabla-ext-cuda-shell:
	PS1="nnabla-ext-cuda-build: " bash --norc -i

########################################################################################################################
# test
.PHONY: nnabla-ext-cuda-test
nnabla-ext-cuda-test:
	$(call with-venv, \
		$(NNABLA_EXT_CUDA_DIRECTORY), \
		$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/env, \
		-f build-tools/make/build.mk, \
		nnabla-ext-cuda-test-local)

.PHONY: nnabla-ext-cuda-test-local
nnabla-ext-cuda-test-local: nnabla-install nnabla-ext-cuda-install
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& PYTHONPATH=$(NNABLA_EXT_CUDA_DIRECTORY)/python/test \
	&& $(NNABLA_DIRECTORY)/build-tools/make/pytest.sh $(NNABLA_DIRECTORY)/python/test \
	&& $(NNABLA_DIRECTORY)/build-tools/make/pytest.sh $(NNABLA_EXT_CUDA_DIRECTORY)/python/test

.PHONY: nnabla-ext-cuda-multi-gpu-test-local
nnabla-ext-cuda-multi-gpu-test-local: nnabla-install nnabla-ext-cuda-install
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& PYTHONPATH=$(NNABLA_EXT_CUDA_DIRECTORY)/python/test:$(NNABLA_DIRECTORY)/python/test \
		mpiexec -q -n 2 $(NNABLA_DIRECTORY)/build-tools/make/pytest.sh \
			--test-communicator \
			--communicator-gpus=0,1 \
			$(NNABLA_DIRECTORY)/python/test/communicator
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& $(NNABLA_DIRECTORY)/build-tools/make/pytest.sh $(NNABLA_DIRECTORY)/python/test \
	&& $(NNABLA_DIRECTORY)/build-tools/make/pytest.sh $(NNABLA_EXT_CUDA_DIRECTORY)/python/test

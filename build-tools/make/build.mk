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

########################################################################################################################
# cleaning
.PHONY: nnabla-ext-cuda-clean
nnabla-ext-cuda-clean:
	@git clean -fdX

.PHONY: nnabla-ext-cuda-clean-all
nnabla-ext-cuda-clean-all:
	@git clean -fdx

########################################################################################################################
# cpplib
.PHONY: nnabla-ext-cuda-cpplib
nnabla-ext-cuda-cpplib:
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)
	cd $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-D CUDA_SELECT_NVCC_ARCH_ARG:STRING="Common" \
		-DBUILD_CPP_LIB=ON \
		-DBUILD_TEST=ON \
		-DBUILD_PYTHON_PACKAGE=OFF \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY)
	$(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB) -j$(PARALLEL_BUILD_NUM)

.PHONY: nnabla-ext-cuda-cpplib-multi-gpu
nnabla-ext-cuda-cpplib-multi-gpu:
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_MULTI_GPU)
	cd $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_MULTI_GPU) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-D CUDA_SELECT_NVCC_ARCH_ARG:STRING="Common" \
		-DWITH_NCCL=ON \
		-DBUILD_CPP_LIB=ON \
		-DBUILD_TEST=ON \
		-DBUILD_PYTHON_PACKAGE=OFF \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY)
	$(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_MULTI_GPU) -j$(PARALLEL_BUILD_NUM)

########################################################################################################################
# wheel
.PHONY: nnabla-ext-cuda-wheel
nnabla-ext-cuda-wheel:
	$(call with-virtualenv, \
		$(NNABLA_EXT_CUDA_DIRECTORY), \
		$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/env, \
		-f build-tools/make/build.mk, \
		nnabla-ext-cuda-wheel-local)

.PHONY: nnabla-ext-cuda-wheel-local
nnabla-ext-cuda-wheel-local: nnabla-install \
		$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)/lib/libnnabla_cuda.so
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DBUILD_CPP_LIB=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		-DMAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DCPPLIB_BUILD_DIR=$(BUILD_DIRECTORY_CPPLIB) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		-DCPPLIB_CUDA_LIBRARY=$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB)/lib/libnnabla_cuda.so \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY) \
	&& $(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) wheel

.PHONY: nnabla-ext-cuda-wheel-multi-gpu
nnabla-ext-cuda-wheel-multi-gpu: \
			nnabla-cpplib \
			nnabla-wheel \
			nnabla-install \
			nnabla-ext-cuda-cpplib-multi-gpu
	mkdir -p $(BUILD_EXT_CUDA_DIRECTORY_WHEEL_MULTI_GPU)
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL_MULTI_GPU) \
	&& cmake \
		-DPYTHON_VERSION_MAJOR=$(PYTHON_VERSION_MAJOR) \
		-DPYTHON_VERSION_MINOR=$(PYTHON_VERSION_MINOR) \
		-DMULTIGPU_SUFFIX=_nccl2 \
		-DWHEEL_SUFFIX=$(WHEEL_SUFFIX) \
		-DMAKE_MANYLINUX_WHEEL=$(MAKE_MANYLINUX_WHEEL) \
		-DWITH_NCCL=ON \
		-DBUILD_CPP_LIB=OFF \
		-DBUILD_PYTHON_PACKAGE=ON \
		-DNNABLA_DIR=$(NNABLA_DIRECTORY) \
		-DCPPLIB_BUILD_DIR=$(BUILD_DIRECTORY_CPPLIB) \
		-DCPPLIB_LIBRARY=$(BUILD_DIRECTORY_CPPLIB)/lib/libnnabla.so \
		-DCPPLIB_CUDA_LIBRARY=$(BUILD_EXT_CUDA_DIRECTORY_CPPLIB_MULTI_GPU)/lib/libnnabla_cuda.so \
		$(CMAKE_OPTS) \
		$(NNABLA_EXT_CUDA_DIRECTORY)
	$(MAKE) -C $(BUILD_EXT_CUDA_DIRECTORY_WHEEL_MULTI_GPU) wheel

.PHONY: nnabla-ext-cuda-install
nnabla-ext-cuda-install:
	pip install --force-reinstall --no-deps $(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/dist/*-$(INSTALL_WHEEL_ARCH)*.whl

.PHONY: nnabla-ext-cuda-multi-gpu-install
nnabla-ext-cuda-multi-gpu-install:
	pip install --force-reinstall --no-deps $(BUILD_DIRECTORY_WHEEL)/dist/*.whl
	pip install --force-reinstall --no-deps $(BUILD_EXT_CUDA_DIRECTORY_WHEEL_MULTI_GPU)/dist/*.whl

########################################################################################################################
# Shell (for rapid development)
.PHONY: nnabla-ext-cuda-shell
nnabla-ext-cuda-shell:
	PS1="nnabla-ext-cuda-build: " bash --norc -i

########################################################################################################################
# test
.PHONY: nnabla-ext-cuda-test
nnabla-ext-cuda-test:
	$(call with-virtualenv, \
		$(NNABLA_EXT_CUDA_DIRECTORY), \
		$(BUILD_EXT_CUDA_DIRECTORY_WHEEL)/env, \
		-f build-tools/make/build.mk, \
		nnabla-ext-cuda-test-local)

.PHONY: nnabla-ext-cuda-test-local
nnabla-ext-cuda-test-local: nnabla-install nnabla-ext-cuda-install
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL) \
	&& PYTHONPATH=$(NNABLA_EXT_CUDA_DIRECTORY)/python/test \
		python -m pytest $(NNABLA_DIRECTORY)/python/test

.PHONY: nnabla-ext-cuda-multi-gpu-test-local
nnabla-ext-cuda-multi-gpu-test-local: nnabla-ext-cuda-multi-gpu-install
	cd $(BUILD_EXT_CUDA_DIRECTORY_WHEEL_MULTI_GPU) \
	&& PYTHONPATH=$(NNABLA_EXT_CUDA_DIRECTORY)/python/test:$(NNABLA_DIRECTORY)/python/test \
			mpiexec -q -n 2 python -m pytest --test-communicator --communicator-gpus=0,1 $(NNABLA_DIRECTORY)/python/test/communicator/ \
	&& python -m pytest $(NNABLA_DIRECTORY)/python/test/

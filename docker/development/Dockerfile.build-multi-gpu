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

# for nnabla>=1.0.17

ARG BASE
FROM ${BASE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    ccache \
    clang-format-3.9 \
    curl \
    g++ \
    git \
    libarchive-dev \
    libhdf5-dev \
    libopenmpi-dev \
    lsb-release \
    make \
    openmpi-bin \
    patchelf \
    ssh \
    unzip \
    wget \
    zip

################################################### cmake
ENV CMAKEVER=3.14.3
RUN mkdir /tmp/deps \
    && cd /tmp/deps \
    && apt-get install -y cmake \
    && curl -L https://github.com/Kitware/CMake/releases/download/v${CMAKEVER}/cmake-${CMAKEVER}.tar.gz -o cmake-${CMAKEVER}.tar.gz \
    && tar xf cmake-${CMAKEVER}.tar.gz \
    && cd cmake-${CMAKEVER} \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TESTING=FALSE .. \
    && make \
    && make install \
    && apt-get purge -y cmake \
    && rm -rf /var/cache/yum/* \
    && cd / \
    && rm -rf /tmp/*

################################################## protobuf
ENV PROTOVER=3.4.1
RUN mkdir /tmp/deps \
    && cd /tmp/deps \
    && curl -L https://github.com/google/protobuf/archive/v${PROTOVER}.tar.gz -o protobuf-v${PROTOVER}.tar.gz \
    && tar xvf protobuf-v${PROTOVER}.tar.gz \
    && cd protobuf-${PROTOVER} \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_TESTS=OFF \
        ../cmake \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

################################################## miniconda3
ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

ADD python/requirements.txt /tmp/deps/

RUN umask 0 \
    && cd /tmp/deps \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && . /opt/miniconda3/bin/activate \
    && conda create -n nnabla-build python=${PYVERNAME} \
    && conda activate nnabla-build \
    && pip install --only-binary -U -r /tmp/deps/requirements.txt \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/envs/nnabla-build/bin:$PATH

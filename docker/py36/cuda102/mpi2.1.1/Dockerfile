# Copyright 2020,2021 Sony Corporation.
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

#Flatbuffer compiler
FROM ubuntu:18.04 as flatc

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
       cmake \
       make \
       g++ \
       git \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && git clone https://github.com/google/flatbuffers.git \
    && cd flatbuffers \
    && cmake -G "Unix Makefiles" \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

RUN eval ${APT_OPTS} && apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       libglib2.0-0 \
       libgl1-mesa-glx \
       python3.6 \
       python3-pip \
       openmpi-bin \
       openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 0
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 0

RUN pip3 install ${PIP_INS_OPTS} --upgrade pip
RUN pip install ${PIP_INS_OPTS} wheel setuptools
RUN pip install ${PIP_INS_OPTS} opencv-python || true

RUN pip install ${PIP_INS_OPTS} --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 \
    || echo "Skip DALI installation (CUDA=10.0)"


ARG NNABLA_VER
RUN pip install ${PIP_INS_OPTS} nnabla-ext-cuda102-nccl2-mpi2.1.1==${NNABLA_VER} nnabla_converter==${NNABLA_VER}

COPY --from=flatc /usr/local/bin/flatc /usr/local/bin/flatc

# Solve nccl error that No space left on device
# while creating shared memory segment.
RUN echo NCCL_SHM_DISABLE=1 >> /etc/nccl.conf
RUN echo NCCL_P2P_LEVEL=SYS >> /etc/nccl.conf

# Prepare default user
RUN useradd -m nnabla
RUN chown -R nnabla:nnabla /home/nnabla

USER nnabla
WORKDIR /home/nnabla
CMD [ "/bin/bash" ]


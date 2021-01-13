# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

FROM ubuntu:18.04 as openmpi

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gfortran \
        libibverbs-dev

RUN apt-get install -y --no-install-recommends ca-certificates

RUN mkdir /tmp/openmpi
RUN curl https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.6.tar.bz2 -o /tmp/openmpi/openmpi-3.1.6.tar.bz2
RUN tar Cxvf /tmp/openmpi /tmp/openmpi/openmpi-3.1.6.tar.bz2
RUN cd tmp/openmpi/openmpi-3.1.6 \
    && ./configure \
        --prefix=/opt/openmpi --enable-orterun-prefix-by-default --with-sge \
        CC=gcc \
        CXX=g++ \
        F77=gfortran \
        FC=gfortran \
    && make -j8 \
    && make install

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && PATH=/opt/miniconda3/bin:$PATH \
    && conda install python=3.7 \
    && conda install pip wheel \
    && conda install opencv || true \
    && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 \
    || echo "Skip DALI installation (CUDA=10.0)" \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/bin:$PATH

COPY --from=openmpi /opt/openmpi /opt/openmpi
ENV PATH /opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/openmpi/lib:$LD_LIBRARY_PATH


ARG NNABLA_VER
RUN pip install nnabla-ext-cuda100-nccl2-mpi3.1.6==${NNABLA_VER} nnabla_converter==${NNABLA_VER}

# Prepare default user
RUN useradd -m nnabla
RUN chown -R nnabla:nnabla /home/nnabla

USER nnabla
WORKDIR /home/nnabla
CMD [ "/bin/bash" ]


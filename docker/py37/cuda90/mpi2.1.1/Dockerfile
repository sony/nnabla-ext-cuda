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

FROM ubuntu:16.04 as openmpi

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gfortran

RUN apt-get install -y --no-install-recommends ca-certificates

RUN mkdir /tmp/openmpi
RUN curl https://download.open-mpi.org/release/open-mpi/v2.1/openmpi-2.1.1.tar.bz2 -o /tmp/openmpi/openmpi-2.1.1.tar.bz2
RUN tar Cxvf /tmp/openmpi /tmp/openmpi/openmpi-2.1.1.tar.bz2
RUN cd tmp/openmpi/openmpi-2.1.1 && ./configure --prefix=/opt/openmpi CC=gcc CXX=g++ F77=gfortran FC=gfortran && make && make install

FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

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
    && conda install pip wheel opencv \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/bin:$PATH

COPY --from=openmpi /opt/openmpi /opt/openmpi
ENV PATH /opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/openmpi/lib:$LD_LIBRARY_PATH

RUN pip3 install -U nnabla-ext-cuda90-nccl2-mpi2.1.1

# Prepare default user
RUN useradd -m nnabla
RUN chown -R nnabla:nnabla /home/nnabla

USER nnabla
WORKDIR /home/nnabla
CMD [ "/bin/bash" ]

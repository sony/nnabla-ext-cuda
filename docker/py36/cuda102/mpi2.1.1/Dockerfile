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

FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       openmpi-bin \
       openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && PATH=/opt/miniconda3/bin:$PATH \
    && conda install python=3.6 \
    && conda install pip wheel \
    && conda install opencv || true \
    && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda102 \
    || echo "Skip DALI installation (CUDA=10.2)" \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install -U nnabla-ext-cuda102-nccl2-mpi2.1.1 nnabla_tf_converter

# Prepare default user
RUN useradd -m nnabla
RUN chown -R nnabla:nnabla /home/nnabla

USER nnabla
WORKDIR /home/nnabla
CMD [ "/bin/bash" ]


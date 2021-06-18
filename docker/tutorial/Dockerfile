FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

RUN eval ${APT_OPTS} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       libglib2.0-0 \
       libgl1-mesa-glx \
       python3.8 \
       python3-pip \
       git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0

RUN pip3 install ${PIP_INS_OPTS} --upgrade pip
RUN pip install ${PIP_INS_OPTS} wheel setuptools
RUN pip install ${PIP_INS_OPTS} opencv-python || true

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 \
    || echo "Skip DALI installation (CUDA=11.0)"

ARG NNABLA_VER
RUN pip install ${PIP_INS_OPTS} nnabla-ext-cuda110==${NNABLA_VER}
RUN pip install ${PIP_INS_OPTS} jupyter sklearn imageio matplotlib

RUN git clone --depth=1 https://github.com/sony/nnabla && mv nnabla/tutorial . && rm -rf nnabla
RUN git clone --depth=1 https://github.com/sony/nnabla-examples && rm -rf nnabla-examples/.git  && mv nnabla-examples tutorial
COPY ./run-nnabla-examples.ipynb tutorial/

WORKDIR /tutorial

# The following command doesn't make the container ready to use NVIDIA drivers
# when jupyter is launched. We are not sure why.
# CMD jupyter notebook --ip=* --port=8888 --allow-root --no-browser --NotebookApp.token='nnabla'

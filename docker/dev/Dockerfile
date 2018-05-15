FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    clang-format-3.8 \
    cmake \
    curl \
    g++ \
    git \
    make \
    python3-dev \
    python3-pip \
    python3-setuptools \
    unzip \
    wget \
    ca-certificates \
    bzip2 \
    zip

RUN curl -L https://github.com/google/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip -o /tmp/protoc-3.1.0-linux-x86_64.zip \
    && cd /usr/local \
    && unzip /tmp/protoc-3.1.0-linux-x86_64.zip \
    && chmod 755 /usr/local/bin/protoc \
    && rm -f /tmp/protoc-3.1.0-linux-x86_64.zip readme.txt

# Conda environment for Python3 versions
RUN umask 0 && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
     && rm -rf Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

ADD python/requirements.txt /tmp/deps/

RUN conda create -n py2.7 python=2.7 \
     && bash -c "source activate py2.7 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/requirements.txt"

RUN conda create -n py3.4 python=3.4 \
     && bash -c "source activate py3.4 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/requirements.txt"

RUN conda create -n py3.5 python=3.5 \
     && bash -c "source activate py3.5 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/requirements.txt"

RUN conda create -n py3.6 python=3.6 \
     && bash -c "source activate py3.6 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/requirements.txt"

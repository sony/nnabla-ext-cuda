ARG BASE
FROM ${BASE}

RUN cd /tmp \
	&& apt-get update \
	&& apt-get install -y wget bzip2 libopenmpi-dev openmpi-bin \
        && rm -rf /var/lib/apt/lists/* \
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

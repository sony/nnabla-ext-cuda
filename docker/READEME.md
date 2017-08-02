# NNabla CUDA extension Dockers

All docker files are based on NVidia docker images
that are run with nvidia-docker command.

## Image Tags Hosted on DockerHub

### Latest (default): `nnabla/nnabla-ext-cuda`

This image contains the latest nnabla Python package
with the CUDA extension working with Python3.

```
nvidia-docker run -it --rm nnabla/nnabla-ext-cuda
```

### Python2: `nnabla/nnabla:py2`

This image contains the latest nnabla Python package
with the CUDA extension working with Python2.

```
nvidia-docker run -it --rm nnabla/nnabla-ext-cuda:py2
```

### Tutorial: `nnabla/nnabla:tutorial`

This image contains the latest NNabla with the CUDA extension
and its tutorials on Python3.
The following command runs a jupyter server listening 8888 on the host OS.

```
nvidia-docker run -it --rm -p 8888:8888 nnabla/nnabla-ext-cuda:tutorial jupyter notebook --ip=* --allow-root --NotebookApp.token=nnabla
```

You can connect the server with your browser by accessing
`http://<Host OS address>:8888`. The login password is `nnabla`.

## Dockerfiles for Developers

### Dev: `dev/Dockerfile`

Dockerfile used to create an image containing requirements for
building NNabla C++ libraries and Python package with the CUDA extension.

This must be build at the root directory of nnabla-ext-cuda.

```
docker build -t local/nnabla-ext-cuda:dev -f dev/Dockerfile ../
```

### Dist: `dist/Dockerfile`

Dockerfile for creating an image for building Python package wheels
for many linux distributions.

```
docker build -t local/nnabla-ext-cuda:dist dist
```

TODO: Write more details.

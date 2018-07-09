# NNabla CUDA extension Dockers

All docker files are based on NVidia docker images
that run by nvidia-docker command.

## Image Tags Hosted on DockerHub

The available tags are as following.

| Tag    | CUDA runtime | CUDNN | Python | **CUDA driver** | Dockerfile location |
| ------ |:------------:|:-----:|:------:|:---------------:|:------------------- |
| latest | 8.0          | 7.1   | 3.5    | >=361           | py3/cuda80/         |
| 92     | 9.2          | 7.1   | 3.5    | >=396           | py3/cuda92/         |
| py2    | 8.0          | 7.1   | 2.7    | >=361           | py2/cuda80/         |
| py2-92 | 9.2          | 7.1   | 2.7    | >=396           | py2/cuda92/         |

The docker image can be executed as below.

```
nvidia-docker run <options> nnabla/nnabla-ext-cuda:<tag> <command>
```

You can also build docker images from Dockerfiles located in this folder (describe the above table).

```
docker build <options> -t <image name>:<tag> <Dockerfile folder>
```

## Tutorial image

The image contains nnabla Python (3.5)  with CUDA extension (CUDA8.0 and CUDNN7.1) and [nnabla-examples](https://github.com/nnabla-examples/) repository.
The following command runs a jupyter server listening 8888 port on the host OS.

```
nvidia-docker run -it --rm -p 8888:8888 nnabla/nnabla-ext-cuda:tutorial jupyter notebook --ip=* --allow-root --NotebookApp.token=nnabla
```

You can connect the jupyter server with your browser by accessing
`http://<Host OS address>:8888`. The login password is `nnabla`.

After logging in, the page lists a directory that contains jupyter `.ipynb` tutorials and the `nnabla-examples/` foler.
You can open any tutorial by clicking a `.ipynb` file.
A DCGAN in `nnabla-examples` is demonstrated in `run-nnabla-examples.ipynb`.

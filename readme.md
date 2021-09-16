# A CUDA Extension of Neural Network Libraries

This repository provides an official CUDA/cuDNN-accelerated extension of the
[Neural Network Libraries](https://github.com/sony/nnabla/) deep learning framework.

In order to use it, the default context needs to be changed from `'cpu'` to
`cudnn'`:
```python
from nnabla.ext_utils import get_extension_context

ctx = get_extension_context('cudnn', device_id='0')
nn.set_default_context(ctx)
```

Float 16-bit precision (fp16, half) can also be used by setting `type_config` options as following.

```python
ctx = get_extension_context('cudnn', device_id='0', type_config='half')
```

See [Mixed precision training tutorial](http://nnabla.readthedocs.io/en/latest/python/tutorial/mixed_precision_training.html) for a stable training technique with fp16.

Currently, the binary package install manual and the usage documentation are integrated into the [NNabla's documentation](http://nnabla.readthedocs.io/en/latest/).
For build instructions, see below.

* [Build CUDA extension](doc/build/README.md)

## Performance notes

### Automatic Convolution algorithm selection

If CUDNN is enabled, the extension library uses the specific Convolution algorithms pre-optimized by CUDNN.

Optionally, this library can automatically select the fastest algorithms for your own network using the given configuration of parameters (filter size, stride, dilation, pad, etc), by exhaustively executing and measuring the time of each computation of algorithms (`cudnnFindConvolution*Algorithm`). The best algorithm will be cached, then re-used when an identical configuration is passed to our Convolution interface. It is very powerful in speed, even in non-static (dynamic) neural network. This mode becomes enabled by setting an environment variable `NNABLA_CUDNN_ALGORITHM_BY_HEURISTIC` 0.

However, it often consumes much memory due to a big workspace memory required by automatically found algorithms, and sometimes doesn't work on a GPU with small memory. To avoid this, you can specify the limit of the workspace size by setting an environment variable `NNABLA_CUDNN_WORKSPACE_LIMIT` (in bytes) read at runtime (not compilation time). For example, `NNABLA_CUDNN_WORKSPACE_LIMIT=134217728` limits the workspace size up to 128 MB. The default value is `-1` which means there is no limit of the workspace size.

In some cases it may be desired to restrict the automatic search for CUDNN Convolution algorithms to those that give deterministic (reproducable) results. This can be achived by setting an environment variable `NNABLA_CUDNN_DETERMINISTIC` to some value other than `0`.

### TensorFloat-32 (TF32)

In NNabla, the environment variable `NNABLA_CUDA_ALLOW_TF32` controls whether TF32 (about TF32, see [a blog post](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/) from NVIDIA) is allowed to be used. If `NNABLA_CUDA_ALLOW_TF32` is not set (default) or 0, TF32 is disabled. Otherwise, it is enabled. `NNABLA_CUDA_ALLOW_TF32` always takes priority of `NVIDIA_TF32_OVERRIDE`. `NNABLA_CUDA_ALLOW_TF32` is only evaluated when initializing NNabla CUDA extension. If it is changed within the user program, the behavior is undefined.


## FAQ

No FAQ so far.

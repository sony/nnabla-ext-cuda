# A CUDA Extension of Neural Network Libraries

This repository provides an official CUDA/cuDNN-accelerated extension of the
[Neural Network Libraries](https://github.com/sony/nnabla/) deep learning framework.

In order to use it, the default context needs to be changed from `'cpu'` to
`'cuda.cudnn'`:
```python
from nnabla.contrib.context import extension_context

ctx = extension_context('cuda.cudnn', device_id=args.device_id)
nn.set_default_context(ctx)
```

Currently, the binary package install manual and the usage documentation are integrated into the [NNabla's documentation](http://nnabla.readthedocs.io/en/latest/).
For build instructions, see below.

* [Build CUDA extension](doc/build/README.md)

## FAQ

* Coming soon

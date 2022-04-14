# Copyright 2019,2020,2021 Sony Corporation.
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

import nvidia.dali
from nvidia.dali.pipeline import Pipeline


def feed_ndarray(dali_tensor, arr, ctx, non_blocking=False):
    """
    Copy contents of DALI tensor to NNabla's NdArray.

    Args:
        dali_tensor(nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU) : 
            Tensor from which to copy
        arr (nnabla.NdArray):
            Destination of the copy
        ctx (nnabla.Context):
            CPU context or GPU context. It must match dali tensor device.
        non_blocking (bool, optional):
            If True, copy from a DALI tensor to a nnabla tensor is performed
            asynchronously wrt host.
    """
    import ctypes
    import numpy as np
    from nvidia.dali.types import to_numpy_type
    assert dali_tensor.shape() == list(arr.shape), \
        ("Shapes do not match: DALI tensor has size {0}"
         ", but NNabla Tensor has size {1}".format(dali_tensor.shape(), list(arr.size)))
    dtype = to_numpy_type(dali_tensor.dtype)
    # turn raw int to a c void pointer
    # NOTE : If data_ptr() is called with write_only=False, Array::zero() is executed.
    # c void pointer obtained here is used by DALI, but it may be accessed at the same time by different
    # cuda stream than CudaArray::zero(). To prevent this conflict, write_only must be True.
    c_type_pointer = ctypes.c_void_p(arr.data_ptr(dtype, ctx, True))
    kw = {}
    if non_blocking:
        kw.update(dict(non_blocking=non_blocking))
    dali_tensor.copy_to_external(c_type_pointer, **kw)
    return arr


class DaliIterator(object):
    '''A simple DALI iterator from a DALI pipeline.

    Args:
        pipeline (~nvidia.dali.pipeline.Pipeline):
            A DALI pipeline. A member method `.build()` is called in
            initialization of this class.

        non_blocking (bool, optional):
            If True, copy from a DALI tensor to a nnabla tensor is performed
            asynchronously wrt host. Default is False (synchronous).
            Do not use True if you are not sure what happens.
            It should be safe if device null stream is always synchronized to
            host thread between two `.next()` calls.

    '''

    def __init__(self, pipeline, non_blocking=False):

        # TODO: Enable non_blocking option when it's in DALI.
        assert not non_blocking,\
            "The option `non_blocking` is not in DALI yet (https://github.com/NVIDIA/DALI/pull/1254)."
        self.pipeline = pipeline
        self.pipeline.build()
        self.non_blocking = non_blocking

    def next(self):
        from nnabla.ext_utils import get_extension_context
        from nnabla import NdArray
        from nvidia.dali.backend import TensorCPU

        try:
            outputs = self.pipeline.run()
        except StopIteration:
            # Depending on implementation, StopIteration is raised when you reach the end of an epoch.
            # Automatically reset to restart a new epoch.
            self.pipeline.reset()
            outputs = self.pipeline.run()

        # force outputs to be list object.
        if not hasattr(outputs, '__iter__'):
            outputs = [outputs]

        # Get current cpu&gpu context first.
        cpu_ctx = get_extension_context("cpu")
        device_id = self.pipeline.device_id
        assert device_id >= 0, "Negative device_id is set in pipeline."
        gpu_ctx = get_extension_context('cuda', device_id=str(device_id))

        ret = []
        for output in outputs:
            # check all outputs are dense tensor. (= Each Output has the same shape along batch axis.)
            if not output.is_dense_tensor:
                raise ValueError(
                    "Different shape of data are included in a single batch.")

            tensor = output.as_tensor()
            arr = NdArray(tensor.shape())
            ctx = gpu_ctx
            non_blocking = self.non_blocking
            if isinstance(tensor, TensorCPU):
                ctx = cpu_ctx
                non_blocking = False

            ret.append(feed_ndarray(
                tensor, arr, ctx, non_blocking))

        return ret

    def __next__(self):
        return self.next()


class DataIteratorPipeline(Pipeline):

    '''Creates a DALI pipeline from nnabla DataIterator.

    During computation, it asynchronously transfers data for next mini-batch
    from DataIterator to GPU memory on a device specified by device_id.

    Example:

        .. code-block:: python

            import numpy as np
            from nnabla.utils.data_iterator import data_iterator_simple
            from nnabla_ext.cuda.experimental import dali_iterator

            # Create an data iterator which yeilds two `numpy.ndarray`s.
            nn_it = data_iterator_simple(
                lambda i: (np.full((2, 3), 0, dtype=np.float32),
                           np.full((2, 3), i, dtype=np.int32)),
                512, 32, shuffle=True)

            # Create a pipeline from data iterator
            p = dali_iterator.DataIteratorPipeline(nn_it, device_id=0)
            dali_it = dali_iterator.DaliIterator(p)

            # In a training loop
            for i in range(args.max_iter):

                # .next() returns two `nnabla.NdArray`s
                a1, a2 = dali_it.next()

                # Suppose we have two input `Variable`s
                x1.data = a1
                x2.data = a2

                loss.forward()
                ...

    Args:
        it (:obj:`~nnabla.utils.data_iterator.DataIterator` like object):
            An iterator returns a tuple of data by `.next()` method.
        num_threads (int): Fed into Pipeline. See DALI's doc.
        device_id (int): Fed into Pipeline. See DALI's doc.
        seed (int): Fed into Pipeline. See DALI's doc.
        batch_size (int, optional):
            The batchsize fed into Pipeline class. If None (default),
            is determined by a first dimension of a first element
            in a tuple returned by a data iterator.

    '''

    def __init__(self, it, num_threads=2, device_id=0, seed=726, batch_size=None):
        import nvidia.dali.ops as ops
        # Use first mini-batch to see how iterator returns arrays
        values = it.next()
        self.initial_values = values  # Use it at first batch
        # Assuming batch size is at first dimension
        if batch_size is None:
            batch_size = values[0].shape[0]
        super(DataIteratorPipeline, self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=seed)
        self.it = it
        self.inputs = []
        for _ in range(len(values)):
            self.inputs.append(ops.ExternalSource())

    def define_graph(self):
        self.graph_inputs = []
        ret = tuple()
        for i in self.inputs:
            g = i()
            self.graph_inputs.append(g)
            # Prefetch to GPU
            ret += (g.gpu(),)
        return ret

    def iter_setup(self):
        if self.initial_values is not None:
            values = self.initial_values
            self.initial_values = None
        else:
            values = self.it.next()
        for g, v in zip(self.graph_inputs, values):
            self.feed_input(g, v)


def create_dali_iterator_from_data_iterator(
        it, num_threads=2, device_id=0, seed=726, batch_size=None, non_blocking=False):
    '''
    A helper function which creates a `DaliIterator` from a data iterator.

    See `DataIteratorPipeline` and `DaliIterator` for argument definitions.

    Example:

        .. code-block:: python

            import numpy as np
            from nnabla.utils.data_iterator import data_iterator_simple
            from nnabla_ext.cuda.experimental import dali_iterator

            # Create an data iterator which yeilds two `numpy.ndarray`s.
            nn_it = data_iterator_simple(
                lambda i: (np.full((2, 3), 0, dtype=np.float32),
                           np.full((2, 3), i, dtype=np.int32)),
                512, 32, shuffle=True)

            # Create a pipeline from data iterator
            dali_it = dali_iterator.create_dali_iterator_from_data_iterator(
                nn_it, device_id=0, non_blocking=False)

            # In a training loop
            for i in range(args.max_iter):

                # .next() returns two `nnabla.NdArray`s
                a1, a2 = dali_it.next()

                # Suppose we have two input `Variable`s
                x1.data = a1
                x2.data = a2

                loss.forward()
                ...

    '''
    p = DataIteratorPipeline(it, num_threads, device_id, seed)
    return DaliIterator(p, non_blocking=non_blocking)

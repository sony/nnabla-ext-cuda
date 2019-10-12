# Copyright (c) 2019 Sony Corporation. All Rights Reserved.
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
    assert dali_tensor.shape() == list(arr.shape), \
        ("Shapes do not match: DALI tensor has size {0}"
         ", but NNabla Tensor has size {1}".format(dali_tensor.shape(), list(arr.size)))
    dtype = np.dtype(dali_tensor.dtype())
    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr(dtype, ctx))
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

        outputs = self.pipeline.run()

        # force outputs to be list object.
        if not isinstance(outputs, list):
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

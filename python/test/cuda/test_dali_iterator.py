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

from __future__ import print_function

import pytest
import numpy as np
import nnabla as nn

from nnabla_ext.cuda.experimental import dali_iterator

if dali_iterator.enabled:
    from nvidia.dali.pipeline import Pipeline

    class IteratorPipeline(Pipeline):
        def __init__(self, it, num_threads=2, device_id=0, seed=12):
            import nvidia.dali.ops as ops
            # Use first mini-batch to see how iterator returns arrays
            values = it.next()
            self.initial_values = values  # Use it at first batch
            # Assuming batch size is at first dimension
            batch_size = values[0].shape[0]
            super(IteratorPipeline, self).__init__(
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


@pytest.mark.skipif("not dali_iterator.enabled")
def test_dali_iterator():

    class Iterator1(object):
        def __init__(self, shape1, shape2):
            self.i = 0
            self.shape1 = shape1
            self.shape2 = shape2

        def next(self):
            df = np.full(self.shape1, self.i, dtype=np.float32)
            di = np.full(self.shape2, self.i, dtype=np.int32)
            self.i += 1
            return df, di

    shape1 = (2, 3)
    shape2 = (2, 2)
    it = Iterator1(shape1, shape2)
    for i in range(5):
        df, di = it.next()
        assert df.shape == shape1
        assert di.shape == shape2

    it = Iterator1(shape1, shape2)
    p = IteratorPipeline(it)
    # Testing non_blocking=True because we know it's safe in the following loop.
    # --> TODO: Noticed that non_blocking option is not suppported as of 2019/10/11.
    dali_it = dali_iterator.DaliIterator(p, False)

    for i in range(5):
        ddf, ddi = dali_it.next()
        assert isinstance(ddf, nn.NdArray)
        assert isinstance(ddi, nn.NdArray)
        assert ddf.dtype == np.float32
        assert ddi.dtype == np.int32
        # The following synchronizes null stream to host.
        # So, it's safe in non_blocking mode.
        np.testing.assert_allclose(ddf.get_data('r'), i)
        np.testing.assert_equal(ddi.get_data('r'), i)

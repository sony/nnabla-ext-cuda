# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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
import nnabla.functions as F


@pytest.mark.parametrize("m", [1, 2, 3])
def test_cuda_large_blocks(cuda_test_opts, m):
    if cuda_test_opts.disable_test_large_blocks:
        pytest.skip('`--disable-test-large-blocks` is passed')
    CUDA_THREAD_PER_BLOCK = 512
    CUDA_MAX_BLOCKS = 65536
    size = CUDA_MAX_BLOCKS * CUDA_THREAD_PER_BLOCK * m + 3
    print("Variable size:", size)
    x = np.zeros((size,), np.float32)
    v = nn.Variable(x.shape)
    v.d = x
    from nnabla.ext_utils import get_extension_context
    with nn.context_scope(get_extension_context('cuda')):
        y = F.relu(v)
        y.forward()

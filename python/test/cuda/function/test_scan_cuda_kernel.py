# Copyright 2021 Sony Group Corporation.
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

import pytest
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from nnabla.testing import assert_allclose
from refs import cumsum as ref_cumsum

# The test cases must be examined and adjasted after modifying the scan CUDA kernel.


class Case:
    def __init__(self, shape, axis, rtol=1e-6):
        # rtol (relative tolerance) 1e-6 is default for assert_allclose
        self.shape = shape
        self.axis = axis
        self.rtol = rtol

    # Print this message by pytest when a test fails.
    def __repr__(self):
        return 'Case(shape=' + str(self.shape) + \
               ' axes=' + str(self.axis) + \
               ', rtol=' + str(self.rtol) + ')'


# Error torelances are selected manually for each case because it is difficult
# to select them generally for random input values.
test_cases = [
    # --------------------------------
    # General cases
    # --------------------------------
    # Large cases
    Case((512 * 512, 128), 0, 3e-5),
    Case((128, 512 * 512), 1, 3e-5),

    # Corner cases
    Case((1,), 0),
    Case((1, 1, 1, 1), 2),
    Case((1, 1, 1, 1), 3),
    Case((1, 123), 0),
    Case((123, 1), 1),
    Case((1, 123, 1), 0),
    Case((1, 123, 1), 1),
    Case((1, 123, 1), 2),
    Case((1, 63), 1),
    Case((63, 1), 0),
    Case((1, 63, 1), 0),
    Case((1, 63, 1), 1),
    Case((1, 63, 1), 2),
    # Parallel inter-block algorithm selection border
    Case((1, 1027), 1),
    Case((1, 1028), 1),
    Case((1, 1029), 1),
    # Sequential inter-block algorithm selection border
    Case((127, 1), 0),
    Case((128, 1), 0),
    Case((129, 1), 0),

    # Large dimension cases
    Case((2, 3, 5, 7, 11, 3, 1, 4, 1, 5), 0),
    Case((2, 3, 5, 7, 11, 3, 1, 4, 1, 5), 3),
    Case((2, 3, 5, 7, 11, 3, 1, 4, 1, 5), 9),

    # --------------------------------
    # Kernel wise cases
    # --------------------------------

    # Parallel scan with single kernel
    Case((3,), 0),
    Case((67,), 0),
    Case((67, 1), 0),
    Case((1, 67, 1), 1),
    Case((123, 456, 1), 1, 1.5e-6),
    Case((456, 123, 1), 1),
    Case((2048,), 0),

    # Parallel scan with inter-block kernels
    Case((2049,), 0),  # Minimal case
    Case((2048+123), 0),
    Case((2048+123, 1), 0),
    Case((1, 2048+123), 1),
    Case((1, 2048+123, 1), 1),
    Case((2048+123, 2048+456, 1), 1, 3e-6),

    # Sequential scan with single kernel
    Case((3, 3), 0),
    Case((123, 3), 0),
    Case((123, 234, 345), 0, 3e-6),
    Case((345, 123, 234), 1, 3e-6),

    # Sequential scan with inter-block kernels
    Case((123, 234, 345), 1, 3e-6),
    Case((345, 123, 234), 0, 3e-6),
    Case((2048+123, 53), 0, 3e-6),
]


# The negative values during Sum could induce the unecessary numerical
# instability for test. In this case, the option with_negative=False can be
# used.
def create_inputs(shape, seed, with_negative):
    # np.random.RandomState(seed).randn(*shape).astype(np.float32) always
    # generate float64, consuming larger memory. It can be avoided by
    # the following code.
    np_input = np.random.default_rng(
        seed=seed).standard_normal(size=shape, dtype='float32')
    if not with_negative:
        np_input = np.abs(np_input)
    v_input = nn.Variable.from_numpy_array(np_input)
    return np_input, v_input


# Testing the values of the reduction CUDA kernel by using Sum.
@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_scan_cuda_kernel_value(seed, test_case, exclusive, reverse):
    ctx = get_extension_context('cuda', type_config='float')
    with nn.context_scope(ctx):
        np_input, v_input = create_inputs(test_case.shape, seed, False)
        v_output = F.cumsum(v_input, test_case.axis, exclusive, reverse)
        v_output.forward()
    ref_val = ref_cumsum(np_input, test_case.axis, exclusive, reverse)
    assert_allclose(v_output.d, ref_val, rtol=test_case.rtol, atol=0)

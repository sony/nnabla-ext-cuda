# Copyright 2021 Sony Corporation.
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


def ref_cumsum(x, axis, exclusive, reverse):
    if reverse:
        x = np.flip(x, axis)
    y = np.cumsum(x, axis)
    if exclusive:
        # Make first element to zero on the axis.
        pad_width = np.repeat([[0, 0]], x.ndim, axis=0)
        pad_width[axis][0] = 1
        y = np.pad(y, pad_width)
        take_range = range(y.shape[axis]-1)
        y = np.take(y, take_range, axis)
    if reverse:
        y = np.flip(y, axis)
    return y


# Error torelances are selected manually for each case because it is difficult
# to select them generally for random input values.
test_cases = [
    Case((2, 3, 4, 5), 0),
    Case((2, 3, 4, 5), 1),
    Case((2, 3, 4, 5), 2),
    Case((2, 3, 4, 5), 3),
    Case((1,), 0),
    Case((1, 1, 1, 1), 2),
    Case((1, 1, 1, 1), 3),
    Case((64+7,), 0),
    Case((1024,), 0),
    Case((1024 + 123,), 0),
    Case((123, 1024 + 123,), 0),
    Case((32, 1024+2,), 1, 2e-6),
    Case((123, 1024 + 123,), 1, 2e-6),
    Case((5, 3), 0),
    Case((5, 3), 1),
    Case((2001, 3), 0, 2e-6),
    Case((2001, 3), 1),
    Case((3, 2001), 0),
    Case((3, 2001), 1, 2e-6),
    Case((1555, 2001), 0, 2.5e-6),
    Case((1555, 2001), 1, 3e-6),
]


# # The negative values during Sum could induce the unecessary numerical
# # instability for test. In this case, the option with_negative=False can be
# # used.
# def create_inputs(shape, seed, with_negative):
#     # np.random.RandomState(seed).randn(*shape).astype(np.float32) always
#     # generate float64, consuming larger memory. It can be avoided by
#     # the following code.
#     np_input = np.random.default_rng(
#         seed=seed).standard_normal(size=shape, dtype='float32')
#     if not with_negative:
#         np_input = np.abs(np_input)
#     v_input = nn.Variable.from_numpy_array(np_input)
#     return np_input, v_input
def create_inputs(shape, seed, with_negative):
    # np.random.RandomState(seed).randn(*shape).astype(np.float32) always
    # generate float64, consuming larger memory. It can be avoided by
    # the following code.
    np_input = np.random.default_rng(
        seed=seed).standard_normal(size=shape, dtype='float32')
    # np_input = np.ones(shape, dtype='float32')
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

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
from refs import cumprod as ref_cumprod


# Calculate cumprod backward in O(N).
# See `nnabla/src/nbla/function/generic/cumprod.cpp` for the detail algorithm.
def ref_grad_cumprod(x, dy, axis, exclusive, reverse, **kw):
    zero_mask = np.where(x == 0, 1, 0)

    # Create masks for following calculation
    # [1, ... , 1, 0, 0, ... , 0]
    before_first_zero_mask = np.where(ref_cumsum(
        zero_mask, axis, False, reverse) == 0, 1, 0)
    # [0, ... , 0, 0, 1, ... , 1]
    after_first_zero_mask = np.where(ref_cumsum(
        zero_mask, axis, True, reverse) == 0, 0, 1)
    # [0, ... , 0, 1, 0, ... , 0]
    first_zero_mask = np.where(
        before_first_zero_mask + after_first_zero_mask == 0, 1, 0)

    masked_x = np.where(first_zero_mask == 1, 1, x)

    # Check if masks are correctly generated.
    assert (np.all(first_zero_mask + before_first_zero_mask +
                   after_first_zero_mask == 1))
    assert (np.all(masked_x - x == first_zero_mask))
    assert (np.all(x * first_zero_mask == 0))

    normal_cumprod = ref_cumprod(x, axis, exclusive, reverse)
    masked_cumprod = ref_cumprod(masked_x, axis, exclusive, reverse)

    cumprod_dy = normal_cumprod * dy
    masked_cumprod_dy = masked_cumprod * dy

    reversed_cumprod_dy = ref_cumsum(cumprod_dy, axis, exclusive, not reverse)
    reversed_masked_cumprod_dy = ref_cumsum(
        masked_cumprod_dy, axis, exclusive, not reverse)

    # Calculate dx
    full_masked_x = np.where(x == 0, 1, x)  # Avoid generating `nan`
    dx_before_zero_pos = reversed_cumprod_dy / \
        full_masked_x * before_first_zero_mask
    dx_zero_pos = reversed_masked_cumprod_dy * first_zero_mask
    dx = dx_before_zero_pos + dx_zero_pos
    return dx


class Case:
    def __init__(self, shape, axis, rtol=1e-6, atol=0):
        # rtol (relative tolerance) 1e-6 is default for assert_allclose
        self.shape = shape
        self.axis = axis
        self.rtol = rtol
        self.atol = atol

    # Print this message by pytest when a test fails.
    def __repr__(self):
        return 'Case(shape=' + str(self.shape) + \
               ' axes=' + str(self.axis) + \
               ', rtol=' + str(self.rtol) + \
               ', atol=' + str(self.atol) + ')'


# Error torelances are selected manually for each case because it is difficult
# to select them generally for random input values.
test_cases = [
    # Test cases must be examined and adjasted after modifying the CumProd backward implementation.
    # --------------------------------
    # Cases for the naive kernel implementation
    # --------------------------------
    Case((512, 1, 2), 1),  # Minimal case
    Case((1, 1024), 0),
    Case((123, 1024), 0, atol=2e-5),
    Case((128, 1024), 0, atol=2e-5),  # Largest scan size
    Case((234, 123, 345), 1, atol=2e-5),  # With outer and inner dimension

    # --------------------------------
    # Cases for the multiple kernel composition implementation
    # --------------------------------
    Case((1,), 0),  # Minimal case
    Case((123,), 0, atol=3e-7),
    Case((12345,), 0, atol=3e-7),  # Large case
    Case((123, 234, 345), 2, atol=3e-7),  # With outer dimension
]


# The negative values during Prod could induce the unecessary numerical
# instability for test. In this case, the option with_negative=False can be
# used.
def create_input(rng, shape, axis, with_negative, with_mask):
    np_data = rng.randn(*shape)
    if not with_negative:
        np_data = np.abs(np_data)
    if with_mask:
        # Make zero elements with the probability of `1 / x_shape[axis]`.
        # It is the probability of existence of one zero element in each scan axis.
        mask = rng.rand(*shape) > (1.0 / shape[axis])
        np_data = np_data * mask
    v_input = nn.Variable.from_numpy_array(np_data, need_grad=True)
    return np_data, v_input


def create_grad(rng, shape, with_negative):
    # `create_inputs` can be used to generate grad.
    np_grad, _ = create_input(rng, shape, 0, with_negative, False)
    return np_grad


# Testing the values of the reduction CUDA kernel by using Sum.
@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("test_case", test_cases)
@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumprod_backward_cuda_kernel_value(seed, test_case, exclusive, reverse):
    ctx = get_extension_context('cuda', type_config='float')
    rng = np.random.RandomState(seed)
    shape = test_case.shape
    axis = test_case.axis
    with nn.context_scope(ctx):
        np_x, v_x = create_input(rng, shape, axis, False, True)
        v_y = F.cumprod(v_x, test_case.axis, exclusive, reverse)
        v_y.forward()
        np_dy = create_grad(rng, shape, False)
        v_x.grad.zero()
        v_y.backward(np_dy)
    np_dx = ref_grad_cumprod(np_x, np_dy, test_case.axis, exclusive, reverse)
    assert_allclose(v_x.g, np_dx, rtol=test_case.rtol, atol=test_case.atol)

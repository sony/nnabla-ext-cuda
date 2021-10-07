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

# The test cases must be examined and adjasted after modifying the reduction
# CUDA kernel.


class TestCase:
    def __init__(self, shape, axes, type_config, rtol=1e-7):
        # rtol (relative tolerance) 1e-7 is default for assert_allclose
        self.shape = shape
        self.axes = axes
        self.type_config = type_config
        self.rtol = rtol

    # Print this message by pytest when a test fails.
    def __repr__(self):
        return 'TestCase(shape=' + str(self.shape) + \
               ' axes=' + str(self.axes) + \
               ', type_config=' + str(self.type_config) + \
               ', rtol=' + str(self.rtol) + ')'


# Error torelances are selected manually for each case because it is difficult
# to select them generally for random input values.
test_cases = [
    #
    # Float
    #
    # Misaligned memory layout appears whith vectrized load.
    TestCase((2, 3, 4, 5), (3,), 'float'),
    # Misaligned memory layout appears without vectrized load.
    TestCase((2, 3, 4, 5), (2,), 'float'),
    TestCase((2, 1, 4, 5), (2,), 'float'),  # shape with 1
    TestCase((2, 3, 1, 5), (2,), 'float'),  # shape with 1
    TestCase((1, 2, 1, 1, 1, 2), (0, 3), 'float'),  # just copy
    # Many dimensions with x reduction
    TestCase(tuple([2] * 10), (0, 2, 4, 6, 8), 'float', 2e-7),
    # Many dimensions without x reduction
    TestCase(tuple([2] * 10), (1, 3, 5, 7, 9), 'float', 2e-7),
    # Input sizes induce inter-block reduction
    TestCase((10000,), (0,), 'float'),  # Inter-block reduce x
    TestCase((10, 100, 10), (1,), 'float', 5e-7),  # Inter-block reduce y
    # Inter-block reduce x with y reduction
    TestCase((100, 10, 100), (0, 2), 'float', 2e-7),
    #
    # Half (Sum uses float to keep numerical precision. Max uses half directly
    #       because it keeps precision.)
    #
    # Misaligned memory layout appears whith vectrized load.
    TestCase((2, 3, 4, 5), (3,), 'half', 5e-4),
    # Misaligned memory layout appears without vectrized load.
    TestCase((2, 3, 4, 5), (2,), 'half', 5e-4),
    TestCase((2, 1, 4, 5), (2,), 'half', 5e-4),  # shape with 1
    TestCase((2, 3, 1, 5), (2,), 'half', 5e-4),  # shape with 1
    TestCase((1, 2, 1, 1, 1, 2), (0, 3), 'half', 3e-4),  # just copy
    # Many dimensions with x reduction
    TestCase(tuple([2] * 10), (0, 2, 4, 6, 8), 'half', 5e-4),
    # Many dimensions without x reduction
    TestCase(tuple([2] * 10), (1, 3, 5, 7, 9), 'half', 4e-4),
    # Input sizes induce inter-block reduction
    TestCase((10000,), (0,), 'half', 2e-4),  # Inter-block reduce x
    TestCase((10, 100, 10), (1,), 'half', 5e-4),  # Inter-block reduce y
    # Inter-block reduce x with y reduction
    TestCase((100, 10, 100), (0, 2), 'half', 4e-4),

    # TODO: Test grid-strided loop
    # The cases inducing grid-strided loop are not tested because this case
    # consumes too large memory to be executed.
]


# The negative values during Sum could induce the unecessary numerical
# instability for test. In this case, the option with_negative=False can be
# used.
def create_inputs(shape, seed, with_negative):
    # np.random.RandomState(seed).randn(*shape).astype(np.float32) always
    # generate float64, consuming larger memory. It can be avoided by
    # the following code.
    if with_negative:
        np_input = np.random.default_rng(
            seed=seed).standard_normal(size=shape, dtype='float32')
    else:
        np_input = np.abs(np.random.default_rng(
            seed=seed).standard_normal(size=shape, dtype='float32'))
    v_input = nn.Variable.from_numpy_array(np_input)
    return np_input, v_input


# Testing whether the reduction CUDA kernel reduces the all elements.
# The test by a large input with random values could overlook a few missing
# elements.
@pytest.mark.parametrize("test_case", test_cases)
def test_reduction_cuda_kernel_exact(test_case):
    print(test_case)
    ctx = get_extension_context('cuda', type_config=test_case.type_config)
    with nn.context_scope(ctx):
        np_input = np.ones(test_case.shape, dtype='float32')
        v_input = nn.Variable.from_numpy_array(np_input)
        v_output = F.sum(v_input, axis=test_case.axes)
        v_output.forward()
        assert (v_output.d == np.sum(np_input, axis=test_case.axes)).all()


# Testing the values of the reduction CUDA kernel by using Sum.
@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("test_case", test_cases)
def test_reduction_cuda_kernel_value(seed, test_case):
    ctx = get_extension_context('cuda', type_config=test_case.type_config)
    with nn.context_scope(ctx):
        np_input, v_input = create_inputs(test_case.shape, seed, False)
        v_output = F.sum(v_input, axis=test_case.axes)
        v_output.forward()
    assert_allclose(v_output.d, np.sum(
        np_input, axis=test_case.axes), rtol=test_case.rtol, atol=0)


# Testing the indices of the reduction CUDA kernel by using Max.
@pytest.mark.parametrize("seed", [123])
@pytest.mark.parametrize("test_case", test_cases)
def test_reduction_cuda_kernel_index(seed, test_case):
    ctx = get_extension_context('cuda', type_config=test_case.type_config)
    with nn.context_scope(ctx):
        np_input, v_input = create_inputs(test_case.shape, seed, True)
        v_output, v_indices = F.max(
            v_input, axis=test_case.axes, with_index=True)
        v_output.forward()
    assert_allclose(v_output.d, np_input.max(axis=test_case.axes),
                    rtol=test_case.rtol, atol=0)

    # Assuming that cpu backend is testd by nnabla repository.
    ctx = get_extension_context('cpu', type_config=test_case.type_config)
    with nn.context_scope(ctx):
        ref_v_output, ref_v_indices = F.max(
            v_input, axis=test_case.axes, with_index=True)
        ref_v_output.forward()
    assert_allclose(v_output.d, ref_v_output.d, rtol=test_case.rtol, atol=0)
    assert (v_indices.d == ref_v_indices.d).all()


# TODO: Larger input to test the reduction CUDA kernel for Size_t (64 bit).

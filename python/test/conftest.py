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


import pytest


def pytest_addoption(parser):
    parser.addoption('--disable-test-large-blocks', action='store_true', default=False,
                     help='Whether to run test_cuda_large_blocks which consumes quite large memory.')


@pytest.fixture(scope='session')
def cuda_test_opts(request):
    """Parse options and expose as a fixture.

    Returns: NNablaOpts
        An  object which has ext, ext_kwargs, benchmark_output_dir and
        function_benchmark_writer as attributes.
    """
    from collections import namedtuple
    getoption = request.config.getoption
    disable_test_large_blocks = getoption("--disable-test-large-blocks")
    NNablaExtCudaTestOpts = namedtuple("NNablaExtCudaTestOpts",
                                       [
                                           'disable_test_large_blocks',
                                           ])
    return NNablaExtCudaTestOpts(disable_test_large_blocks)

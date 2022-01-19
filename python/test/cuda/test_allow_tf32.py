# Copyright 2021,2022 Sony Group Corporation.
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
import os
import subprocess


def teardown_function(function):
    if os.getenv('NNABLA_CUDA_ALLOW_TF32'):
        del os.environ["NNABLA_CUDA_ALLOW_TF32"]
    if os.getenv('NVIDIA_TF32_OVERRIDE'):
        del os.environ["NVIDIA_TF32_OVERRIDE"]


'''
This test is verify that tf32 is set correctly by the environment variables.
'''


@pytest.mark.parametrize("nnabla_tf32", [None, "0", "1"])
@pytest.mark.parametrize("nvidia_tf32", [None, "0", "1"])
@pytest.mark.parametrize("context", ['cuda', 'cudnn'])
def test_allow_tf32(nnabla_tf32, nvidia_tf32, context):

    if nnabla_tf32:
        os.environ["NNABLA_CUDA_ALLOW_TF32"] = nnabla_tf32
    if nvidia_tf32:
        os.environ["NVIDIA_TF32_OVERRIDE"] = nvidia_tf32

    d = os.path.dirname(os.path.abspath(__file__))
    test_main_path = d + "/test_allow_tf32_main.py"

    if nnabla_tf32 == "1":
        result = subprocess.run(
            ["python", test_main_path, "--context", context, "--allow-tf32"])
    else:
        result = subprocess.run(
            ["python", test_main_path, "--context", context])

    assert result.returncode == 0

# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import os
import platform
import pytest
from subprocess import check_call, call, CalledProcessError


def command_exists(command):
    return call(['which', command]) == 0


def test_use_nvtx():
    script = os.path.join(os.path.dirname(__file__), 'nvtx-check.py')
    check_call(['python', script])


@pytest.mark.skipif(platform.system() != "Linux", reason='Test on Linux only')
def test_use_nsys():
    outfile = 'nvtx-test'
    script = os.path.join(os.path.dirname(__file__), 'nvtx-check.py')

    if not command_exists('nsys'):
        pytest.skip('nsys command does not exist.')

    try:
        check_call(['nsys', 'profile', '-o', outfile,
                   '--export=text', '-f', 'true', 'python', script])
    except CalledProcessError:
        # maybe segfault occured, but it's ok if log is saved correctly.')
        pass

    with open(os.path.join(os.getcwd(), f'{outfile}.txt')) as f:
        lines = f.readlines()

    setup = [line for line in lines if 'nnabla-push-pop-test' in line]
    forward = [line for line in lines if 'nnabla-range-test' in line]

    assert len(setup) != 0
    assert len(forward) != 0

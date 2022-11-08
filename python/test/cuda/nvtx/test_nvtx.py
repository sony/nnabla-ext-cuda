# Copyright 2022 Sony Group Corporation.
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
from nnabla import logger


def command_exists(command):
    return call(['which', command]) == 0


# Current nsys command may fail to generate final output files correctly due to some running environment problems.
# The function will retry 3 times to help generate it from the intermediate file.
def generate_nvtx_txt(outfile, script):
    try:
        check_call(['nsys', 'profile', '-o', outfile,
                   '--export=text', '-f', 'true', 'python', script])
    except CalledProcessError:
        # maybe segfault occured, but it's ok if log is saved correctly.')
        pass

    outfile_txt = os.path.join(os.getcwd(), f'{outfile}.txt')
    # if fail to generate final output .txt and .qdrep files but only intermediate .qdstrm file
    if not os.path.exists(outfile_txt) or os.stat(outfile_txt).st_size == 0:
        retry = 1
        outfile_qdrep = os.path.join(os.getcwd(), f'{outfile}.qdrep')
        interfile_qdstrm = os.path.join(os.getcwd(), f'{outfile}.qdstrm')
        qdstrm_importer = 'QdstrmImporter'
        if not (os.path.exists(interfile_qdstrm) and command_exists(qdstrm_importer)):
            logger.log(
                99, 'No available .qdstrm file or cannot find QdstrmImporter command')
            raise FileNotFoundError
        while True:
            try:
                if os.path.exists(outfile_qdrep):
                    call(['rm', '-f', outfile_qdrep])
                check_call([qdstrm_importer, '--input-file', interfile_qdstrm])
                check_call(['nsys', 'export', '-o', outfile_txt,
                           '--force-overwrite', 'true', '-t', 'text', outfile_qdrep])
                break
            except CalledProcessError:
                retry += 1
                if retry > 3:
                    logger.log(
                        99, 'Could not generate .txt file from intermediate .qdstrm file.')
                    raise


def test_use_nvtx():
    script = os.path.join(os.path.dirname(__file__), 'nvtx-check.py')
    check_call(['python', script])


@pytest.mark.skipif(platform.system() != "Linux", reason='Test on Linux only')
def test_use_nsys():
    outfile = 'nvtx-test'
    script = os.path.join(os.path.dirname(__file__), 'nvtx-check.py')

    if not command_exists('nsys'):
        pytest.skip('nsys command does not exist.')

    generate_nvtx_txt(outfile, script)

    with open(os.path.join(os.getcwd(), f'{outfile}.txt')) as f:
        lines = f.readlines()

    setup = [line for line in lines if 'nnabla-push-pop-test' in line]
    forward = [line for line in lines if 'nnabla-range-test' in line]

    assert len(setup) != 0
    assert len(forward) != 0

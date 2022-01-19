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

import numpy as np
import nnabla as nn
import os
import argparse

'''
This test is called by test_allow_tf32.py to verify that tf32 is set correctly by the environment variables.
In pytest, init_cuda is executed only once throughout the whole tests, and this test case is added as a separate test because the correct behavior could not be confirmed.
'''


def main(args):
    context = args.context

    from nnabla.ext_utils import get_extension_context
    with nn.context_scope(get_extension_context(context)):
        import nnabla_ext.cuda.init as cuda_init
        allow_tf32 = cuda_init.is_tf32_enabled()

        assert args.allow_tf32 == allow_tf32


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-tf32", action='store_true', default=False)
    parser.add_argument("--context", type=str,
                        default="cuda", choices=["cudnn", "cuda"])
    args = parser.parse_args()
    main(args)

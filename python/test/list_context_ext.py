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

import os
import sys
import nnabla as nn


def list(func_name):

    sys.path.append(os.path.join(os.path.dirname(__file__),
                                 '..', '..', 'build-tools', 'code_generator'))
    from load_implements_rst import Implements

    l = [(nn.Context(), func_name)]

    info = Implements().info
    if func_name in info:
        if 'cuda' in info[func_name]:
            import nnabla_ext.cuda
            l.append((nnabla_ext.cuda.context(), func_name + 'Cuda'))
        if 'cudnn' in info[func_name]:
            import nnabla_ext.cuda.cudnn
            l.append((nnabla_ext.cuda.cudnn.context(), func_name + 'CudaCudnn'))
    return l

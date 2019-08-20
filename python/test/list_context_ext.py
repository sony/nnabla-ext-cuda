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

import nnabla as nn
from os.path import join, dirname, realpath

root_dir = realpath(join(dirname(__file__), '..', '..'))


def list(func_name):
    import yaml
    function_types = yaml.load(open(
        join(root_dir, 'build-tools', 'code_generator', 'function_types.yaml'), 'r'))
    function_types_cudnn = yaml.load(open(join(
        root_dir, 'build-tools', 'code_generator', 'function_types_cudnn.yaml'), 'r'))
    solver_types = yaml.load(
        open(join(root_dir, 'build-tools', 'code_generator', 'solver_types.yaml'), 'r'))

    l = [(nn.Context(), func_name)]

    if (func_name in function_types and function_types[func_name]) \
       or (func_name in solver_types and solver_types[func_name]):
        import nnabla_ext.cuda
        l.append((nnabla_ext.cuda.context(), func_name + 'Cuda'))
    if func_name in function_types_cudnn and function_types_cudnn[func_name]:
        import nnabla_ext.cudnn
        l.append((nnabla_ext.cudnn.context(), func_name + 'CudaCudnn'))
    return l

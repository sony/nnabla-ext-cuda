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

import io
import os
from os.path import abspath, dirname, join

# Set path to <NNabla root>/build-tools/code_generator to import the following two
from utils.common import check_update, get_version
from utils.type_conv import type_from_proto
import code_generator_utils as utils

import itertools

here = abspath(dirname(abspath(__file__)))
base = abspath(here + '/../..')


def generate():
    function_info = utils.load_function_info(flatten=True)
    solver_info = utils.load_solver_info()
    function_types = utils.load_yaml_ordered(open(
        join(here, 'function_types.yaml'), 'r'))
    function_types_cudnn = utils.load_yaml_ordered(open(
        join(here, 'function_types_cudnn.yaml'), 'r'))
    solver_types = utils.load_yaml_ordered(open(
        join(here, 'solver_types.yaml'), 'r'))
    function_template = join(
        base, 'src/nbla/cuda/function_types.cpp.tmpl')
    function_template_cudnn = join(
        base, 'src/nbla/cuda/cudnn/function_types.cpp.tmpl')
    solver_template = join(
        base, 'src/nbla/cuda/solver_types.cpp.tmpl')
    init_template = join(
        base, 'src/nbla/cuda/init.cpp.tmpl')
    init_template_cudnn = join(
        base, 'src/nbla/cuda/cudnn/init.cpp.tmpl')
    utils.generate_init(function_info, function_types,
                        solver_info, solver_types, ext_info={}, template=init_template)
    utils.generate_init(function_info, function_types_cudnn,
                        solver_info, solver_types, ext_info={}, template=init_template_cudnn)
    """
    utils.generate_function_types(
        function_info, function_types, ext_info=ext_info, template=function_template)
    utils.generate_solver_types(
        solver_info, solver_types, ext_info=ext_info, template=solver_template)
    utils.generate_version(template=join(
        base, 'python/src/nnabla_ext/%s/_version.py.tmpl' % ext_info['ext_name_snake']), rootdir=base)
    func_src_template = join(
        base,
        'src/nbla/%s/function/generic/function_impl.cpp.tmpl' % ext_info['ext_name_snake'])
    utils.generate_skelton_function_impl(
        function_info, function_types, ext_info, template=func_src_template)
    func_header_template = join(
        base,
        'include/nbla/%s/function/function_impl.hpp.tmpl' % ext_info['ext_name_snake'])
    utils.generate_skelton_function_impl(
        function_info, function_types, ext_info,
        template=func_header_template, output_format='%s.hpp')
    """


if __name__ == '__main__':
    generate()


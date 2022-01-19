# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

from os.path import abspath, dirname, join
import sys

# Set path to <NNabla root>/build-tools/code_generator to import the following two
import code_generator_utils as utils


here = abspath(dirname(abspath(__file__)))
base = abspath(here + '/../..')


def generate(argv):
    function_info = utils.load_function_info(flatten=True)
    solver_info = utils.load_solver_info()
    function_types = utils.load_yaml_ordered(open(
        join(here, 'function_types.yaml'), 'r'))
    function_types_cudnn = utils.load_yaml_ordered(open(
        join(here, 'function_types_cudnn.yaml'), 'r'))
    solver_types = utils.load_yaml_ordered(open(
        join(here, 'solver_types.yaml'), 'r'))
    function_template = join(
        base, 'src/nbla/cuda/function/function_types.cu.tmpl')
    function_template_cudnn = join(
        base, 'src/nbla/cuda/cudnn/function/function_types.cu.tmpl')
    solver_template = join(
        base, 'src/nbla/cuda/solver/solver_types.cu.tmpl')
    init_template = join(
        base, 'src/nbla/cuda/init.cpp.tmpl')
    init_template_cudnn = join(
        base, 'src/nbla/cuda/cudnn/init.cpp.tmpl')
    utils.generate_init(function_info, function_types,
                        solver_info, solver_types, ext_info={}, template=init_template)
    utils.generate_init(function_info, function_types_cudnn,
                        solver_info, solver_types, ext_info={}, template=init_template_cudnn)
    utils.generate_function_types(
        function_info, function_types, ext_info={}, template=function_template, output_format='%s.cu')
    utils.generate_function_types(
        function_info, function_types_cudnn, ext_info={}, template=function_template_cudnn, output_format='%s.cu')
    utils.generate_solver_types(
        solver_info, solver_types, ext_info={}, template=solver_template, output_format='%s.cu')

    if len(argv) == 4:
        version = argv[1]
        cuda_version = argv[2]
        cudnn_version = argv[3]
    else:
        version = 'unknown'
        cuda_version = 'unknown'
        cudnn_version = 'unknown'
        
    utils.generate_version(template=join(
        base, 'python/src/nnabla_ext/cuda/_version.py.tmpl'),
        rootdir=base, version=version, cuda_version=cuda_version, cudnn_version=cudnn_version)
    utils.generate_version(template=join(
        base, 'python/src/nnabla_ext/cudnn/_version.py.tmpl'),
        rootdir=base, version=version, cuda_version=cuda_version, cudnn_version=cudnn_version)
    utils.generate_version(template=join(
        base, 'src/nbla/cuda/version.cpp.tmpl'),
        rootdir=base, version=version, cuda_version=cuda_version, cudnn_version=cudnn_version)

    # Generate function skeletons
    func_src_template = join(
        base,
        'src/nbla/cuda/function/generic/function_impl.cu.tmpl')
    func_src_template_cudnn = join(
        base,
        'src/nbla/cuda/cudnn/function/generic/function_impl.cu.tmpl')
    func_header_template = join(
        base,
        'include/nbla/cuda/function/function_impl.hpp.tmpl')
    func_header_template_cudnn = join(
        base,
        'include/nbla/cuda/cudnn/function/function_impl.hpp.tmpl')
    utils.generate_skeleton_function_impl(
        function_info, function_types, ext_info={},
        template=func_src_template, output_format='%s.cu')
    utils.generate_skeleton_function_impl(
        function_info, function_types, ext_info={},
        template=func_header_template, output_format='%s.hpp')
    utils.generate_skeleton_function_impl(
        function_info, function_types_cudnn, ext_info={},
        template=func_src_template_cudnn, output_format='%s.cu')
    utils.generate_skeleton_function_impl(
        function_info, function_types_cudnn, ext_info={},
        template=func_header_template_cudnn, output_format='%s.hpp')

if __name__ == '__main__':
    generate(sys.argv)

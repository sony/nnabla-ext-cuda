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
import os
import re
import glob
from mako.template import Template
from mako import exceptions

NCCL_H_IN = None

search_dirs = [
    "/usr/include",
    "/usr/local/include"
]

env_path = os.getenv("NCCL_HOME", None)
if env_path:
    search_dirs.append(env_path + '/include')

for nccl_dir in search_dirs:
    nccl_h = os.path.join(nccl_dir, "nccl.h")
    if os.path.exists(nccl_h):
        NCCL_H_IN = nccl_h
        break

if NCCL_H_IN is None:
    raise ValueError("Please install NCCL development package!")

CURRENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DL_NCCL_H_IMPL = os.path.join(CURRENT_PATH, "include", "nbla", "cuda", "communicator", "dl_nccl.h.tmpl")
DL_NCCL_CPP_IMPL = os.path.join(CURRENT_PATH, "src", "nbla", "cuda", "communicator", "dl_nccl.cpp.tmpl")

func_name_extractor = re.compile(r'ncclResult_t\s*([\w]*)\(([\w_*\s\,]*)\);')
argument_extractor = re.compile(r'([\w\_\*]*)\s*([\w_\*]*)')


def extract_func_info(line):
    m = func_name_extractor.match(line)
    func_name = m.group(1)
    arguments = m.group(2)
    func_args = []
    for arg in arguments.split(','):
        arg_segs = arg.split(' ')
        arg_name = arg_segs[-1]
        arg_type = ' '.join(arg_segs[:-1])
        i = 0
        for c in arg_name:
            if c == '*':
                arg_type += c
                i += 1
            else:
                break
        arg_name = arg_name[i:]
        func_args.append((arg_type, arg_name))
    ret_type = 'ncclResult_t'

    func_decl = f"{ret_type} (*)({','.join([t for t, _ in func_args])})"
    prototype = f"{ret_type} (*{func_name})({','.join([f'{t} {n}' for t, n in func_args])})"

    return func_name, func_decl, prototype


class State:
    new_line = 1
    cont_line = 2


source_files = glob.glob(os.path.join(CURRENT_PATH, "src", "nbla", "cuda", "communicator", "*.cu"))

nccl_func_matcher = re.compile(r'(nccl[A-Z][a-z0-9\_A-Z]*)\(')


class State:
    new_line = 1
    cont_line = 2


def collect_nccl_function_reference():
    def grasp_function(functions, line):
        m = nccl_func_matcher.findall(line)
        if m is not None:
            functions += m
        return functions

    functions = []
    for source_file in source_files:
        state = State.new_line
        current_line = ''
        with open(source_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if state == State.new_line:
                    if line.endswith(';'):
                        functions = grasp_function(functions, line)
                    else:
                        state = State.cont_line
                        current_line = line
                elif state == State.cont_line:
                    current_line += line
                    if line.endswith(';'):
                        functions = grasp_function(functions, current_line)
                        state = State.new_line
    return set(functions)


def grasp_function_list_from_header_file():
    function_list = []
    state = State.new_line
    current_line = ''
    with open(NCCL_H_IN, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if state == State.new_line:
                if line.startswith('ncclResult_t'):
                    if line.endswith(';'):
                        function_list.append(extract_func_info(line))
                    else:
                        state = State.cont_line
                        current_line = line
            elif state == State.cont_line:
                current_line += line
                if line.endswith(';'):
                    function_list.append(extract_func_info(current_line))
                    state = State.new_line

    # Special case for ncclGetErrorString
    return function_list


func_reference = collect_nccl_function_reference()

func_list = grasp_function_list_from_header_file()

func_list = list(filter(lambda x: x[0] in func_reference, func_list))
func_list.append(('ncclGetErrorString', 'const char* (*)(ncclResult_t)',
                      'const char* (*ncclGetErrorString)(ncclResult_t result)'
                      ))

for tmpl in [DL_NCCL_H_IMPL, DL_NCCL_CPP_IMPL]:
    output = tmpl.replace('.tmpl', '')
    with open(output, "w") as f:
        tmpl = Template(filename=tmpl)
        try:
            f.write(tmpl.render(function_list=func_list))
            print(f"{output} is generated!")
        except Exception as e:
            import sys
            print('-' * 78, file=sys.stderr)
            print('Template exceptions', file=sys.stderr)
            print('-' * 78, file=sys.stderr)
            print(exceptions.text_error_template().render(), file=sys.stderr)
            print('-' * 78, file=sys.stderr)
            raise e


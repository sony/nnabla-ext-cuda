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

from generator_common.init_cpp_common import generate_init_cpp
import os

def generate(info, template):
    includes, registers = generate_init_cpp(
        info, backend='cuda', rank=1)

    #TODO: Delete after dynamic loading support for cudnn and nccl libs.
    # Check if Distributed Training is on or not
    if os.environ.get("NCCL_HOME") is not None:
        include_communicator = \
                "#include <nbla/cuda/communicator/data_parallel_communicator.hpp>"
        register_communicators = '  typedef DataParallelCommunicatorNccl<float> DataParallelCommunicatorNcclf;' \
        '\n  NBLA_REGISTER_COMMUNICATOR_IMPL(DataParallelCommunicator, DataParallelCommunicatorNcclf, 1, "cuda", "default");'

    else:
        include_communicator = ""
        register_communicators = ""

    return template.format(
        include_functions=
        '\n'.join(['#include <nbla/cuda/function/{}>'.format(i) \
                   for i in includes]), 
        register_functions='\n'.join(registers), 
        include_communicator=include_communicator,
        register_communicators=register_communicators
    )


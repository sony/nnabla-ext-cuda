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

import generator_common.common as common


def generate(info, func_name, func_name_snakecase, template):
    io_info = common.function_io(info)
    ctypes = ', '.join(io_info['template_types'])
    templates = ', '.join(io_info['templates'])
    template_defines = ', '.join(['typename {}'.format(t)
                                  for t in io_info['templates']])

    return template.format(func_name=func_name,
                           func_name_snakecase=func_name_snakecase,
                           template_defines=template_defines,
                           templates=templates,
                           ctypes=ctypes)

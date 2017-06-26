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

import collections
import sys
import re
import os


class Implements:

    def __init__(self):

        base = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
        filename = '{}/doc/implements.rst'.format(base)

        error_exists = False
        info = collections.OrderedDict()
        with open(filename, 'r') as f:
            list_num = -1
            list_line = 0

            implements = []
            func_name = None

            for n, l in enumerate(f.readlines()):
                l = l.rstrip()

                m = re.match(r'^   ([\s\*]) -\s*(.*)$', l)
                if m:
                    if m.group(1) == '*':
                        list_num += 1
                        list_line = 0

                        if list_num > 0:
                            func_name = m.group(2)
                            info[func_name] = collections.OrderedDict()
                    else:
                        if list_num == 0:
                            implements.append(m.group(2))
                        else:
                            if m.group(2) != '':
                                info[func_name][implements[list_line - 1]] = m.group(2)

                    list_line += 1

        if error_exists:
            exit(-1)
        self.info = info


# Main command
# ===============
def main():
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    f = Implements()
    print('Info Keys = {}'.format(f.info.keys()))
    for k, v in f.info.items():
        print(k, v)

if __name__ == '__main__':
    main()

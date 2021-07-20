# Copyright 2020,2021 Sony Corporation.
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
import glob
import os
import re
import sys

with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'cudnn_version.bat'), 'w') as w:
    for inc in glob.glob(os.path.join(sys.argv[1], 'include', 'cudnn*.h')):
        with open(inc) as f:
            for l in f.readlines():
                m = re.match('^#define\s+CUDNN_(\S+)\s+(\d+)', l)
                if m and m.group(1) in ['MAJOR', 'MINOR', 'PATCHLEVEL']:
                    print(f'SET CUDNN_{m.group(1)}={m.group(2)}', file=w)

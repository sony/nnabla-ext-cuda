# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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


import requests
from html.parser import HTMLParser
import re
from mako.template import Template
import os
from gpu_info import incompatible_arcs, gpu_compute_capability_to_arc

basedir = os.path.dirname(os.path.abspath(__file__))
r = requests.get('https://developer.nvidia.com/cuda-gpus', stream=True)


class GetGpuListFromNvidiaSite(HTMLParser):
    def __init__(self):
        super().__init__()
        self.td = False
        self.last_value = None
        self.last_data = ''
        self.gpu_data = {}

    def handle_starttag(self, tag, attrs):
        if tag == 'td':
            self.td = True

    def handle_endtag(self, tag):
        if tag == 'td':
            if self.td:
                m = re.match(r'((\d+)\.(\d+))', self.last_data.strip())
                if m:
                    cap = m.group(1)
                    cap_major = int(m.group(2))
                    cap_minor = int(m.group(3))

                    arch = gpu_compute_capability_to_arc.get(cap_major)
                    if arch is None:
                        arch = gpu_compute_capability_to_arc.get(
                            (cap_major, cap_minor))
                        if arch is None:
                            print(f'Error: unknown capability [{cap}]')
                            arch = ''
                    name = self.last_value.lower().replace(
                        'nvidia ', '').replace('tesla ', '')  # remove prefix
                    self.gpu_data[name] = (arch, cap)
                self.last_value = self.last_data.strip()
                self.last_data = ''
                self.td = False

    def handle_data(self, data):
        if self.td:
            self.last_data += data


parser = GetGpuListFromNvidiaSite()
parser.feed(''.join([l.decode() for l in r.raw.readlines()]))
gpus_info = parser.gpu_data

incompatible_gpus = {}
for k in incompatible_arcs:
    if not incompatible_gpus.get(k):
        incompatible_gpus[k] = []
    iarc = incompatible_arcs[k]
    for gpu_name in gpus_info.keys():
        if gpus_info[gpu_name][0] in iarc:
            incompatible_gpus[k].append(gpu_name)

fname = os.path.join(basedir, 'skel', 'incompatibale_gpu_list.py.tmpl')
tmpl = Template(filename=fname)
lines = tmpl.render(args=incompatible_gpus)

with open("./python/src/nnabla_ext/cuda/incompatibale_gpu_list.py", 'w') as f:
    for l in lines:
        f.write(l)

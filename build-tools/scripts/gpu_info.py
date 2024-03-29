# Copyright 2021 Sony Corporation.
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

incompatible_arcs = {
    ('100', '7'): ['Ampere', 'Fermi'],
    ('102', '7'): ['Ampere', 'Fermi'],
    ('102', '8'): ['Ampere', 'Fermi'],
    ('110', '8'): ['Fermi'],
}

gpu_compute_capability_to_arc = {
    8: 'Ampere',
    (7, 5): 'Turing',
    (7, 0): 'Volta',
    (7, 2): 'Volta',
    6: 'Pascal',
    5: 'Maxwell',
    3: 'Kepler',
    2: 'Fermi'
}

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

from libcpp.string cimport string

cdef extern from "nbla/cuda/nvtx.hpp" namespace "nbla":
    void nvtx_mark_A(string) except +
    void nvtx_range_push_A(string) except +
    void nvtx_range_push_with_C(string) except +
    void nvtx_range_pop() except +


def mark_A(str msg):
    nvtx_mark_A(msg)

def range_push(str msg, coloring=True):
    if coloring:
        nvtx_range_push_with_C(msg)
    else:
        nvtx_range_push_A(msg)

def range_pop():
    nvtx_range_pop()

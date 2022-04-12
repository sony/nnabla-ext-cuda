# Copyright 2019,2020,2021 Sony Corporation.
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

from libcpp.string cimport string
from libcpp cimport bool as cpp_bool
from nnabla.function cimport Function
import nnabla.callback as callback


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


cdef class Range:
    """Handle to range operation class for NVTX profiling

    This class provide profiling of CUDA related function with nsys.
    See Nsight Systems page for installation and user's guide.

        https://docs.nvidia.com/nsight-systems/

    To profile using nsys::

        nsys profile -o <output_file> <command>

    Args:
        msg (string): A string representing the scope of this statement.
        coloring (bool, optional, default=True): If this is True,
            automatically assign colors to nvtx range box.
            Default is True.
        logging (bool, optional, default=False): If this is True,
            profile each nnabla.function.
            Please note that this will increase load.
            It is recommended that only 1 logging is enabled
            when using nested with-statements to avoid overload.

    Simple example:

    .. code-block:: python

        import nnabla_ext.cuda.nvtx as nvtx

        with nvtx.Range('training', logging=True):
            for i in range(iteration):
                with nvtx.Range('forward')
                    loss.forward()
                with nvtx.Range('backard'):
                    loss.backward()
                solver.update()
    """
    cdef string msg
    cdef cpp_bool coloring
    cdef cpp_bool logging

    def __cinit__(self, string msg, cpp_bool coloring=True, cpp_bool logging=False):
        self.msg = msg
        self.coloring = coloring
        self.logging = logging

    def _func_pre_hook(self, func):
        name = str(<Function>func)
        range_push(name, self.coloring)

    def _func_post_hook(self, func):
        range_pop()

    def enable(self):
        """
        Starts nnabla.function profiling.
        If logging is already started, the operation will be ignored.
        """
        if not self.logging:
            callback.set_function_pre_hook('nvtx_func_pre_hook', self._func_pre_hook)
            callback.set_function_post_hook('nvtx_func_post_hook', self._func_post_hook)
            self.logging = True

    def disable(self):
        """
        Stop nnabla.function profiling.
        If logging is not started, the operation will be ignored.
        """
        if self.logging:
            callback.unset_function_pre_hook('nvtx_func_pre_hook')
            callback.unset_function_post_hook('nvtx_func_post_hook')
            self.logging = False

    def __enter__(self):
        range_push(self.msg, self.coloring)
        if self.logging:
            callback.set_function_pre_hook('nvtx_func_pre_hook', self._func_pre_hook)
            callback.set_function_post_hook('nvtx_func_post_hook', self._func_post_hook)
        return self

    def __exit__(self, ex_type, ex_value, trace):
        if self.logging:
            callback.unset_function_pre_hook('nvtx_func_pre_hook')
            callback.unset_function_post_hook('nvtx_func_post_hook')
        range_pop()

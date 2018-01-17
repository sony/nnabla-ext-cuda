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

from nnabla.logger import logger
from nnabla import add_available_context

import nnabla._init as cpu_init
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "nbla/cuda/init.hpp" namespace "nbla":
    void init_cuda() except+
    void clear_cuda_memory_cache() except+
    vector[string] cuda_array_classes() except +
    void _cuda_set_array_classes(const vector[string] & a) except +
    void cuda_device_synchronize(const string & device) except +
    int cuda_get_device_count() except +
    vector[string] cuda_get_devices() except +


logger.info('Initializing CUDA extension...')
try:
    init_cuda()
    add_available_context('cuda')
except:
    logger.warning(
        'CUDA initialization failed. Please make sure that the CUDA is correctly installed.')


def clear_memory_cache():
    """Clear memory cache for all devices."""
    clear_cuda_memory_cache()


###############################################################################
# Array pereference API
# TODO: Move these to C++
###############################################################################
_original_array_classes = cuda_array_classes()


def prefer_cached_array(prefer):
    a = cuda_array_classes()
    a = sorted(enumerate(a), key=lambda x: (
        prefer != ('Cached' in x[1]), x[0]))
    _cuda_set_array_classes(map(lambda x: x[1], a))


def reset_array_preference():
    global _original_array_classes
    _cuda_set_array_classes(_original_array_classes)


def array_classes():
    return cuda_array_classes()


# Initialize preference according to CPU cache preference.
tmp = cpu_init._cached_array_prefered()
if tmp is not None:
    prefer_cached_array(tmp)
del tmp

# Register preference API.
cpu_init._add_callback_prefer_cached_array(prefer_cached_array)
cpu_init._add_callback_reset_array_preference(reset_array_preference)
###############################################################################


###############################################################################
# Wrappers for CUDA Runtime API.
###############################################################################
def device_synchronize(str device):
    """Call ``cudaDeviceSynchronize`` in runtime API`.

    Args:
        device (int): Device ID.

    """
    cuda_device_synchronize(device)


def get_device_count():
    """Call ``cudaGetDeviceCount`` in runtime API`.

    Retuns:
        int: Number of devices available.

    """
    return cuda_get_device_count()


def get_devices():
    """Get available devices.

    Returns:
        list of str: List of available devices.

    """
    return cuda_get_devices()
###############################################################################

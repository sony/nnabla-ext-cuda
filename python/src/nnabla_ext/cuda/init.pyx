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
cimport nnabla._init as ccpu_init
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

from libc.stdint cimport uintptr_t
from libcpp cimport bool

cdef extern from "nbla/cuda/init.hpp" namespace "nbla":
    void init_cuda() except+
    void clear_cuda_memory_cache() except+
    void clear_cuda_virtual_memory_cache() except+
    void print_cuda_memory_cache_map() except+
    void print_cuda_virtual_memory_cache_map() except+
    size_t get_cuda_caching_allocator_fragmentation_bytes(const string& device_id) except +
    size_t get_cuda_caching_allocator_max_available_bytes(const string& device_id) except +
    vector[int] get_cuda_cached_memory_used_counts(const string& device_id) except +
    size_t get_cuda_virtual_caching_allocator_fragmentation_bytes(const string& device_id) except +
    size_t get_cuda_virtual_caching_allocator_max_available_bytes(const string& device_id) except +
    vector[int] get_cuda_virtual_memory_used_counts(const string& device_id) except +
    vector[string] cuda_array_classes() except +
    void _cuda_set_array_classes(const vector[string] & a) except +
    void cuda_device_synchronize(const string & device) except +
    int cuda_get_device_count() except +
    vector[string] cuda_get_devices() except +
    shared_ptr[void] cuda_create_stream(int device_id) except +
    void* cuda_stream_shared_to_void(shared_ptr[void]) except +
    void print_stream_flag(shared_ptr[void]) except +
    void print_stream_priority(shared_ptr[void]) except +
    void cuda_stream_synchronize(shared_ptr[void]) nogil except +
    void cuda_nullstream_synchronize() nogil except +
    void cuda_stream_destroy(shared_ptr[void]) except +
    shared_ptr[void] cuda_create_event(int device_id) except +
    shared_ptr[void] cuda_create_event(int device_id, unsigned int) except +
    void cuda_default_stream_event(shared_ptr[void]) except +
    void cuda_stream_wait_event(shared_ptr[void], shared_ptr[void]) except +
    void cuda_event_synchronize(shared_ptr[void]) nogil except +
    float cuda_event_elapsed_time(shared_ptr[void], shared_ptr[void]) except +
    void cuda_event_record(shared_ptr[void]) except +
    void set_cuda_vma_chunk_size(size_t size) except +

cdef extern from "nbla/cuda/common.hpp" namespace "nbla":
    vector[size_t] cuda_mem_get_info() except +

logger.info('Initializing CUDA extension...')
try:
    init_cuda()
    add_available_context('cuda')
except:
    logger.warning(
        'CUDA initialization failed. Please make sure that the CUDA is correctly installed.')


def clear_memory_cache():
    """Clear memory cache in caching allocator for all devices."""
    clear_cuda_memory_cache()

def clear_virtual_memory_cache():
    """Clear memory cache in virtual caching allocator for all devices."""
    clear_cuda_virtual_memory_cache()

def print_memory_cache_map():
    """Dump cuda memory cache map."""
    print_cuda_memory_cache_map()

def print_virtual_memory_cache_map():
    """Dump cuda virtual memory cache map."""
    print_cuda_virtual_memory_cache_map()

def get_caching_allocator_fragmentation_bytes(str device_id):
    """Get fragmentation size of caching allocator for cudaCachedMemory."""
    return get_cuda_caching_allocator_fragmentation_bytes(device_id)

def get_caching_allocator_max_available_bytes(str device_id):
    """Get maximum available bytes in caching allocator for cudaCachedMemory."""
    return get_cuda_caching_allocator_max_available_bytes(device_id)

def get_cached_memory_used_counts(str device_id):
    """Get # of cuda cached memory which is currently used."""
    return get_cuda_cached_memory_used_counts(device_id)

def get_virtual_caching_allocator_fragmentation_bytes(str device_id):
    """Get fragmentation size of virtual caching allocator for cudaVirtualMemoy."""
    return get_cuda_virtual_caching_allocator_fragmentation_bytes(device_id)

def get_virtual_caching_allocator_max_available_bytes(str device_id):
    """Get maximum available bytes in virtual caching allocator for cudaVirtualMemory."""
    return get_cuda_virtual_caching_allocator_max_available_bytes(device_id)

def get_virtual_memory_used_counts(str device_id):
    """Get # of cuda virtual memory which is currently used."""
    return get_cuda_virtual_memory_used_counts(device_id)

###############################################################################
# Array preference API
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


def prefer_cpu_pinned_array():
    a = ccpu_init._cpu_array_classes()
    a = sorted(a, key=lambda x: (x != 'CudaCachedHostArray'))
    ccpu_init._cpu_set_array_classes(a)


def prefer_cuda_cached_array():
    a = cuda_array_classes()
    a = sorted(a, key=lambda x: (x != 'CudaCachedArray'))
    _cuda_set_array_classes(a)


def prefer_unified_array():
    a = cuda_array_classes()
    a = sorted(a, key=lambda x: (x != 'CudaCachedUnifiedArray'))
    _cuda_set_array_classes(a)


def prefer_cuda_virtual_array():
    a = cuda_array_classes()
    a = sorted(a, key=lambda x: (x != 'CudaCachedVirtualArray'))
    _cuda_set_array_classes(a)

# Initialize preference according to CPU cache preference.
tmp = cpu_init._cached_array_preferred()
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

    Returns:
        int: Number of devices available.

    """
    return cuda_get_device_count()


def get_devices():
    """Get available devices.

    Returns:
        list of str: List of available devices.

    """
    return cuda_get_devices()


def get_device_memory_size():
    """Get free and total device memory size.

    Returns:
        list of free and total device memory size.

    """
    return cuda_mem_get_info()
###############################################################################


###############################################################################
# Wrapper class for CudaEvent.
###############################################################################

cdef class Event:
    cdef shared_ptr[void] event

    def __init__(self, device_id):
        # get cudaEvent with cudaEventDefault flag.
        self.event = cuda_create_event(device_id, 0)

    def record(self):
        cuda_event_record(self.event)

    def synchronize(self):
        with nogil:
            cuda_event_synchronize(self.event)

    def elapsed(self, start_event):
        cdef float ret
        ret = cuda_event_elapsed_time((<Event?>start_event).event, self.event)

        return ret

###############################################################################
# Wrapper class for CudaStream & CudaEvent.
###############################################################################

cdef class StreamEventHandler:
    cdef shared_ptr[void] stream
    cdef shared_ptr[void] event
    cdef public object value
    cdef public int device_id
    cpdef bool is_stream_destroy

    def __cinit__(self, int device_id=-1):
        self.is_stream_destroy = True
        self.device_id = device_id

    def __init__(self, int device_id=-1):
        self.stream_create(device_id)
        self.event = cuda_create_event(device_id)
        self.add_default_stream_event()

    def stream_wait_event(self):
        if not self.is_stream_destroy:
            cuda_stream_wait_event(self.stream, self.event)

    def add_default_stream_event(self):
        cuda_default_stream_event(self.event)

    def event_synchronize(self):
        with nogil:
            cuda_event_synchronize(self.event)

    def stream_destroy(self):
        cuda_stream_destroy(self.stream)
        self.is_stream_destroy = True

    def stream_create(self, device_id):
        if not self.is_stream_destroy:
            self.stream_destroy()

        self.stream = cuda_create_stream(device_id)

        cdef void* stream_vp = cuda_stream_shared_to_void(self.stream)
        self.value = <uintptr_t>stream_vp

        self.is_stream_destroy = False

    def stream_synchronize(self):
        if not self.is_stream_destroy:
            with nogil:
                cuda_stream_synchronize(self.stream)

    def default_stream_synchronize(self):
        with nogil:
            cuda_nullstream_synchronize()

###############################################################################
# CudaVirtualMemoryAllocator
###############################################################################

def set_cuda_virtual_memory_chunk_size(size):
    set_cuda_vma_chunk_size(size)

# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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

from __future__ import absolute_import
from nnabla.variable import Context

import nnabla

from ._version import (
    __version__,
    __cuda_version__,
    __cudnn_version__,
    __author__,
    __email__
)
from .incompatible_gpu_list import incompatible_gpus
import os
import sys

#
# Workaround for loading shared library.
#

# From python3.8, PATH and the current working directory are no longer used to load DLL
# User must use os.add_dll_directory() to add DLLs directory
if sys.platform == 'win32':
    path_env = os.environ.get('PATH').lower()
    # Under Windows, CUDA libraries in alllib wheel need to be added to PATH
    if 'cuda' not in path_env or 'cudnn' not in path_env:
        alllib_cuda_path = os.path.dirname(os.path.realpath(__file__))
        os.environ['PATH'] += os.pathsep + alllib_cuda_path
    if sys.version_info.minor >= 8:
        path_env = os.environ.get('PATH')
        if path_env is not None:
            path_env = path_env.lower().split(';')
            for path in path_env:
                if 'cuda' in path or 'cudnn' in path:
                    os.add_dll_directory(path)

MAX_RETRY_LOAD_SHARED_LIB = 100


def load_shared_from_error(err):
    import ctypes
    base = os.path.dirname(__file__)
    es = str(err).split(':')
    if len(es) > 0:
        fn = os.path.join(base, es[0])
        if os.path.exists(fn):
            retry = 0
            while retry < MAX_RETRY_LOAD_SHARED_LIB:
                retry += 1
                try:
                    ctypes.cdll.LoadLibrary(fn)
                    retry = MAX_RETRY_LOAD_SHARED_LIB
                except OSError as err:
                    load_shared_from_error(err)
        else:
            raise err


def check_gpu_compatibility():
    import os
    from nnabla.utils import nvml

    def list_local_gpu():
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        local_gpus = []
        for device_index in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            gpu_name = nvml.nvmlDeviceGetName(
                handle).decode('utf-8').lower()
            local_gpus.append(gpu_name)
        nvml.nvmlShutdown()
        return local_gpus

    def compare_gpu(local_gpus, incompatible_gpu, cuda_ver, cudnn_ver):
        unusable_gpu = []
        available_gpu_names = os.environ.get('AVAILABLE_GPU_NAMES')
        if available_gpu_names is not None:  # GPU in white list is usable
            available_gpu_names = available_gpu_names.lower().split(',')
            local_gpus = list(set(local_gpus) - set(available_gpu_names))
        for gpu in local_gpus:
            for inc_gpu in incompatible_gpu.get((cuda_ver, cudnn_ver), []):
                if inc_gpu in gpu:  # mark gpu incompatible if it in incompatible gpu list
                    if gpu not in unusable_gpu:
                        unusable_gpu.append(gpu)
                    break
        if len(unusable_gpu) > 0:
            raise ValueError("Currently, nnabla-ext-cuda" + cuda_ver + " does not support your " + ",".join(unusable_gpu) + " GPU." +
                             " It may take a long time to initialize cudnn and can't converge well!\n" +
                             "You can set environment variable AVAILABLE_GPU_NAMES=\"" +
                             ",".join(unusable_gpu) + "\" to avoid this error.")
    cuda_ver = __cuda_version__.replace('.', '')
    cudnn_ver = __cudnn_version__[0]

    try:
        local_gpus = list_local_gpu()
    except:
        print(
            "GPU compatibility could not be verified due to a problem getting the GPU list.")
        return
    compare_gpu(list_local_gpu(), incompatible_gpus, cuda_ver, cudnn_ver)


check_gpu_compatibility()


retry = 0
while retry < MAX_RETRY_LOAD_SHARED_LIB:
    retry += 1
    try:
        from .init import (
            clear_memory_cache,
            array_classes,
            device_synchronize,
            get_device_count,
            get_devices,
            StreamEventHandler)
        retry = MAX_RETRY_LOAD_SHARED_LIB
    except ImportError as err:
        load_shared_from_error(err)


def context(device_id=0, type_config='float', **kw):
    """CUDA context."""
    backends = ['cuda:float', 'cpu:float']
    if type_config == 'half':
        backends = ['cuda:half', 'cuda:float', 'cpu:float']
    elif type_config == 'mixed_half':
        backends = ['cuda:mixed_half', 'cuda:float', 'cpu:float']
    elif type_config == 'float':
        pass
    else:
        raise ValueError("Unknown data type config is given %s" % type_config)
    return Context(backends, array_classes()[0], device_id=str(device_id))


def synchronize(device_id=0, **kw):
    """Call ``cudaDeviceSynchronize`` in runtime API`.

    Args:
        device_id (str): Device ID. e.g. "0", "1".

    """
    return device_synchronize(device_id)

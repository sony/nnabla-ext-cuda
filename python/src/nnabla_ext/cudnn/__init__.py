# Copyright 2018,2019,2020,2021 Sony Corporation.
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

import nnabla
import nnabla_ext.cuda
from . import init

from nnabla.variable import Context
from ._version import (
    __version__,
    __author__,
    __email__
)

from nnabla_ext.cuda import (
    clear_memory_cache,
    array_classes,
    device_synchronize,
    get_device_count,
    get_devices,
    synchronize,
)

from nnabla_ext.cudnn.init import (
    set_conv_fwd_algo_blacklist,
    set_conv_bwd_data_algo_blacklist,
    set_conv_bwd_filter_algo_blacklist,
    unset_conv_fwd_algo_blacklist,
    unset_conv_bwd_data_algo_blacklist,
    unset_conv_bwd_filter_algo_blacklist
)


def context(device_id='0', type_config='float', *kw):
    """CUDNN context"""
    check_gpu(device_id)
    from nnabla_ext.cuda import array_classes
    backends = ['cudnn:float', 'cuda:float', 'cpu:float']
    if type_config == 'half':
        backends = ['cudnn:half', 'cudnn:float',
                    'cuda:half', 'cuda:float', 'cpu:float']
    elif type_config == 'mixed_half':
        backends = ['cudnn:mixed_half', 'cudnn:half', 'cudnn:float',
                    'cuda:mixed_half', 'cuda:half', 'cuda:float', 'cpu:float']
    elif type_config == 'float':
        pass
    else:
        raise ValueError("Unknown data type config is given %s" % type_config)
    return Context(backends, array_classes()[0], device_id=str(device_id))


def check_gpu(device_id):
    """Check whether the specific GPU is usable.

    Args:
        device_id (str) : GPU device ID in local machine.

    Returns:
        None if this GPU device is usable, otherwise, error will be reported.

    Example:

        .. code-block:: python

            import nnabla_ext.cudnn
            device_id = '0'
            nnabla_ext.cudnn.check_gpu(device_id)

    """
    import os
    from nnabla.utils import nvml
    incompatible_gpus = nnabla_ext.cuda.incompatible_gpus
    cuda_ver = nnabla_ext.cuda.__cuda_version__.replace('.', '')
    cudnn_ver = nnabla_ext.cuda.__cudnn_version__[0]
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(int(device_id))
    gpu_name = nvml.nvmlDeviceGetName(handle).decode('utf-8').lower()
    nvml.nvmlShutdown()
    available_gpu_names = os.environ.get('AVAILABLE_GPU_NAMES')
    if available_gpu_names is not None:
        available_gpu_names = available_gpu_names.lower().split(',')
        if gpu_name in available_gpu_names:
            return
    for inc_gpu in incompatible_gpus.get((cuda_ver, cudnn_ver), []):
        if inc_gpu in gpu_name:  # raise error if GPU is in incompatible gpu list and not in white list
            raise ValueError(
                "Currently, nnabla-ext-cuda{} does not support your {} GPU.".format(cuda_ver, gpu_name))

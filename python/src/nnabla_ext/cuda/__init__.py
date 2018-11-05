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

from __future__ import absolute_import

import nnabla

from ._version import (
    __version__,
    __cuda_version__,
    __cudnn_version__,
    __author__,
    __email__
)

try:
    from .init import (
        clear_memory_cache,
        array_classes,
        device_synchronize,
        get_device_count,
        get_devices)
except:
    print('Please install CUDA version {}.'.format(__cuda_version__))
    print('          and CUDNN version {}.'.format(__cudnn_version__))
    print('Or install correct nnabla_ext_cuda for installed version of CUDA/CUDNN.')
    raise

from nnabla.variable import Context


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

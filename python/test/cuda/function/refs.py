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

import numpy as np


def scan(func, init_val, x, axis, exclusive, reverse):
    if reverse:
        x = np.flip(x, axis)
    y = func(x, axis)
    if exclusive:
        # Make first element to `init_val` on the axis.
        shape = y.shape
        if axis < 0:
            ndim = len(shape)
            axis += ndim
        shape_3d = ()
        shape_3d += (np.prod(shape[:axis], dtype=np.int64),)
        shape_3d += (shape[axis],)
        shape_3d += (np.prod(shape[(axis+1):], dtype=np.int64),)
        y = np.reshape(y, shape_3d)
        scan_size = shape[axis]
        y[:, 1:scan_size, :] = y[:, :(scan_size-1), :]
        y[:, 0, :] = init_val
        y = np.reshape(y, shape)
    if reverse:
        y = np.flip(y, axis)
    return y


def cumsum(x, axis, exclusive, reverse):
    return scan(np.cumsum, 0, x, axis, exclusive, reverse)


def cumprod(x, axis, exclusive, reverse):
    return scan(np.cumprod, 1, x, axis, exclusive, reverse)

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

import pytest
import os
import numpy as np
import tempfile

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla_ext.cuda.init as cuda_init
import nnabla._init as cpu_init
import nnabla_ext.cuda.lms as lms
import nnabla.ext_utils as ext_utils

from model import WaveNet
from config import data_config, WavenetConfig

# Parameters
batch_size = 2
num_dilations = 5
learning_rate = 1e-3

# Set GPU memory size for test
# The values below of GPU memory size were based on GeForce 1050 Ti with 4GB memory.
mem_size = 1e9
total_mem_size = cuda_init.get_device_memory_size()[1];
skip_test = (total_mem_size < 3e9)
print(total_mem_size)


def create_network(batch_size, num_dilations, learning_rate):
    # model
    x = nn.Variable(shape=(batch_size, data_config.duration, 1))  # (B, T, 1)
    onehot = F.one_hot(x, shape=(data_config.q_bit_len, ))  # (B, T, C)
    wavenet_input = F.transpose(onehot, (0, 2, 1))  # (B, C, T)

    # speaker embedding
    s_emb = None

    net = WaveNet(num_dilations)
    wavenet_output = net(wavenet_input, s_emb)

    pred = F.transpose(wavenet_output, (0, 2, 1))

    # (B, T, 1)
    t = nn.Variable(shape=(batch_size, data_config.duration, 1))

    loss = F.mean(F.softmax_cross_entropy(pred, t))
    #loss.visit(PrintFunc())

    # Create Solver.
    solver = S.Adam(learning_rate)
    solver.set_parameters(nn.get_parameters())

    return x, t, loss, solver


class GetVariableSizeFunc(object):
    def __init__(self):
        # Large value delt with float
        self.max_size = 0.0

    def __call__(self, nnabla_func):
        total_size = 0.0

        for v in nnabla_func.inputs:
            total_size += v.size()

        for v in nnabla_func.outputs:
            total_size += v.size()

        if self.max_size < total_size:
            self.max_size = total_size

    def get_max_size_as_float(self):
        return self.max_size * 4.0 # computed as 4 byte float


@pytest.mark.parametrize("type_config, device_id, batch_size, " \
                         "num_dilations, learning_rate, max_iter, gpu_memory_size", 
                         [('float', '0', batch_size, num_dilations, learning_rate, 2, mem_size),
                          ('half', '0', batch_size, num_dilations, learning_rate, 2, mem_size)])


@pytest.mark.skipif(skip_test, reason='Out of GPU memory by the variables used in a single function.')
def test_lms(type_config, device_id, batch_size, num_dilations, 
             learning_rate, max_iter, gpu_memory_size):
    # Use pinned host memory
    cuda_init.prefer_cpu_pinned_array()
 
    # Set context.
    from nnabla.ext_utils import get_extension_context
    cpu_ctx = get_extension_context('cpu',
                                    device_id='',
                                    type_config=type_config)
    gpu_ctx = get_extension_context('cudnn',
                                    device_id=device_id,
                                    type_config=type_config)
    nn.set_default_context(gpu_ctx)

    # Input data
    x0 = []
    t0 = []

    with tempfile.NamedTemporaryFile(mode='w', suffix='.h5', delete=False) as tf:
        # Init inputs
        np.random.seed(seed=32)
        for i in range(max_iter):
            x0 += [np.random.randint(0, 256, size=(batch_size, data_config.duration, 1))]
            t0 += [np.random.randint(0, 256, size=(batch_size, data_config.duration, 1))]
        
        initial_param1 = []
        initial_param2 = []

        ###############################
        # Normal
        ###############################
        with nn.parameter_scope('network1'):
            # Create network
            x, t, loss, solver = create_network(batch_size, num_dilations, learning_rate)

            # Store the initial parameter
            nn.save_parameters(tf.name)

            # Load the initial parameter
            nn.load_parameters(tf.name)
            for k, p in nn.get_parameters(grad_only=False).items():
                initial_param1.append(p.d)

            # Training loop.
            for i in range(max_iter):
                x.d = x0[i]
                t.d = t0[i]

                solver.zero_grad()
                loss.forward(clear_no_need_grad=True)
                loss.backward(clear_buffer=True)
                solver.update()

            # Synchronization
            ext = ext_utils.import_extension_module('cudnn')
            ext.device_synchronize(device_id)

        ###############################
        # Swap in/out scheduler
        ###############################
        with nn.parameter_scope('network2'):
            # Create network
            x, t, loss, solver = create_network(batch_size, num_dilations, learning_rate)

            # Load the initial parameter
            nn.load_parameters(tf.name)
            for k, p in nn.get_parameters(grad_only=False).items():
                initial_param2.append(p.d)

            # Create a scheduler
            scheduler = lms.SwapInOutScheduler(cpu_ctx, gpu_memory_size)

            # Training loop.
            for i in range(max_iter):
                x.d = x0[i]
                t.d = t0[i]

                solver.zero_grad()
                
                # Init the scheduler
                scheduler.start_scheduling()

                loss.forward(clear_no_need_grad=True, 
                             function_pre_hook=scheduler.function_pre_hook,
                             function_post_hook=scheduler.function_post_hook)
                loss.backward(clear_buffer=True,
                              function_pre_hook=scheduler.function_pre_hook,
                              function_post_hook=scheduler.function_post_hook)
                solver.update(update_pre_hook=scheduler.update_pre_hook,
                              update_post_hook=scheduler.update_post_hook)
                
                # Finalize the scheduler
                scheduler.end_scheduling()

            # Synchronization
            ext = ext_utils.import_extension_module('cudnn')
            ext.device_synchronize(device_id)

        ###############################
        # Test
        ###############################
        for p1, p2 in zip(initial_param1, initial_param2):
            # Check the identity of initial parameters
            assert np.array_equal(p1, p2)

        with nn.parameter_scope('network1'):
            param1 = nn.get_parameters(grad_only=False)

        with nn.parameter_scope('network2'):
            param2 = nn.get_parameters(grad_only=False)
        
        for (k1, p1), (k2, p2) in zip(param1.items(), param2.items()):
            assert np.allclose(p1.d, p2.d, atol=1e-3)

        # Remove the file
        tf.close()
        os.remove(tf.name)
        assert not os.path.exists(tf.name)

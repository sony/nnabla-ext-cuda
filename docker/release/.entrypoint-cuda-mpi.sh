#!/bin/bash
# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

set -e

export BASH_ENV=/etc/bash.bashrc
export ENV=/etc/shinit_v2

# Solve nccl error that No space left on device
# while creating shared memory segment.
echo NCCL_SHM_DISABLE=1 >> /etc/nccl.conf
echo NCCL_P2P_LEVEL=SYS >> /etc/nccl.conf

/bin/bash

$@

#!/bin/sh
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

test -f /etc/shinit_v2 && . /etc/shinit_v2

if [ -d "/opt/mpi/hpcx" ]; then
    curdir=$PWD
    cd /opt/mpi/hpcx
    . ./hpcx-init.sh
    hpcx_load
    cd $curdir
    unset curdir
else
    export PATH=/opt/mpi/bin:$PATH
    export LD_LIBRARY_PATH=/opt/mpi/lib:$LD_LIBRARY_PATH
fi

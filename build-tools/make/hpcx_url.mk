# Copyright 2023 Sony Corporation.
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

########################################################################################################################
# Suppress most of make message.
.SILENT:

# Map specific openmpi version to HPCX download url

export HPCX_URL_ubuntu_4.1.5='https://content.mellanox.com/hpc/hpc-x/v2.12/hpcx-v2.12-gcc-MLNX_OFED_LINUX-5-ubuntu20.04-cuda11-gdrcopy2-nccl2.12-x86_64.tbz'
export HPCX_URL_centos_4.1.5='https://content.mellanox.com/hpc/hpc-x/v2.12/hpcx-v2.12-gcc-MLNX_OFED_LINUX-5-redhat7-cuda11-gdrcopy2-nccl2.12-x86_64.tbz'

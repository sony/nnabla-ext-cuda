# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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

find_package(CUDA REQUIRED)
find_package(cuDNN REQUIRED)
foreach(cudnn_include_dir ${CUDNN_INCLUDE_DIRS})
  set(cudnn_version_file OFF)
  if (EXISTS ${cudnn_include_dir}/cudnn_version.h)
    set(cudnn_version_file "${cudnn_include_dir}/cudnn_version.h")
  elseif(EXISTS ${cudnn_include_dir}/cudnn.h)
    set(cudnn_version_file "${cudnn_include_dir}/cudnn.h")
  endif()

  if (cudnn_version_file)
    file(STRINGS ${cudnn_version_file} cudnn_defines)
    string(REGEX REPLACE [[^.*CUDNN_MAJOR +([0-9]+).*$]] [[\1]] cudnn_major ${cudnn_defines})
    string(REGEX REPLACE [[^.*CUDNN_MINOR +([0-9]+).*$]] [[\1]] cudnn_minor ${cudnn_defines})
    string(REGEX REPLACE [[^.*CUDNN_PATCHLEVEL +([0-9]+).*$]] [[\1]] cudnn_patchlevel ${cudnn_defines})
  endif()
endforeach()
set(CUDNN_VERSION "${cudnn_major}.${cudnn_minor}.${cudnn_patchlevel}")


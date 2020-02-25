:: Copyright (c) 2017 Sony Corporation. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
:: 

IF NOT EXIST %nnabla_build_wheel_folder% (
   ECHO nnabla_build_wheel_folder ^(%nnabla_build_wheel_folder%^) does not exist.
   exit /b 255
)

IF NOT DEFINED nnabla_ext_cuda_build_wheel_folder_name (
  SET nnabla_ext_cuda_build_wheel_folder_name=build_wheel
)

IF NOT DEFINED nnabla_ext_cuda_build_wheel_folder (
  SET nnabla_ext_cuda_build_wheel_folder=%nnabla_ext_cuda_root%\%nnabla_ext_cuda_build_wheel_folder_name%%nnabla_ext_cuda_build_wheel_folder_suffix%
)

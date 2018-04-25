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

:: NNabla folders

IF NOT DEFINED nnabla_root (
   SET nnabla_root=%~dp0..\..\..\..\nnabla
)

IF NOT EXIST %nnabla_root% (
   ECHO nnabla_root ^(%nnabla_root%^) does not exist.
   exit /b 255
)   

PUSHD .
CD %nnabla_root%
SET nnabla_root=%CD%
POPD

CALL %nnabla_root%\build-tools\msvc\tools\default_folders.bat || GOTO :error

IF NOT EXIST %nnabla_build_folder% (
   ECHO nnabla_build_folder ^(%nnabla_build_folder%^) does not exist.
   exit /b 255
)   

:: Ext CUDA folders
SET nnabla_ext_cuda_root=%~dp0..\..\..
PUSHD .
CD %nnabla_ext_cuda_root%
SET nnabla_ext_cuda_root=%CD%
POPD

IF NOT DEFINED nnabla_ext_cuda_build_folder (
  SET nnabla_ext_cuda_build_folder=%nnabla_ext_cuda_root%\build%nnabla_ext_cuda_build_folder_suffix%
)


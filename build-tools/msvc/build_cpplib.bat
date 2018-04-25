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

@ECHO OFF
SETLOCAL

:: Folders
CALL %~dp0tools\default_folders.bat || GOTO :error

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

:: Build CUDA extension library
IF NOT EXIST %nnabla_ext_cuda_build_folder% MKDIR %nnabla_ext_cuda_build_folder%
CD %nnabla_ext_cuda_build_folder%

cmake -G "%generate_target%" ^
      -DPYTHON_COMMAND_NAME=python ^
      -DCUDA_SELECT_NVCC_ARCH_ARG:STRING="Common" ^
      -DBUILD_CPP_LIB=ON ^
      -DBUILD_PYTHON_PACKAGE=OFF ^
      -DNNABLA_DIR=%nnabla_root% ^
      -DNBLA_CUDNN_VERSION=%CUDNNVER% ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla.lib ^
      %nnabla_ext_cuda_root% || GOTO :error

msbuild ALL_BUILD.vcxproj /p:Configuration=%build_type% /verbosity:minimal /maxcpucount || GOTO :error

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%



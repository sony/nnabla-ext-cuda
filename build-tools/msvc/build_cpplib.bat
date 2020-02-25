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


SET third_party_folder=%nnabla_root%\third_party
:: Build third party libraries.
CALL %nnabla_root%\build-tools\msvc\tools\build_zlib.bat       || GOTO :error


:: Build CUDA extension library
IF NOT EXIST %nnabla_ext_cuda_build_folder% MKDIR %nnabla_ext_cuda_build_folder%
CD %nnabla_ext_cuda_build_folder%

cmake -G "%generate_target%" ^
      -DBUILD_CPP_LIB=ON ^
      -DBUILD_CPP_UTILS=ON ^
      -DBUILD_PYTHON_PACKAGE=OFF ^
      -DEXT_CUDA_LIB_NAME_SUFFIX=%ext_cuda_lib_name_suffix% ^
      -DLIB_NAME_SUFFIX=%lib_name_suffix% ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla%lib_name_suffix%.lib ^
      -DCUDA_SELECT_NVCC_ARCH_ARG:STRING="Common" ^
      -DNNABLA_DIR=%nnabla_root% ^
      -DPYTHON_COMMAND_NAME=python ^
      -DZLIB_INCLUDE_DIR=%zlib_include_dir% ^
      -DZLIB_LIBRARY_RELEASE=%zlib_library% ^
      %nnabla_ext_cuda_root% || GOTO :error

msbuild ALL_BUILD.vcxproj /p:Configuration=%build_type% /verbosity:minimal /maxcpucount || GOTO :error
cpack -G ZIP -C %build_type%

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%



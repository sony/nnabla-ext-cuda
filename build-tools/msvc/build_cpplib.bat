@ECHO OFF

REM Copyright 2018,2019,2020,2021 Sony Corporation.
REM Copyright 2021,2022 Sony Group Corporation.
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM 

SETLOCAL

REM Environment
CALL %~dp0tools\env.bat 3.6 %1 %2 %3 || GOTO :error

IF NOT EXIST %nnabla_build_folder% (
   ECHO nnabla_build_folder ^(%nnabla_build_folder%^) does not exist.
   exit /b 255
)   

SET third_party_folder=%nnabla_root%\third_party
REM Build third party libraries.
CALL %nnabla_root%\build-tools\msvc\tools\build_zlib.bat       || GOTO :error


REM Build CUDA extension library
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

msbuild ALL_BUILD.vcxproj /p:Configuration=%build_type% /verbosity:minimal /maxcpucount
cpack -G ZIP -C %build_type%

GOTO :end
:error
ECHO failed with error code %errorlevel%.
exit /b %errorlevel%

:end
ENDLOCAL

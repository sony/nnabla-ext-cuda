@ECHO OFF

REM Copyright 2018,2019,2020,2021 Sony Corporation.
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
CALL %~dp0tools\env.bat %1 %2 %3 || GOTO :error

IF NOT EXIST %nnabla_build_folder% (
   ECHO nnabla_build_folder ^(%nnabla_build_folder%^) does not exist.
   exit /b 255
)

REM Build CUDA extension wheel
IF NOT EXIST %nnabla_ext_cuda_build_wheel_folder% MKDIR %nnabla_ext_cuda_build_wheel_folder%
CD %nnabla_ext_cuda_build_wheel_folder%

IF NOT EXIST %nnabla_build_wheel_folder% (
   ECHO nnabla_build_wheel_folder ^(%nnabla_build_wheel_folder%^) does not exist.
   exit /b 255
)

for /f %%i in ('dir /b /s %nnabla_build_wheel_folder%\dist\nnabla-*.whl') do set WHL=%%~fi
IF NOT DEFINED WHL (
   ECHO NNabla wheel is not found.
   exit /b 255
)

pip install %PIP_INS_OPTS% %WHL% || GOTO :error

chcp 932
cmake -G "%generate_target%" ^
      -DBUILD_CPP_LIB=OFF ^
      -DBUILD_PYTHON_PACKAGE=ON ^
      -DCPPLIB_BUILD_DIR=%nnabla_build_folder% ^
      -DCPPLIB_CUDA_LIBRARY=%nnabla_ext_cuda_build_folder%\bin\%build_type%\nnabla_cuda%ext_cuda_lib_name_suffix%.dll ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla%lib_name_suffix%.dll ^
      -DEXT_CUDA_LIB_NAME_SUFFIX=%ext_cuda_lib_name_suffix% ^
      -DLIB_NAME_SUFFIX=%lib_name_suffix% ^
      -DNNABLA_DIR=%nnabla_root% ^
      -DPYTHON_COMMAND_NAME=python ^
      -DPYTHON_PKG_DIR=%VENV_PYTHON_PKG_DIR% ^
      %nnabla_ext_cuda_root% || GOTO :error

msbuild wheel.vcxproj /p:Configuration=%build_type% /verbosity:minimal || GOTO :error

CALL deactivate.bat || GOTO :error

GOTO :end
:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%

:end
ENDLOCAL



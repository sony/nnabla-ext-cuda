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
CALL %~dp0tools\default_folders_wheel.bat || GOTO :error

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

IF NOT EXIST %nnabla_ext_cuda_build_folder% MKDIR %nnabla_ext_cuda_build_folder%
IF NOT EXIST %nnabla_ext_cuda_build_wheel_folder% MKDIR %nnabla_ext_cuda_build_wheel_folder%

:: Build CUDA extension wheel
CD %nnabla_ext_cuda_build_wheel_folder%

for /f %%i in ('dir /b /s %nnabla_build_wheel_folder%\dist\*.whl') do set WHL=%%~fi
IF NOT DEFINED WHL (
   ECHO NNabla wheel is not found.
   exit /b 255
)

IF EXIST env RD /s /q env
virtualenv --system-site-packages env || GOTO :error
CALL env\scripts\activate.bat || GOTO :error

pip install --no-deps %WHL% || GOTO :error

cmake -G "%generate_target%" ^
      -DPYTHON_COMMAND_NAME=python ^
      -DBUILD_CPP_LIB=OFF ^
      -DBUILD_PYTHON_PACKAGE=ON ^
      -DNNABLA_DIR=%nnabla_root% ^
      -DCPPLIB_BUILD_DIR=%nnabla_build_folder% ^
      -DCPPLIB_LIBRARY=%nnabla_build_folder%\bin\%build_type%\nnabla.dll ^
      -DCPPLIB_CUDA_LIBRARY=%nnabla_ext_cuda_build_folder%\bin\%build_type%\nnabla_cuda.dll ^
      %nnabla_ext_cuda_root% || GOTO :error

msbuild wheel.vcxproj /p:Configuration=%build_type% /verbosity:minimal || GOTO :error

CALL deactivate.bat || GOTO :error

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%


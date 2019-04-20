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

:: Command line args

:: Folders
SET nnabla_ext_cuda_root=%~dp0..\..
CALL %~dp0tools\default_folders.bat || GOTO :error
CALL %~dp0tools\default_folders_wheel.bat || GOTO :error

:: Settings
CALL %~dp0tools\default_settings.bat || GOTO :error

:: Install wheels
CD %nnabla_ext_cuda_build_wheel_folder%

for /f %%i in ('dir /b /s %nnabla_build_wheel_folder%\dist\*.whl') do set WHL=%%~fi
IF NOT DEFINED WHL (
   ECHO NNabla wheel is not found.
   exit /b 255
)

for /f %%i in ('dir /b /s %nnabla_ext_cuda_build_wheel_folder%\dist\*.whl') do set WHLCUDA=%%~fi
IF NOT DEFINED WHLCUDA (
   ECHO NNabla CUDA Extension wheel is not found.
   exit /b 255
)

IF EXIST env RD /s /q env
virtualenv --system-site-packages env || GOTO :error
CALL env\scripts\activate.bat || GOTO :error

pip install --no-deps %WHL% || GOTO :error
pip install --no-deps %WHLCUDA% || GOTO :error

SET PYTHONPATH=%nnabla_ext_cuda_root%\python\test;%PYTHONPATH%
python -m pytest %nnabla_root%\python\test || GOTO :error

CALL deactivate.bat || GOTO :error

ENDLOCAL
exit /b

:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%


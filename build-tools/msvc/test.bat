REM Copyright 2018,2019,2020,2021 Sony Corporation.
REM Copyright 2021 Sony Group Corporation.
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


@ECHO OFF
SETLOCAL

REM Environment
CALL %~dp0tools\env.bat %1 %2 %3 || GOTO :error

for /f %%i in ('dir /b /s %nnabla_build_wheel_folder%\dist\nnabla-*.whl') do set WHL=%%~fi
IF NOT DEFINED WHL (
   ECHO NNabla wheel is not found.
   exit /b 255
)
for /f %%i in ('dir /b /s %nnabla_ext_cuda_build_wheel_folder%\dist\*.whl') do set WHLCUDA=%%~fi
IF NOT DEFINED WHLCUDA (
   ECHO NNabla CUDA Extension wheel is not found.
   exit /b 255
)

pip install %PIP_INS_OPTS% %WHL% || GOTO :error
pip install %PIP_INS_OPTS% --no-deps %WHLCUDA% || GOTO :error
pip install %PIP_INS_OPTS% pytest pytest-xdist[psutil]~=3.3.1

SET PYTHONPATH=%nnabla_ext_cuda_root%\python\test;%VENV%\Lib\site-packages;%PYTHONPATH%
python -m pytest %PYTEST_OPTS% %nnabla_root%\python\test || GOTO :error

CALL deactivate.bat || GOTO :error

GOTO :end
:error
ECHO failed with error code %errorlevel%.
ENDLOCAL
exit /b %errorlevel%

:end
ENDLOCAL


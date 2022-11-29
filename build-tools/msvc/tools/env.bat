
REM Copyright 2020,2021 Sony Corporation.
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

REM NNabla folder
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

CALL %nnabla_root%\build-tools\msvc\tools\env.bat %1 || GOTO :error

IF [%TEST_NO_CUDA%] == [True] (
   ECHO Use test environment without cuda/cudnn for wheel with lib.
   GOTO :CUDA_CUDNN_SKIP
)
set CUDAVER=%2
set CUDNNVER=%3
FOR /F "TOKENS=1 DELIMS=." %%A IN ("%CUDAVER%") DO SET CUDA_MAJOR=%%A
FOR /F "TOKENS=2 DELIMS=." %%A IN ("%CUDAVER%") DO SET CUDA_MINOR=%%A

REM CHECK CUDA installation

SET NVIDIA_TOOLKIT_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit
SET CUDA_PATH=%NVIDIA_TOOLKIT_PATH%\CUDA\v%CUDAVER%
SET PATH=%NVIDIA_TOOLKIT_PATH%\CUDA\v%CUDAVER%\bin;%PATH%

IF NOT EXIST "%CUDA_PATH%" (
   ECHO CUDA_PATH ^(%CUDA_PATH%^) does not found.
   exit /b 255
)

nvcc --help >NUL 2>NUL && GOTO :NVCC_FOUND
ECHO nvcc does not found in PATH.
exit /b 255
:NVCC_FOUND

REM CHECK CUDNN installation

IF NOT DEFINED CUDNN_PATH (
    IF EXIST "%CUDA_PATH%"\include\cudnn.h (
       set CUDNN_PATH=%CUDA_PATH%
    ) ELSE (
        FOR %%d IN ( %USERPROFILE%\cudnn %ProgramData%\cudnn ) DO (
            IF EXIST %%d (
                PUSHD %%d
                FOR /D %%i IN (cudnn-%CUDAVER%-*-x64-v%CUDNNVER%.*) DO (
                    IF EXIST %%i\cuda\include\cudnn.h (
                        SET CUDNN_PATH=%%d\%%i\cuda
                    )
                )
                POPD
            )
        )
    )
)

IF NOT EXIST "%CUDNN_PATH%" (
   ECHO CUDNN_PATH ^(%CUDNN_PATH%^) does not found.
   exit /b 255
)

python %~dp0get-cudnn-version.py "%CUDNN_PATH%"
CALL %~dp0cudnn_version.bat
DEL %~dp0cudnn_version.bat

IF NOT [%CUDNN_MAJOR%] == [%CUDNNVER%] (
   ECHO CUDNN version %CUDNN_MAJOR% in  ^(%CUDNN_PATH%^) does match expected version %CUDNNVER%
   exit /b 255
)

SET PATH=%CUDNN_PATH%\bin;%PATH%
:CUDA_CUDNN_SKIP

REM Ext CUDA folders
SET nnabla_ext_cuda_root=%~dp0..\..\..
PUSHD .
CD %nnabla_ext_cuda_root%
SET nnabla_ext_cuda_root=%CD%
POPD

IF NOT DEFINED nnabla_ext_cuda_build_folder_name         SET nnabla_ext_cuda_build_folder_name=build
IF NOT DEFINED nnabla_ext_cuda_build_folder_suffix       SET nnabla_ext_cuda_build_folder_suffix=_%CUDA_MAJOR%%CUDA_MINOR%_%CUDNNVER%
IF NOT DEFINED nnabla_ext_cuda_build_folder              SET nnabla_ext_cuda_build_folder=%nnabla_ext_cuda_root%\%nnabla_ext_cuda_build_folder_name%%nnabla_ext_cuda_build_folder_suffix%

IF NOT DEFINED nnabla_ext_cuda_build_wheel_folder_name   SET nnabla_ext_cuda_build_wheel_folder_name=build_wheel
IF NOT DEFINED nnabla_ext_cuda_build_wheel_folder_suffix SET nnabla_ext_cuda_build_wheel_folder_suffix=_py%PYVER_MAJOR%%PYVER_MINOR%_%CUDA_MAJOR%%CUDA_MINOR%_%CUDNNVER%
IF [%TEST_NO_CUDA%] == [True]                            SET nnabla_ext_cuda_build_wheel_folder_suffix=_py%PYVER_MAJOR%%PYVER_MINOR%
IF [%WHL_NO_CUDA_SUFFIX%] == [True]                      SET nnabla_ext_cuda_build_wheel_folder_suffix=_py%PYVER_MAJOR%%PYVER_MINOR%
IF NOT DEFINED nnabla_ext_cuda_build_wheel_folder        SET nnabla_ext_cuda_build_wheel_folder=%nnabla_ext_cuda_root%\%nnabla_ext_cuda_build_wheel_folder_name%%nnabla_ext_cuda_build_wheel_folder_suffix%

IF NOT DEFINED ext_cuda_lib_name_suffix                  SET ext_cuda_lib_name_suffix=_%CUDA_MAJOR%%CUDA_MINOR%_%CUDNNVER%

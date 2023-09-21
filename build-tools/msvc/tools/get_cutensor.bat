@ECHO OFF

REM Copyright 2022 Sony Group Corporation.
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

REM Download pre-built cuTENSOR library

SET CUDAVER=%1
IF %CUDAVER% == 10.2 GOTO :LIB_VER_SET
IF %CUDAVER% == 11.0 GOTO :LIB_VER_SET
SET CUDAVER=11

:LIB_VER_SET

SET cutensor_folder=%third_party_folder%\libcutensor-windows-x86_64-1.7.0.1-archive
SET cutensor_library_dir=%cutensor_folder%\lib\%CUDAVER%
SET cutensor_dll=%cutensor_folder%\lib\%CUDAVER%\cutensor.dll
SET cutensor_include_dir=%cutensor_folder%\include

if NOT EXIST %cutensor_folder%.zip (
    powershell "%nnabla_iwr_script%; iwr %nnabla_iwr_options% -Uri https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/windows-x86_64/libcutensor-windows-x86_64-1.7.0.1-archive.zip -OutFile %cutensor_folder%.zip" || GOTO :error
)

IF EXIST %cutensor_library_dir% (
   ECHO cutensor already exists. Skipping...
   GOTO :Skip_Cutensor
)

PUSHD .
CD %third_party_folder%

cmake -E tar xvzf %cutensor_folder%.zip || GOTO :error
POPD

:Skip_Cutensor
COPY %cutensor_include_dir%\include\* %VENV%\include
COPY %cutensor_dll% %VENV%\Scripts

exit /b

:error
ECHO failed with error code %errorlevel%.

exit /b %errorlevel%

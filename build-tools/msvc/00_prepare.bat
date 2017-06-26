chcp 437

if exist %~dp0local-config.cmd call %~dp0local-config.cmd

rem Make build directories
if not exist %~dp0..\..\build mkdir %~dp0..\..\build
if not exist %~dp0..\..\build\msvc mkdir %~dp0..\..\build\msvc

rem Build
cd %~dp0..\..\build\msvc

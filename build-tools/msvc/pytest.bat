echo off
call %~dp000_prepare.bat

msbuild pytest.vcxproj /p:Configuration=Release

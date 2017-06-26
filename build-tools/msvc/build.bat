@echo off
call %~dp000_prepare.bat

msbuild ALL_BUILD.vcxproj /p:Configuration=Release /verbosity:minimal


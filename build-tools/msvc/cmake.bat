echo off
call %~dp000_prepare.bat

cmake -G "Visual Studio 14 Win64" ../..

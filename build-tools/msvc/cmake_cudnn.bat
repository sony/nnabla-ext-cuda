echo off
call %~dp000_prepare.bat

cmake -DWITH_CUDA=ON -DWITH_CUDNN=ON -G "Visual Studio 14 Win64" ../..

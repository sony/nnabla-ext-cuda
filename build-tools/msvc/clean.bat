@echo off
chcp 437

cd %~dp0..\..
rmdir /s /q build
rmdir /s /q python\src\nnabla.egg-info
del /s /f /q *.pyc

@echo off
setlocal

set "PACKAGE_NAME=neuropipeline"  REM Replace with your package name

:loop
echo Installing package in development mode...
pip install -e "%PACKAGE_NAME%"
if errorlevel 1 goto error

echo Package updated.
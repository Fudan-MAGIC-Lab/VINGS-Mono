@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "HELPER_SRC=%SCRIPT_DIR%Invoke-JetsonScript.ps1"
set "HELPER_DST=%TEMP%\Invoke-JetsonScript-%RANDOM%-%RANDOM%.ps1"

copy /Y "%HELPER_SRC%" "%HELPER_DST%" >nul
if errorlevel 1 exit /b %ERRORLEVEL%

powershell -NoProfile -ExecutionPolicy Bypass -File "%HELPER_DST%" %*
set "EXIT_CODE=%ERRORLEVEL%"

del "%HELPER_DST%" >nul 2>nul
exit /b %EXIT_CODE%

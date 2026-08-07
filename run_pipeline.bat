@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ABORT] python was not found on PATH.
    echo.
    echo Press any key to close this window...
    pause >nul
    exit /b 1
)

python run_pipeline.py
set "EXITCODE=%errorlevel%"

echo.
if %EXITCODE% neq 0 (
    echo Pipeline FAILED with exit code %EXITCODE%.
) else (
    echo Pipeline finished successfully.
)

echo.
echo Press any key to close this window...
pause >nul
exit /b %EXITCODE%

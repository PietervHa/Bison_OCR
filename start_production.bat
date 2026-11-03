@echo off
REM Bison OCR - production start with automatic restart.
REM Starts the app with config\production.yaml; if it stops or crashes it is
REM restarted after 5 seconds. Close this window (or Ctrl+C, then Y) to stop.
cd /d "%~dp0"
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" (
  echo Python not found: %PY%
  echo Adjust PY in this file to your virtual environment.
  pause
  exit /b 1
)
:loop
echo [%date% %time%] Starting Bison OCR...
"%PY%" "%~dp0backend\main.py" --config "%~dp0config\production.yaml"
echo [%date% %time%] Bison OCR stopped (exit code %errorlevel%), restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop

@echo off
cd /d "%~dp0"
if not exist "%~dp0my_app.py" (
  echo [MY_APP] my_app.py not found. Copy mini_app.py to my_app.py first.
  pause
  exit /b 1
)
set "PATH=%~dp0..\.conda;%~dp0..\.conda\Scripts;%~dp0..\.conda\Library\bin;%PATH%"
"%~dp0..\.conda\python.exe" -m streamlit run "%~dp0my_app.py"
if errorlevel 1 pause

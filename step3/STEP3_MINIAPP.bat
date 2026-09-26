@echo off
cd /d "%~dp0"
set "PATH=%~dp0..\.conda;%~dp0..\.conda\Scripts;%~dp0..\.conda\Library\bin;%PATH%"
"%~dp0..\.conda\python.exe" -m streamlit run "%~dp0mini_app.py"
if errorlevel 1 pause

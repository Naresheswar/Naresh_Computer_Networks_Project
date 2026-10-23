@echo off
cd /d "%~dp0"
title GuardCode live dashboard
start "" /min cmd /c "timeout /t 5 >nul & start http://127.0.0.1:5000"
.venv\Scripts\python.exe dashboard\app.py
pause

@echo off
REM Re-train the neural code profiles from scratch (about 20-30 min on a laptop CPU)
cd /d "%~dp0"
set PATH=%~dp0.venv\Scripts;%PATH%
python scripts\train_all.py awgn burst
python -m guardcode.evaluate --frames 200000
pause

@echo off
REM Full report-quality benchmark (1,000,000 frames per point - takes a while)
cd /d "%~dp0"
set VPY=.venv\Scripts\python.exe
%VPY% -m guardcode.evaluate --frames 1000000
%VPY% scripts\analysis.py --frames 1000000
%VPY% scripts\summarize.py
pause

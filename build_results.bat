@echo off
REM Re-runs every experiment with the current code and rebuilds the database:
REM   benchmark (300,000 frames per point) -> analyses -> summary -> 1,000,000-frame DB population
cd /d "%~dp0"
title GuardCode - building results
set VPY=.venv\Scripts\python.exe
%VPY% -m guardcode.evaluate --frames 300000 || goto :err
%VPY% scripts\analysis.py --frames 600000 || goto :err
%VPY% scripts\summarize.py || goto :err
%VPY% scripts\populate_db.py --frames 1000000 || goto :err
if not exist data mkdir data
echo RESULTS_OK> data\results_done.txt
echo.
echo All results rebuilt. Start the dashboard with run_dashboard.bat
pause
exit /b 0
:err
echo *** failed - see above ***
if not exist data mkdir data
echo RESULTS_FAILED> data\results_done.txt
pause
exit /b 1

@echo off
REM ==========================================================================
REM  GuardCode - one-click setup for Windows
REM   1. finds (or installs) Python 3
REM   2. creates a virtual environment in .venv and installs the dependencies
REM   3. loads (or re-runs) the benchmark and calibrates the receiver gates
REM   4. builds the database (1,000,000 simulated frames)
REM ==========================================================================
cd /d "%~dp0"
title GuardCode setup
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY ( where python >nul 2>nul && set PY=python )
if not defined PY (
  echo Python not found - installing Python 3.12 with winget...
  winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
  set PY=python
)
echo Using: %PY%
echo [1/5] Creating virtual environment...
if not exist .venv\Scripts\python.exe ( %PY% -m venv .venv || goto :err )
set VPY=.venv\Scripts\python.exe
echo [2/5] Installing dependencies (torch is large - a few minutes)...
%VPY% -m pip install --upgrade pip
%VPY% -m pip install numpy flask matplotlib || goto :err
%VPY% -m pip install torch || echo torch is optional - only needed to retrain the models
echo [3/5] Benchmark sweep (skipped if results are already present)...
if exist results\benchmark.csv (
  %VPY% -m guardcode.evaluate --import-csv || goto :err
) else (
  %VPY% -m guardcode.evaluate --frames 200000 || goto :err
)
echo [4/5] Building the frames database (1,000,000 rows)...
%VPY% scripts\populate_db.py --frames 1000000 || goto :err
echo [5/5] Supporting analyses (skipped if already present)...
if not exist results\analysis.json ( %VPY% scripts\analysis.py --frames 300000 || goto :err )
echo.
echo ==================================================================
echo  Setup complete. Start the live dashboard with run_dashboard.bat
echo ==================================================================
echo SETUP_OK> data\setup_done.txt
pause
exit /b 0
:err
echo.
echo *** Setup failed - see the messages above. ***
pause
exit /b 1

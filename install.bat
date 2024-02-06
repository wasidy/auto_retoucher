@echo off

set PYTHON_VER=3.10.9

python --version 2>nul | findstr /b /c:"Python %PYTHON_VER%" >nul
if errorlevel 1 (
    echo Warning: Python version %PYTHON_VER% is recommended.
)

IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

mkdir ".\models" > nul 2>&1
mkdir ".\outputs" > nul 2>&1

call .\venv\Scripts\deactivate.bat
call .\venv\Scripts\activate.bat

pip install -r requirements.txt


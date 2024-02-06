@echo off

call .\venv\Scripts\activate.bat
python auto_retoucher.py
call .\venv\Scripts\deactivate.bat


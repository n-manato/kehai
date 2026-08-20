@echo off
setlocal
cd /d C:\DEV\kehai
call .\.venv\Scripts\activate.bat
python run_server.py
echo.
echo kehai exited with code %ERRORLEVEL%.
pause

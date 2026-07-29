@echo off
setlocal
cd /d C:\DEV\open-llm-vtuber
call .\.venv\Scripts\activate.bat
python run_server.py
echo.
echo open-llm-vtuber exited with code %ERRORLEVEL%.
pause

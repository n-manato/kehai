@echo off
setlocal
REM The GPTSoVits conda env is launched without "conda activate", so prepend the
REM env's DLL directories to PATH (what activation would do); otherwise its
REM native deps fail with STATUS_DLL_NOT_FOUND.
set "ENVDIR=C:\DEV\open-llm-vtuber\third_party\miniconda3\envs\GPTSoVits"
set "PATH=%ENVDIR%;%ENVDIR%\Library\bin;%ENVDIR%\Scripts;%PATH%"
cd /d C:\DEV\open-llm-vtuber\third_party\GPT-SoVITS
set PYTHONPATH=%CD%
"%ENVDIR%\python.exe" api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs\tts_infer.yaml
echo.
echo GPT-SoVITS exited with code %ERRORLEVEL%.
pause

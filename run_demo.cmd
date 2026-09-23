@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set DEMO_UTF8_REEXEC=1
python "%~dp0run_demo.py" %*
endlocal

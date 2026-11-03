@echo off
title BERT_READER debug console
cd /d "%~dp0"
echo === BERT_READER X debug console ===
echo Python: .venv-build\Scripts\python.exe
echo All app prints (data loading / errors) stream below.
echo ====================================
".venv-build\Scripts\python.exe" main.py
echo.
echo === App exited (code %errorlevel%) — window stays open ===
pause

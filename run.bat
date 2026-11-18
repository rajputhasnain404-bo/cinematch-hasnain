@echo off
REM CineMatch AI — run the app, then open it in the browser (port 8002).
cd /d %~dp0
python -m pip install -q -r requirements.txt
python -m uvicorn backend.main:app --port 8002

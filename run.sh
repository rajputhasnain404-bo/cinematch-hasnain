#!/bin/bash
# CineMatch AI — run the app, then open it in the browser (port 8002).
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 -m uvicorn backend.main:app --port 8002

"""Vercel entrypoint: re-export the FastAPI app.

Vercel's Python runtime detects this file, loads `app` as the ASGI
application, and routes every request to it. Local runs are unchanged
(run `uvicorn backend.main:app` as before).
"""
import os

# Serverless: never try to open a desktop browser on startup.
os.environ["AUTO_OPEN_BROWSER"] = "0"

from backend.main import app

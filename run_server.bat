@echo off
echo Starting API server...
uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
pause


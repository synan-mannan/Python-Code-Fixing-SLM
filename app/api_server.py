from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from agent.debugger_agent import PythonDebuggerAgent

app = FastAPI(title="Python Debug AI Agent API", version="1.0.0")

class DebugRequest(BaseModel):
    code: str
    traceback: str = ""

agent = PythonDebuggerAgent()  # Loads SLM

@app.post("/debug", response_model=Dict[str, str])
async def debug_python_error(request: DebugRequest):
    
    try:
        result = agent.debug(request.code, request.traceback)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


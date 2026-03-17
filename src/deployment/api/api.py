from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.main import RAG

def _project_root() -> Path:
    # api.py -> src/deployment/api/api.py -> go up 4 levels to project root
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _project_root()

# Make relative paths like "config/config.yaml" work even when uvicorn is started
# from src/deployment/api or elsewhere.
os.chdir(PROJECT_ROOT)

app = FastAPI(title="RAG API")

rag: Optional[RAG] = None

class Request(BaseModel):
    query: str

class Response(BaseModel):
    recommendation: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag is not None,
        "project_root": str(PROJECT_ROOT),
    }


def _get_rag() -> RAG:
    global rag
    if rag is None:
        rag = RAG()
    return rag


@app.post("/reload")
async def reload_rag():
    global rag
    rag = RAG()
    return {"status": "reloaded"}

@app.post("/chat")
async def process_query(query: Request) -> Response:
    try:
        recommendation = _get_rag().process_query(query.query)
        if recommendation is None:
            raise RuntimeError("RAG returned empty response")
        return Response(recommendation=recommendation)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ERROR while processing query: {e}",
        )
        
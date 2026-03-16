from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import RAG

app = FastAPI(title="RAG API")

rag = RAG()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "rag_initialized": True}

@app.post("/chat")
async def process_query(query: str) -> str:
    try:
        response = rag.process_query(query)
        assert response != None
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail="ERROR while processing query: {e}")
        
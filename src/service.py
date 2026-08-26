import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from generate import Generator

app = FastAPI(title="RAG Pipeline API", version="1.0.0")
generator = Generator()


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]


@app.get("/")
def read_root():
    return {"status": "online", "service": "RAG Pipeline API"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = generator.answer_question(request.query)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)

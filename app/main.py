from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import rag
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    rating: str

@app.post("/query")
def query(request: QueryRequest):
    result = rag(request.query)
    return result

feedback_store = []

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    feedback_store.append({
        "query": request.query,
        "rating": request.rating
    })
    return {"status":"received"}

@app.get("/health")
def health():
    return {"status": "ok"}
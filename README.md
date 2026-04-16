# Smart FAQ RAG System
A Retrieval-Augmented Generation (RAG) system for answering FAQ questions using semantic search and LLMs.

# Features
- Semantic search with Qdrant
- Context-aware answers using Groq (LLaMA 3)
- Fallback handling for unknown queries
- Response caching
- Latency tracking
- Feedback collection API

# Architecture
User → FastAPI → Retrieval (Qdrant) → LLM → Response

# API usage
http://localhost:8000/docs

POST /query

{
  "query": "How can I reset my password?"
}

# Example output
{
  "answer": "...",
  
  "mode": "rag",
  
  "latency": 0.3,
  
  "cached": false
}

# How to run
docker run -p 6333:6333 qdrant/qdrant

uvicorn app.main:app --reload

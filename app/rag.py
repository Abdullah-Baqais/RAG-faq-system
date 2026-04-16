import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import time


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
model = SentenceTransformer("all-mpnet-base-v2")
client = QdrantClient(host="localhost", port=6333)
cache = {}

def retrieve(query, top_k=5):
    query_vec = model.encode(query).tolist()

    results = client.query_points(
        collection_name="faq-rag",
        query=query_vec,
        limit=top_k
    )

    docs = []
    scores = []

    for r in results.points:
        docs.append(r.payload)
        scores.append(r.score)

    return docs, scores

# Memory
def build_context(docs):
    context = ""
    for d in docs:
        context += f"Q: {d['question']}\nA: {d['answer']}\n\n"
    return context

def build_prompt(query, context):
    return f"""
        You are a helpful customer support assistant.

        Answer ONLY using the context below.
        If the answer is not in the context, say "Please call the support, or send an email to the supprt".

        Context:
        {context}

        Question:
        {query}

        Answer:
        """


def rag(query):
    start_time = time.time()

    # 1. Check cache
    if query in cache:
        result = cache[query]
        result["cached"] = True
        result["latency"] = time.time() - start_time
        return result

    # 2. Retrieve
    docs, scores = retrieve(query)

    # 3. Fallback logic
    if max(scores) < 0.5:
        fallback_prompt = f"""
        You are a customer support assistant.

        If the question is unrelated to the company or its services, respond with:
        "I can only answer questions related to our services."

        Question:
        {query}

        Answer:
        """
        response = llm.invoke(fallback_prompt)
        result = {
            "answer": response.content,
            "sources": [],
            "mode": "fallback"
        }
    else:
        context = build_context(docs)
        prompt = build_prompt(query, context)

        response = llm.invoke(prompt)

        result = {
            "answer": response.content,
            "sources": docs,
            "mode": "rag"
        }

    result["cached"] = False
    result["latency"] = time.time() - start_time

    # 4. Store in cache
    cache[query] = result

    return result

if __name__ == "__main__":
    q = "How can I reset my password?"
    result = rag(q)

    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
    print(f"\nMode: {result["mode"]}")

    q = "Who is your CEO name?"
    result = rag(q)
    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
    print(f"\nMode: {result["mode"]}")

    q = "Who is your CEO name?"
    result = rag(q)

    print("Answer:", result["answer"])
    print("\nSources:", result["sources"])
    print(f"\nMode: {result["mode"]}")

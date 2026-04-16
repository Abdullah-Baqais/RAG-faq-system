import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

data = []
with open("../data/faq.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(len(data))
print(data[0])

model = SentenceTransformer("all-mpnet-base-v2")

client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists(collection_name="faq-rag"):
    client.create_collection(
        collection_name="faq-rag",
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )


#Helper
def build_text(item):
    return f"Question: {item['question']}\nAnswer: {item['answer']}"


points = []

for i, item in enumerate(data):
    text = build_text(item)
    vector = model.encode(text,show_progress_bar=True).tolist()

    points.append({
        "id": i,
        "vector": vector,
        "payload": {
            "question": item["question"],
            "answer": item["answer"]
        }
    })


client.upsert(
    collection_name="faq-rag",
    points=points
)


#Demo
'''
query = "how do I reset my password?"
query_vec = model.encode(query).tolist()

results = client.query_points(    
    collection_name="faq-rag",
    query=query_vec,
    limit=3
    )

for r in results.points:
    print(r.payload)
'''

import requests
import json
from sentence_transformers import SentenceTransformer

BASE_URL = "http://localhost:8080"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "notes_index"
DIMENSION = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

def create_index():
    url = f"{BASE_URL}/api/v1/index/create"
    payload = {
        "index_name": INDEX_NAME,
        "dimension": DIMENSION
    }
    response = requests.post(url, json=payload)
    print("Create Index:", response.text)

def insert_documents(docs):
    url = f"{BASE_URL}/api/v1/vector/upsert"
    vectors = []

    for i, doc in enumerate(docs):
        embedding = model.encode(doc).tolist()
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": doc}
        })

    payload = {
        "index_name": INDEX_NAME,
        "vectors": vectors
    }

    response = requests.post(url, json=payload)
    print("Insert Response:", response.text)

def search(query):
    url = f"{BASE_URL}/api/v1/vector/search"
    query_vector = model.encode(query).tolist()

    payload = {
        "index_name": INDEX_NAME,
        "query_vector": query_vector,
        "top_k": 3
    }

    response = requests.post(url, json=payload)
    print("Search Results:", response.text)

if __name__ == "__main__":
    docs = [
        "Operating systems manage hardware and software resources.",
        "Deadlock occurs when processes wait indefinitely for resources.",
        "CPU scheduling determines which process runs at a given time.",
        "Memory management handles allocation and deallocation of memory.",
        "Computer networks enable communication between systems."
    ]

    create_index()
    insert_documents(docs)

    query = input("Enter your search query: ")
    search(query)

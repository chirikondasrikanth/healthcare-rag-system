import json
import faiss
import numpy as np
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def load_vector_store():
    """Load FAISS index and metadata"""
    index = faiss.read_index('data/embeddings/faiss_index.bin')
    with open('data/embeddings/chunks_metadata.json', 'r') as f:
        chunks = json.load(f)
    print(f"Vector store loaded with {index.ntotal} vectors!")
    return index, chunks

def search_similar_chunks(query, index, chunks, top_k=3):
    response = client.embeddings.create(
        input=query,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    query_embedding = np.array([response.data[0].embedding]).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx].copy()
        chunk['similarity_score'] = float(distances[0][i])
        results.append(chunk)
    
    return results

if __name__ == "__main__":
    index, chunks = load_vector_store()
    
    # Test search
    query = "Who is eligible for Medicare?"
    results = search_similar_chunks(query, index, chunks)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for r in results:
        print(f"\nChunk: {r['chunk_id']}")
        print(f"Category: {r['category']}")
        print(f"Score: {r['similarity_score']:.4f}")
        print(f"Text: {r['text'][:100]}...")
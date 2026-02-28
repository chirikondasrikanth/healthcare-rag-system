import json
import os
import faiss
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config/.env')

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def load_chunks(filepath):
    """Load chunks from file"""
    with open(filepath, 'r') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks

def generate_embedding(text):
    """Generate embedding using Azure OpenAI"""
    response = client.embeddings.create(
        input=text,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    return response.data[0].embedding

def build_vector_store(chunks):
    """Build FAISS vector store from chunks"""
    print("Generating embeddings...")
    
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}: {chunk['chunk_id']}")
        embedding = generate_embedding(chunk['text'])
        embeddings.append(embedding)
        chunk['embedding_index'] = i
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Build FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    print(f"Vector store built with {index.ntotal} vectors!")
    return index, chunks

def save_vector_store(index, chunks):
    """Save FAISS index and chunks"""
    os.makedirs('data/embeddings', exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, 'data/embeddings/faiss_index.bin')
    
    # Save chunks with metadata
    with open('data/embeddings/chunks_metadata.json', 'w') as f:
        chunks_to_save = [{k: v for k, v in chunk.items() 
                          if k != 'embedding'} for chunk in chunks]
        json.dump(chunks_to_save, f, indent=2)
    
    print("Vector store saved!")

def run_embedding_pipeline():
    print("Starting Embedding Pipeline...")
    
    # Step 1 - Load chunks
    chunks = load_chunks('data/processed/cms_faq_chunks.json')
    
    # Step 2 - Build vector store
    index, chunks = build_vector_store(chunks)
    
    # Step 3 - Save
    save_vector_store(index, chunks)
    
    print("\nEmbedding Pipeline Complete!")
    print(f"Total vectors stored: {index.ntotal}")

if __name__ == "__main__":
    run_embedding_pipeline()
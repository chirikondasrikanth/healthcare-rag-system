import json
import os

def load_processed_data(filepath):
    """Load processed data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} processed records")
    return data

def chunk_documents(data):
    """Convert each FAQ into a chunk ready for embedding"""
    chunks = []
    
    for record in data:
        chunk = {
            "chunk_id": f"{record['id']}_chunk_1",
            "source_id": record['id'],
            "category": record['category'],
            "source": record['source'],
            "text": f"Question: {record['question']}\nAnswer: {record['answer']}",
            "metadata": {
                "category": record['category'],
                "source": record['source'],
                "last_updated": record['last_updated']
            }
        }
        chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks")
    return chunks

def save_chunks(chunks, output_path):
    """Save chunks to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved chunks to {output_path}")

def run_chunking():
    print("Starting Chunking Pipeline...")
    
    # Step 1 - Load processed data
    data = load_processed_data('data/processed/cms_faq_processed.json')
    
    # Step 2 - Chunk
    chunks = chunk_documents(data)
    
    # Step 3 - Save
    save_chunks(chunks, 'data/processed/cms_faq_chunks.json')
    
    print("\nChunking Complete!")
    for chunk in chunks[:3]:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Category: {chunk['category']}")
        print(f"Text Preview: {chunk['text'][:100]}...")

if __name__ == "__main__":
    run_chunking()
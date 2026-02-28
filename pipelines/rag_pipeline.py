import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys
sys.path.append('.')
from vectorstore.vector_store import load_vector_store, search_similar_chunks


load_dotenv('config/.env')

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def generate_answer(query, context_chunks):
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    system_prompt = """You are a helpful Medicare healthcare assistant.
    Answer questions ONLY based on the provided context.
    If the answer is not in the context, say 'I don't have information about that.'
    Always be accurate and cite the source category."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def rag_query(query):
    print(f"\nQuery: {query}")
    print("-" * 50)
    
    index, chunks = load_vector_store()
    relevant_chunks = search_similar_chunks(query, index, chunks, top_k=3)
    print(f"Retrieved {len(relevant_chunks)} relevant chunks")
    
    answer = generate_answer(query, relevant_chunks)
    
    print(f"\nAnswer: {answer}")
    print("-" * 50)
    
    return {
        "query": query,
        "answer": answer,
        "sources": [c['category'] for c in relevant_chunks]
    }

if __name__ == "__main__":
    test_queries = [
        "Who is eligible for Medicare?",
        "What does Medicare Part A cover?",
        "How much is the Medicare Part B premium?"
    ]
    
    for query in test_queries:
        result = rag_query(query)
        print(f"Sources: {result['sources']}\n")
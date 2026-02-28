import sys
sys.path.append('.')

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
from vectorstore.vector_store import load_vector_store, search_similar_chunks

load_dotenv('config/.env')

app = FastAPI(
    title="Healthcare RAG API",
    description="Medicare FAQ System powered by Azure OpenAI and RAG",
    version="1.0.0"
)

# CORS fix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

print("Loading vector store...")
index, chunks = load_vector_store()
print("Vector store loaded!")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    confidence: str

def generate_answer(query, context_chunks):
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    system_prompt = """You are a helpful Medicare healthcare assistant.
    Answer questions ONLY based on the provided context.
    If the answer is not in the context, say 'I dont have information about that.'"""
    user_prompt = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
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

@app.get("/")
def home():
    return {"message": "Healthcare RAG API is running!", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "vectors_loaded": index.ntotal}

@app.post("/query", response_model=QueryResponse)
def query_healthcare(request: QueryRequest):
    try:
        question = request.question.lower().strip()
        
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if any(g in question for g in greetings):
            return QueryResponse(
                question=request.question,
                answer="Hello! 👋 I'm your Medicare Assistant. Ask me about Medicare eligibility, coverage, costs, and more!",
                sources=["Healthcare Assistant"],
                confidence="high"
            )
        
        if len(question) < 5:
            return QueryResponse(
                question=request.question,
                answer="Please ask a Medicare related question. Example: 'Who is eligible for Medicare?'",
                sources=["Healthcare Assistant"],
                confidence="high"
            )

        relevant_chunks = search_similar_chunks(request.question, index, chunks, top_k=3)
        answer = generate_answer(request.question, relevant_chunks)
        sources = list(set([c['category'] for c in relevant_chunks]))
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            confidence="high" if len(relevant_chunks) >= 3 else "medium"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def get_categories():
    categories = list(set([c['category'] for c in chunks]))
    return {"categories": categories}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
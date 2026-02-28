import sys
sys.path.append('.')

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

load_dotenv('config/.env')

app = FastAPI(
    title="Healthcare RAG API",
    description="Medicare FAQ System powered by Azure OpenAI and RAG",
    version="1.0.0"
)

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

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    confidence: str

def search_chunks(query):
    """Search using Azure AI Search"""
    response = client.embeddings.create(
        input=query,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    embedding = response.data[0].embedding

    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=3,
        fields="embedding"
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=3
    )

    return [{"content": r["content"], "category": r["category"]} for r in results]

def generate_answer(query, chunks):
    context = "\n\n".join([c["content"] for c in chunks])
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are a Medicare healthcare assistant. Answer ONLY from the provided context. If not in context say 'I dont have information about that.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
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
    return {"status": "healthy"}

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

        chunks = search_chunks(request.question)
        answer = generate_answer(request.question, chunks)
        sources = list(set([c['category'] for c in chunks]))

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            confidence="high"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
def get_categories():
    return {"categories": ["Medicare Eligibility", "Medicare Parts", "Medicare Cost", "Enrollment", "Prescription Drugs"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
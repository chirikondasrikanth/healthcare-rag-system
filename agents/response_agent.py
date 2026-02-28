import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def response_agent(state):
    """Agent 3 - Generates final response"""
    print("💬 Response Agent running...")
    
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    validation_passed = state.get("validation_passed", False)
    
    if not validation_passed:
        state["final_answer"] = "I don't have enough information to answer that question."
        state["sources"] = []
        return state
    
    # Build context
    context = "\n\n".join([c["content"] for c in chunks])
    sources = list(set([c["category"] for c in chunks]))
    
    system_prompt = """You are a helpful Medicare healthcare assistant.
    Answer ONLY based on the provided context.
    If answer is not in context say 'I dont have information about that.'
    Be concise and accurate."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    
    state["final_answer"] = response.choices[0].message.content
    state["sources"] = sources
    print(f"✅ Response generated!")
    return state
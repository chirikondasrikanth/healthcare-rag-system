import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

search_client = SearchClient(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
)

def search(query):
    # Generate embedding
    response = openai_client.embeddings.create(
        input=query,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    embedding = response.data[0].embedding

    # Vector search
    from azure.search.documents.models import VectorizedQuery
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

    print(f"\nQuery: {query}")
    print("-" * 50)
    for r in results:
        print(f"Category: {r['category']}")
        print(f"Content: {r['content'][:100]}...")
        print()

if __name__ == "__main__":
    search("Who is eligible for Medicare?")
    search("What does Part A cover?")
    search("What is medigap?")
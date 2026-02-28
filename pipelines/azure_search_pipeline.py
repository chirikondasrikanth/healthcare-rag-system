import json
import os
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

# Clients
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX")

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=AzureKeyCredential(search_key)
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

def create_search_index():
    """Create Azure AI Search index with vector support"""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw"
        )]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )

    index_client.create_or_update_index(index)
    print(f"Index '{index_name}' created!")

def generate_embedding(text):
    """Generate embedding using Azure OpenAI"""
    response = openai_client.embeddings.create(
        input=text,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    return response.data[0].embedding

def upload_documents():
    """Upload documents with embeddings to Azure AI Search"""
    with open('data/processed/cms_faq_chunks.json', 'r') as f:
        chunks = json.load(f)

    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key)
    )

    documents = []
    for chunk in chunks:
        print(f"Processing {chunk['chunk_id']}...")
        embedding = generate_embedding(chunk['text'])
        documents.append({
            "id": chunk['chunk_id'],
            "content": chunk['text'],
            "category": chunk['category'],
            "source": chunk['source'],
            "embedding": embedding
        })

    search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents!")

def run_azure_search_pipeline():
    print("Starting Azure AI Search Pipeline...")
    
    # Step 1 - Create index
    create_search_index()
    
    # Step 2 - Upload documents
    upload_documents()
    
    print("\nAzure AI Search Pipeline Complete!")

if __name__ == "__main__":
    run_azure_search_pipeline()
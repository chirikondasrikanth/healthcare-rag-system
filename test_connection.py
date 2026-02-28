import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv('config/.env')

print("Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("Deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
print("API Version:", os.getenv("AZURE_OPENAI_API_VERSION"))

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

try:
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print("SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("ERROR:", e)
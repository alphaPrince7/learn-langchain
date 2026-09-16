from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "Delhi is the capital of India.",
    "The capital of India is Delhi.",
    "Paris is the capital of France."
]

result = embedding.embed_documents(documents)

print(str(result))
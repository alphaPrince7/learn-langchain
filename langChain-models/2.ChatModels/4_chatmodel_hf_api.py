import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=256,
    provider="auto",
    huggingfacehub_api_token=token,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)
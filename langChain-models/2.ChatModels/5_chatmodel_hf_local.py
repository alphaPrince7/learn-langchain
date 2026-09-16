from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
     model_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=256,
        provider="auto"
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of India")

print(result.content)
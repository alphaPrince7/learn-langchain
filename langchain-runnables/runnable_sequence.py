from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

prompt1 = PromptTemplate(
    template = 'write a joke about best humour and high risk {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=256,
    provider="auto",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'AI'})

print(result)
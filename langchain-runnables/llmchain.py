import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

llm = OpenAI(model='gpt-5.6-luna', api_key=os.getenv("OPENAI_API_KEY"))

prompt = PromptTemplate(
    template='suggest a catchy blog about {topic}',
    input_variables=['topic']
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = input('Enter a topic')
output = chain.run(topic)

print("Generated Blog Title:", output)
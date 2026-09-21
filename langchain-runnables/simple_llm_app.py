import os

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = OpenAI(model='gpt-5.6-luna', api_key=os.getenv("OPENAI_API_KEY"))

prompt = PromptTemplate(
    template='suggest a catchy blog about {topic}',
    input_variables=['topic']
)

topic = input('enter a topic')

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.invoke(formatted_prompt)

print('Generated Blog title:', blog_title)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="generate a detail report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate the 5 pointer summary from the following \n {text}',
    input_variables=['text']
)

model = ChatOpenAI(model='gpt-5.6-luna')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'unemployment in IT in india'})

# print(result)

chain.get_graph().print_ascii()
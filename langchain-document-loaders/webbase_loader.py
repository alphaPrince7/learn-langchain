from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=256,
    provider="auto"
)


model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='write a answer the following \n {question} from the following \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.apple.com/in/mac/compare/?modelList=MacBook-Air-M2,Mac-mini-M2'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser
print(chain.invoke({'question':'what is the product we are talking about?', 'text':docs[0].page_content}))
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = TextLoader("docs.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

query = "what are the key takeaways from the document?"
retrieved_docs = retriever.get_relevant_documents(query)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

llm = OpenAI(model='gpt-5.6-luna')

prompt = f'Based on the following text, answer the question: {query} \n \n {retrieved_text}'
answer = llm.predict(prompt)

print("Answer:",answer)
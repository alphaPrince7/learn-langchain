from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

passthrough = RunnablePassthrough()

# print(passthrough.invoke({'name':'prince'}))

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

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'cricket'})
print(result)
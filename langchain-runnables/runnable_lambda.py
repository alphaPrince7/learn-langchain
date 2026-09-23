from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=256,
    provider="auto",
)

model = ChatOpenAI(model='gpt-5.6-luna')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt, model, parser)

def word_counter(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})
# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(lambda x: len(x.split()))
# })


# runnable_word_counter = RunnableLambda(word_counter)

# print(runnable_word_counter.invoke('Hi there how are you?'))

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)
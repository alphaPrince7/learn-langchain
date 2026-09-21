from abc import ABC, abstractclassmethod
import random

class Runnable(ABC):
    # @abstractmethod
    def invoke(input_data):
        pass

class NakliLLM(Runnable):
    def __init__(self):
        print('LLM created')

    def invoke(self, prompt):
        response_list =[
                    'Delhi is the capital os India',
                    'IPL is the cricket league',
                    'AI stands for Artificial Intelligence'
                ]
        
        return{'response': random.choice(response_list)}
        

    def predict(self, prompt):
        response_list =[
            'Delhi is the capital os India',
            'IPL is the cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return{'response': random.choice(response_list)}


llm = NakliLLM()

# print(llm.predict('what is the capital of India'))

class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


template = NakliPromptTemplate(
    template = 'write a {length} poem about {topic}',
    input_variables=['topic']
)

prompt = template.format({'length':'short', 'topic':'india'})

# print(llm.predict(prompt))


class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']

class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']

chain = NakliLLMChain(llm, template)


# print(chain.run({'length':'short', 'topic':'india'}))


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data

parser = NakliStrOutputParser()

chain = RunnableConnector([template, llm, parser])

print(chain.invoke({'length':'long', 'topic':'india'}))

template1 = NakliPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = NakliPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

chain1 = RunnableConnector({template1, llm})
# print(chain1.invoke({'topic':'AI'}))

chain2 = RunnableConnector({template2, llm, parser})
# chain2.invoke({'response': 'this is a joke'})

final_chain = RunnableConnector({chain1, chain2})

print(final_chain.invoke({'topic':'cricket'}))
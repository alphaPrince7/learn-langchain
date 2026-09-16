from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "virat kohli is the best cricker in the world",
    "sachin tendulkar is the former cricketer of india",
    "india is a democratic country",
    "jasprit bumrah is the best bowler in the world",
]

query = "tell me who is the best cricketer in the world"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



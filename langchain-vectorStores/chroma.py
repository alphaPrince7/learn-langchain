from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

doc1 = Document(
    page_content='Virat Kohli (born November 5, 1988) is an Indian international cricketer widely regarded as one of the greatest batsmen in the history of the game',
    metadata={"team": "Royal challenger banglore"}
)

doc2 = Document(
    page_content='Rohit Sharma does not play for or captain a cricket team from Uttar Pradesh (UP); his domestic career is tied exclusively to Mumbai. However, UP has its own prominent presence in domestic circuits and the UP T20 League featuring local franchises and stars like Rinku Singh. There was also a historical first-class player named Rohit Sharma who played for Uttar Pradesh back in the late 1980s and 1990s, who is a completely different person from the international star.',
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content='Mahendra Singh Dhoni does not own or captain a specific Uttar Pradesh (UP) domestic cricket team, but his ancestral roots trace back to Uttarakhand (which was part of Uttar Pradesh when his family lived there).',
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content='Jasprit Bumrah is an elite Indian fast bowler known for his unique action, express pace, and lethal yorkers, while "Thums Up" is a popular Indian cola brand that frequently features Bumrah in high-energy marketing campaigns (such as his "Toofan" collaborations).',
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content='Ravindra Jadeja is one of the world premier active cricketing all-rounders, representing the Indian National Cricket Team internationally and the Rajasthan Royals (RR) in the Indian Premier League (IPL). Known for his lightning-fast left-arm orthodox spin, reliable lower-order left-handed batting, and world-class fielding athleticism, Jadeja has established himself as an indispensable multi-format player.',
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function= HuggingFaceEndpointEmbeddings(),
    persist_directory='chroma_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

vector_store.get(include=['embeddings', 'documents', 'metadatas'])

result = vector_store.similarity_search(
    query='who amomg these are a bowler?',
    k=2
)

# print(result)

vector_store.similarity_search_with_score(
    query="who among there is bowler?",
    k=2
)

vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

updated_doc1 = Document(
    page_content='Virat Kohli is back in India's ODI squad as he prepares for the upcoming three-match series against the West Indies starting September 27, 2026.',
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='', document=updated_doc1)

vector_store.get(include=['embeddings', 'documents', 'metadatas'])
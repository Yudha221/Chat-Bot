from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the pinecone
pc = Pinecone(api_key="pcsk_6bCDku_SYnG1QeQTGJheCLLSBdf4i9J9CrXBmUGwhsB5QhBmg6BQdZoUhUSrteQAPYoXbS")
index = pc.Index("medical-chatbot")

#Creating Embeddings for each of the text chunks & storing 
docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], index_name="medical-chatbot", embedding=embeddings)
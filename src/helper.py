from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings  # <- ini ganti
from langchain_community.vectorstores import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Extract data from PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

# Create text chunks
def text_split(extracted_data):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_spliter.split_documents(extracted_data)
    return text_chunks

# Download embedding model
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

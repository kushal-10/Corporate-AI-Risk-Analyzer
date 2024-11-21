from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
import json  # Import the json module
from tqdm import tqdm

def generate_docs(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Use open-source embeddings from Hugging Face's sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    search_query = """Artificial Intelligence"""

    retriever = db.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(search_query)

    return docs

if __name__ == "__main__":
    
    file_path = "annual_txts/Germany/1.SAP_$240.94 B_Information Tech/2023/results.txt"

    loader = TextLoader(file_path)
    docs = generate_docs(loader)

    for doc in docs:
        print(doc.page_content)
        print("-"*100)
    print(len(docs))        

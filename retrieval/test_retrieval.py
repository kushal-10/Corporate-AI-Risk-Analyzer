from langchain.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
import json  # Import the json module
from tqdm import tqdm

NUM_DOCS = 3000
SCORE_THRESHOLD = 0.99

def generate_docs(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = FAISS.from_documents(texts, embeddings)
    bm25_retriever = BM25Retriever.from_documents(texts)

    search_query = "artificial intelligence"

    bm25_retriever.k = 1000
    retriever = faiss_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7, "fetch_k": 10000, "k": 10000})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])

    docs = ensemble_retriever.invoke(search_query)

    filtered_docs = [doc for doc in docs if doc.score > 0.7]

    return filtered_docs

if __name__ == "__main__":
    
    file_path = "annual_txts/Germany/1.SAP_$240.94 B_Information Tech/2018/results.txt"

    loader = TextLoader(file_path)
    docs = generate_docs(loader)

    for doc in docs:
        print(doc.page_content)
        print("-"*100)
    print(len(docs))        

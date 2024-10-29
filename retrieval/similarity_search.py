from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import os
def generate_docs(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Use open-source embeddings from Hugging Face's sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    search_query = """Artificial Intelligence, Machine Learning, Data Science, Neural Networks, Robotics, Big Data, Deep Learning"""

    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )
    docs = retriever.invoke(search_query)

    return docs

if __name__ == "__main__":
    base_dir = "annual_txts"
    country_dirs = os.listdir(base_dir)
    for country_dir in country_dirs:
        for company_dir in os.listdir(os.path.join(base_dir, country_dir)):
            for year_dir in os.listdir(os.path.join(base_dir, country_dir, company_dir)):
                loader = TextLoader(os.path.join(base_dir, country_dir, company_dir, year_dir, "results.txt"))
                docs = generate_docs(loader)
                for doc in docs:
                    print(doc.page_content)
                    print("-"*100)
  

    loader = TextLoader(os.path.join("annual_txts/USA/3.NVIDIA_$2.638 T_Information Tech/2023/results.txt"))
    docs = generate_docs(loader)
    for doc in docs:
        print(doc.page_content)
        print("-"*100)
    print(len(docs))
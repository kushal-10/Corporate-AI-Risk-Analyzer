"""
Use FAISS documents for retrieval
"""

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

    search_query = """Artificial Intelligence, Machine Learning, Data Science, Neural Networks, Robotics, Big Data, Deep Learning"""

    retriever = db.as_retriever()
    docs = retriever.invoke(search_query)

    return docs

if __name__ == "__main__":
    base_dir = "annual_txts"
    country_dirs = os.listdir(base_dir)
    
    # Load existing documents from JSON file if it exists
    docs_dict = {}
    try:
        with open('retrieval/retrieved_docs.json', 'r') as json_file:
            docs_dict = json.load(json_file)  # Load existing data
    except FileNotFoundError:
        pass  # If the file doesn't exist, start with an empty dictionary

    for country_dir in country_dirs:
        for company_dir in tqdm(os.listdir(os.path.join(base_dir, country_dir)), desc=f"Processing {country_dir}"):
            for year_dir in os.listdir(os.path.join(base_dir, country_dir, company_dir)):
                file_path = os.path.join(base_dir, country_dir, company_dir, year_dir, "results.txt")
                
                # Check if the key already exists
                if file_path not in docs_dict:
                    loader = TextLoader(file_path)
                    docs = generate_docs(loader)
                    if len(docs) == 0:
                        print(f"No docs found for {file_path}")
            
                    # Store the documents in the dictionary with the path as the key
                    docs_dict[file_path] = [doc.page_content for doc in docs]
                    
                    # Save the updated dictionary to a JSON file immediately after retrieval
                    with open('retrieval/retrieved_docs.json', 'w') as json_file:
                        json.dump(docs_dict, json_file, indent=4)  # Write the updated dictionary to a JSON file with indentation

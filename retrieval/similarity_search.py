from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

def generate_docs(loader):
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    
    # Use open-source embeddings from Hugging Face's sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    search_query = """
    advancements, risks, or opportunities in AI, ML, cybersecurity, data science, automation, neural networks, robotics, big data, deep learning, or technology innovation.
    """

    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": 10}
    )
    docs = retriever.invoke(search_query)

    return docs

if __name__ == "__main__":
    loader = TextLoader("annual_txts/USA/3.NVIDIA_$2.638 T_Information Tech/2023/results.txt")
    docs = generate_docs(loader)
    loader2 = TextLoader("annual_txts/USA/3.NVIDIA_$2.638 T_Information Tech/2023/results.txt")
    docs2 = generate_docs(loader2)

    print("FETCHED DOCS for nvidia #############################################################################")
    for doc in docs:
        print(doc.page_content)
        print("-"*100)
   
    print("FETCHED DOCS for google #############################################################################")
    for doc in docs2:
        print(doc.page_content)
        print("-"*100)
    
    print(len(docs))
    print(len(docs2))
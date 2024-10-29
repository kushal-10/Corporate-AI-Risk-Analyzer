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
    Find text mentioning or discussing advancements, risks, or opportunities in:
    - Artificial Intelligence (AI)
    - Machine Learning (ML)
    - Cybersecurity
    - Data Science
    - Automation
    - Neural Networks
    - Robotics
    - Big Data
    - Deep Learning
    - Innovation in technology

    Include sections discussing technology-related risks, benefits, regulatory impacts, and strategic initiatives.
    Focus on phrases like "AI strategy," "AI risks," "impact of AI," "AI-driven," "ML-based," "cyber threats," "security risks," "automation benefits," and "technological innovation."

    Retrieve relevant content that may discuss any emerging technologies, ethical considerations, regulatory concerns, or competitive advantages associated with these technologies.
    """

    retriever = db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 10}
    )
    docs = retriever.invoke(search_query)

    return docs

if __name__ == "__main__":
    loader = TextLoader("annual_txts/USA/3.NVIDIA_$2.638 T_Information Tech/2023/results.txt")
    docs = generate_docs(loader)
    print("FETCHED DOCS #############################################################################")
    for doc in docs:
        print(doc.page_content)
        print("-"*100)
    
    print(len(docs))
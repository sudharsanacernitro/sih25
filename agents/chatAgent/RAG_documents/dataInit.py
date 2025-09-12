from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()

def create_vector_db():
    try:
        loader = TextLoader("./agents/chatAgent/RAG_documents/data.txt")
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, embeddings)

        # 🔹 Save FAISS index + metadata
        db.save_local(f"./agents/chatAgent/RAG_documents/vectorEmbeddings")


    except Exception as e:
        raise RuntimeError(f"Failed to create vector DB: {str(e)}")


if __name__ == "__main__":
    create_vector_db()
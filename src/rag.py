import os
import shutil

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.utils import get_embedding_model


load_dotenv(find_dotenv())

DB_PATH = "./chroma_db"
KB_PATH = "./knowledge_base"


def initialize_vector_store():
    """Ingest documents and create the Chroma vector database."""
    print(" --- RAG INIT STARTED ---")

    if not os.path.exists(KB_PATH):
        os.makedirs(KB_PATH, exist_ok=True)
        print(f" Created missing folder: {KB_PATH}")
        return

    loader = DirectoryLoader(KB_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    if not docs:
        print(" No documents found in knowledge_base/")
        return
    print(f" Found {len(docs)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    splits = text_splitter.split_documents(docs)

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(" Cleared old database.")

    print(" Creating embeddings...")

    embedding_fn = get_embedding_model()

    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_fn,
    )

    vectorstore.add_documents(documents=splits)

    print(f" SUCCESS: Vector store created at {DB_PATH}")


def get_retriever():
    """Return the retriever for the agents (lazy vector store init)."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            "Vector DB not found. Run 'python -m src.rag' to build it first."
        )

    embedding_fn = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_fn,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})


if __name__ == "__main__":
    initialize_vector_store()


import os
import shutil
from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.utils import get_embedding_model

load_dotenv(find_dotenv())

DB_PATH = "./chroma_db"
KB_PATH = "./knowledge_base"


class _EmptyRetriever(BaseRetriever):
    """Retriever that returns no documents. Used when no vector DB exists."""

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return []


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
    """Return the retriever for the agents. If no DB exists, tries to build it;
    if still no DB (e.g. empty knowledge_base), returns an empty retriever so
    the app still runs (solver uses no RAG context).
    """
    if os.path.exists(DB_PATH):
        embedding_fn = get_embedding_model()
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_fn,
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})

    # Try to build DB from knowledge_base (e.g. first run with files present)
    initialize_vector_store()
    if os.path.exists(DB_PATH):
        return get_retriever()

    # No DB (e.g. on Spaces with no knowledge_base docs) — return empty retriever
    return _EmptyRetriever()


if __name__ == "__main__":
    initialize_vector_store()

# chromadb vector store

from functools import lru_cache
from langchain_chroma import Chroma
from src.embedder import get_embedding_model
from src.config import CHROMA_COLLECTION, VECTORDB_DIR


@lru_cache(maxsize=1)
def get_vector_store(collection_name: str = None):
    embedding = get_embedding_model()

    vectordb = Chroma(
        collection_name=collection_name or CHROMA_COLLECTION,
        embedding_function=embedding,
        persist_directory=VECTORDB_DIR,
    )

    return vectordb


def get_chunk_count() -> int:
    try:
        vectordb = get_vector_store()
        collection = vectordb._collection
        return collection.count()
    except Exception:
        return 0

# retriever - semantic search on chromadb

from typing import List, Dict, Any
from src.vector_store import get_vector_store
from src.config import TOP_K


def retrieve_documents(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    # search chromadb for relevant chunks
    vectordb = get_vector_store()
    k = top_k or TOP_K
    
    # Similarity search with scores
    results = vectordb.similarity_search_with_score(query, k=k)
    
    documents = []
    for doc, score in results:
        documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": round(float(score), 4),
        })
    
    return documents

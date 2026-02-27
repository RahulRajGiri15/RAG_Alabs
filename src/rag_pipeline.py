# rag pipeline - connects retriever, prompt builder, and generator

from typing import Dict, Any, Iterator, List
from src.retriever import retrieve_documents
from src.generator import generate_streaming_response, generate_response


def format_context(documents: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc["metadata"].get("source", "Unknown")
        page = doc["metadata"].get("page", "N/A")
        context_parts.append(
            f"[Chunk {i}] (Source: {source}, Page: {page})\n{doc['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def query_rag(user_query: str, top_k: int = 5) -> Dict[str, Any]:
    # retrieve chunks -> build context -> stream response from LLM
    # Step 1: Retrieve relevant chunks
    documents = retrieve_documents(user_query, top_k=top_k)
    
    # No documents found → don't hallucinate
    if not documents:
        return {
            "stream": iter(["I couldn't find any relevant information in the document."]),
            "sources": [],
            "blocked": True,
        }
    
    # Step 2: Format context
    context = format_context(documents)
    
    # Step 3: Generate streaming response
    stream = generate_streaming_response(user_query, context)
    
    return {
        "stream": stream,
        "sources": documents,
        "blocked": False,
    }

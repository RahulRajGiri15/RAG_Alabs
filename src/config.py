"""
Configuration settings for the RAG chatbot.
All configurable parameters in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()
#paths
DATA_DIR = "data"
CHUNKS_DIR = "chunks"
VECTORDB_DIR = "vectordb"

#embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#chromadb
CHROMA_COLLECTION = "document_chunks"

#chunking
CHUNK_SIZE = 800        # ~150-200 words
CHUNK_OVERLAP = 100     # overlap between chunks

#llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

#retriever
TOP_K = 5               # number of chunks to retrieve

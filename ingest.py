# document ingestion - loads PDFs, chunks them, embeds and stores in chromadb

import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHUNKS_DIR, VECTORDB_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.vector_store import get_vector_store


def load_pdfs(data_dir: str):
    documents = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            print(f"  Loading: {filename}")
            
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            # Add source filename to metadata
            for doc in docs:
                doc.metadata["source"] = filename
            
            documents.extend(docs)
    
    return documents


def chunk_documents(documents):
    # split into ~100-300 word chunks using sentence-aware splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk_id to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks


def save_chunks_json(chunks, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "word_count": len(chunk.page_content.split()),
        })
    
    output_path = os.path.join(output_dir, "chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(chunks_data)} chunks to {output_path}")


def ingest():
    print("=" * 50)
    print("RAG Document Ingestion Pipeline")
    print("=" * 50)
    
    # Step 1: Load documents
    print(f"\n[1/4] Loading PDFs from '{DATA_DIR}'...")
    documents = load_pdfs(DATA_DIR)
    
    if not documents:
        print("  No PDF files found in data/ directory!")
        print("  Please add your PDF documents to the data/ folder.")
        return
    
    print(f"  Loaded {len(documents)} pages from PDFs")
    
    # Step 2: Chunk documents
    print(f"\n[2/4] Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    
    # Step 3: Save chunks as JSON
    print(f"\n[3/4] Saving chunks to '{CHUNKS_DIR}'...")
    save_chunks_json(chunks, CHUNKS_DIR)
    
    # Step 4: Embed and store in ChromaDB
    print(f"\n[4/4] Embedding and storing in ChromaDB at '{VECTORDB_DIR}'...")
    vectordb = get_vector_store()
    vectordb.add_documents(chunks)
    
    collection = vectordb._collection
    total = collection.count()
    
    print(f"  Stored {total} chunks in ChromaDB")
    print("\n" + "=" * 50)
    print("Ingestion complete!")
    print("=" * 50)


if __name__ == "__main__":
    ingest()

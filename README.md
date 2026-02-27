# RAG Document Chatbot

A chatbot that answers questions from uploaded PDF documents using a RAG (Retrieval-Augmented Generation) pipeline. Uses ChromaDB for vector storage, Groq API for LLM inference (LLaMA 3.3 70B), and Streamlit for the UI.

## How it works

1. PDFs are loaded and split into chunks (~150-200 words each)
2. Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB
3. When a user asks a question, the retriever searches for the most relevant chunks
4. Those chunks + the question are sent to LLaMA 3.3 via Groq API
5. The response is streamed token-by-token back to the Streamlit UI

## Project Structure

```
├── data/               # PDF documents
├── chunks/             # processed chunks (JSON)
├── vectordb/           # persisted ChromaDB
├── notebooks/          # analysis notebook
├── src/
│   ├── config.py       # all config in one place
│   ├── embedder.py     # embedding model wrapper
│   ├── vector_store.py # chromadb setup
│   ├── retriever.py    # semantic search
│   ├── generator.py    # groq api + streaming
│   └── rag_pipeline.py # ties everything together
├── app.py              # streamlit app
├── ingest.py           # ingestion script
├── requirements.txt
└── README.md
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Get a Groq API key**

Go to [console.groq.com](https://console.groq.com), sign up (free), and create an API key. Then:

```bash
cp .env.example .env
# paste your GROQ_API_KEY in .env
```

**3. Put your PDFs in `data/`**

**4. Run ingestion**

```bash
python ingest.py
```

This loads the PDFs, chunks them, generates embeddings, and stores everything in ChromaDB.

**5. Start the chatbot**

```bash
streamlit run app.py
```

## Why these choices

I chose **all-MiniLM-L6-v2** for embeddings because it's lightweight (384 dimensions, ~80MB) and works well for semantic search without needing a GPU. For the vector database I went with **ChromaDB** since it's simple to set up — just runs embedded in Python with disk persistence, no separate server needed.

For the LLM I'm using **LLaMA 3.3 70B** through Groq's API. Groq isn't an LLM itself — it's an inference provider that runs open-source models on their custom hardware, which makes streaming really fast. The free tier is enough for this project.

Chunking is done with LangChain's `RecursiveCharacterTextSplitter` (800 chars, 100 overlap) which splits on paragraph and sentence boundaries so chunks don't break mid-sentence.

## Features

- streaming responses (token by token)
- shows source passages with relevance scores
- grounded answers — refuses to answer if info isn't in the document
- sidebar shows current model, embedding info, and chunk count
- clear chat button

## Sample Queries

> _TODO: Add example queries and screenshots after running with your document._

---

Built for Amlgo Labs AI Engineer Assignment.

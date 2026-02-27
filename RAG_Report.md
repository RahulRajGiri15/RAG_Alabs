# Document-Based RAG Chatbot with Streaming Responses

This project is an AI-powered chatbot designed to answer user questions strictly based on the content of uploaded documents. It follows a Retrieval-Augmented Generation (RAG) approach, where relevant information is first retrieved from documents and then used by a large language model to generate accurate, grounded responses.

The application is built using Streamlit, ChromaDB, and LLaMA 3.3 70B via the Groq API, with real-time streaming support.

**Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1bV4zeH3EUh5dmkZ3o5ymQI9GCzOMFzwF/view?usp=sharing)

---

## System Architecture

The system architecture is designed as a pipeline that processes user queries step by step. When a user submits a question through the Streamlit interface, the query is converted into an embedding and passed to a semantic retriever. The retriever searches the ChromaDB vector database to find the most relevant document chunks based on similarity.

The top relevant chunks are then forwarded to the generation module. This module uses the Groq API with the LLaMA 3.3 70B model to generate a response. The answer is streamed token by token back to the user, making the interaction smooth and interactive.

---

## Project Structure

The project is organized in a modular way to keep the pipeline clean and maintainable.

```
├── data/               # Input PDF documents
├── chunks/             # Extracted and processed text chunks (JSON)
├── vectordb/           # Persistent ChromaDB storage
├── notebooks/          # Experiments and testing notebooks
├── src/
│   ├── config.py       # Global configuration
│   ├── embedder.py     # Embedding generation logic
│   ├── vector_store.py # ChromaDB integration
│   ├── retriever.py    # Semantic search module
│   ├── generator.py    # Groq API + streaming logic
│   └── rag_pipeline.py # End-to-end RAG pipeline
├── app.py              # Streamlit chatbot interface
├── ingest.py           # Document ingestion script
├── requirements.txt
└── README.md
```

---

## Setup and Installation

**1. Install Required Dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure Groq API Key**

Visit https://console.groq.com, sign up and generate an API key. Create and update the .env file:

```bash
cp .env.example .env
# Add your GROQ_API_KEY inside the file
```

**3. Add Documents**

Place the PDF files you want the chatbot to reference inside the `data/` directory.

**4. Run Document Ingestion**

```bash
python ingest.py
```

This step performs the following actions:
- Loads PDF documents from the `data/` folder
- Splits text into sentence-aware chunks (around 100-300 words)
- Generates vector embeddings using all-MiniLM-L6-v2
- Stores embeddings in ChromaDB (`vectordb/`)
- Saves processed chunks in JSON format (`chunks/`)

**5. Launch the Chatbot**

```bash
streamlit run app.py
```

Once started, users can ask questions and receive streamed answers grounded in the uploaded documents.

---

## Model and Technology Choices

| Component | Selected Tool | Reason |
|-----------|--------------|--------|
| Embedding Model | all-MiniLM-L6-v2 | Lightweight, fast, and effective for semantic similarity |
| Vector Database | ChromaDB | Simple, local, and does not require a separate server |
| LLM | LLaMA 3.3 70B (Groq API) | High-quality open-source model with fast inference |
| Chunking Strategy | RecursiveCharacterTextSplitter | Maintains sentence and paragraph structure |

---

## Key Features

- Token-by-token streaming responses for better user experience
- Displays source document chunks with relevance scores
- Prevents hallucinations by answering only from retrieved content
- Sidebar showing model details and indexed chunk count
- Option to clear chat and reset the session
- Sentence-aware document chunking for improved retrieval accuracy

---

## Example Usage

The chatbot can answer document-based questions such as:
- "Summarize the key points from this document"
- "What does the document say about a specific topic?"
- "Explain this concept using the provided PDFs"

**Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1bV4zeH3EUh5dmkZ3o5ymQI9GCzOMFzwF/view?usp=sharing)
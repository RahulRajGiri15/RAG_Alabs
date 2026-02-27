# streamlit chatbot app

import streamlit as st
from src.rag_pipeline import query_rag
from src.vector_store import get_chunk_count
from src.config import GROQ_MODEL, EMBEDDING_MODEL, CHROMA_COLLECTION



st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📄",
    layout="wide",
)

# custom css for source boxes
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# sidebar
with st.sidebar:
    st.title("⚙️ System Info")
    st.divider()
    
    st.markdown(f"**LLM Model:** `{GROQ_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Vector DB:** ChromaDB")
    st.markdown(f"**Collection:** `{CHROMA_COLLECTION}`")
    
    # Show chunk count
    chunk_count = get_chunk_count()
    st.metric("Indexed Chunks", chunk_count)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Built with Streamlit + LangChain + Groq")


# main chat area
st.title("📄 RAG Document Chatbot")
st.caption("Ask questions about the uploaded documents. Answers are grounded in the source text.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(" Source Passages", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    src_file = source["metadata"].get("source", "Unknown")
                    page = source["metadata"].get("page", "N/A")
                    score = source.get("score", "N/A")
                    
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Chunk {i}</strong> | Source: {src_file} | Page: {page} | Score: {score}<br>'
                        f'{source["content"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if user_input := st.chat_input("Ask a question about the document..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and stream response
    with st.chat_message("assistant"):
        result = query_rag(user_input)
        
        # Stream the response
        response = st.write_stream(result["stream"])
        
        # Show sources
        sources = result["sources"]
        if sources:
            with st.expander("📚 Source Passages", expanded=False):
                for i, source in enumerate(sources, 1):
                    src_file = source["metadata"].get("source", "Unknown")
                    page = source["metadata"].get("page", "N/A")
                    score = source.get("score", "N/A")
                    
                    st.markdown(
                        f'<div class="source-box">'
                        f'<strong>Chunk {i}</strong> | Source: {src_file} | Page: {page} | Score: {score}<br>'
                        f'{source["content"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    
    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

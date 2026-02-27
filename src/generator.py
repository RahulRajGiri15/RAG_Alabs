# LLM response generation using Groq API

from typing import Iterator
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL


# prompt template

SYSTEM_PROMPT = """You are a helpful document assistant. Your job is to answer questions 
based ONLY on the provided context from the document.

Rules:
- Answer based ONLY on the context provided below.
- If the answer is not in the context, say "I don't have enough information in the document to answer this."
- Be concise and accurate.
- Use bullet points or numbered lists when appropriate.
- Cite the relevant section when possible."""


def build_prompt(query: str, context: str) -> str:
    return f"""Context from the document:
---
{context}
---

Question: {query}

Answer based on the context above:"""




def get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def generate_response(query: str, context: str) -> str:
    client = get_client()
    prompt = build_prompt(query, context)
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content


def generate_streaming_response(query: str, context: str) -> Iterator[str]:
    # streams tokens as they come from the API
    client = get_client()
    prompt = build_prompt(query, context)
    
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

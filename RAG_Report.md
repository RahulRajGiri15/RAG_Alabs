# RAG Document Chatbot — Technical Report

**Assignment:** Junior AI Engineer — Amlgo Labs  
**Project:** Fine-Tuned RAG Chatbot with Streaming Responses  

---

## 1. Document Structure & Chunking Logic

### Document Overview
The system processes an **eBay User Agreement** document (~10,500+ words, 20 pages) covering Terms & Conditions, Privacy Policies, and Legal Contracts — typical legal documents with dense, structured text.

### Chunking Strategy
Documents are split using **RecursiveCharacterTextSplitter** from LangChain with:
- **Chunk size:** 800 characters (~150-200 words)
- **Chunk overlap:** 100 characters (ensures context continuity across boundaries)
- **Separators:** `["\n\n", "\n", ". ", " ", ""]` — sentence-aware splitting that respects paragraph and sentence boundaries

This produces chunks in the **100-200 word range**, with most chunks falling between 100-135 words. The sentence-aware separators ensure chunks don't break mid-sentence, preserving semantic coherence. Each chunk is assigned a unique `chunk_id` and retains metadata (source file, page number) for traceability.

---

## 2. Embedding Model & Vector Database

### Embedding Model: `all-MiniLM-L6-v2`
- **Type:** Sentence Transformer (HuggingFace)
- **Dimensions:** 384
- **Why chosen:** Fast inference, small model size (~80MB), and strong performance on semantic similarity benchmarks. Suitable for a lightweight RAG system without requiring GPU resources.

### Vector Database: ChromaDB
- **Type:** Embedded vector database (no server setup needed)
- **Storage:** Persistent on disk (`vectordb/` directory)
- **Similarity metric:** Cosine similarity (default)
- **Why chosen:** Zero-config setup, Python-native, built-in persistence. Ideal for a standalone demo project where simplicity and portability matter.

---

## 3. Prompt Format & Generation Logic

### System Prompt
The LLM receives a system-level instruction that constrains it to answer **only** from the provided context:

```
You are a helpful document assistant. Answer based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information."
Be concise. Use bullet points when appropriate. Cite relevant sections.
```

### User Prompt Template
```
Context from the document:
---
[Chunk 1] (Source: document1.pdf, Page: 4)
<chunk content>

---
[Chunk 2] ...
---

Question: <user query>

Answer based on the context above:
```

### LLM: LLaMA 3.3 70B via Groq API
- **Temperature:** 0.3 (low creativity, factual responses)
- **Max tokens:** 1024
- **Streaming:** Enabled (token-by-token via `stream=True`)
- **Why Groq:** Groq provides free API access to open-source LLMs (LLaMA, Mistral) with ultra-fast inference using their custom LPU hardware. It is **not** an LLM itself — it's an inference platform.

---

## 4. Example Queries & Responses

### ✅ Query 1 — Success (Direct Factual Answer)
**Q:** "What are the fees charged to sellers on eBay?"  
**A:** The chatbot correctly retrieves chunks from Section 6 (Fees and Taxes) and provides a structured answer about selling fees, payment due dates, and late fee policies. Sources from pages 4-5 are cited.

### ✅ Query 2 — Success (Policy Explanation)
**Q:** "What happens if my eBay account is suspended?"  
**A:** The chatbot retrieves relevant chunks about account suspension, policy enforcement, and user violations. It correctly explains that eBay may limit, suspend, or terminate accounts for policy violations.

### ✅ Query 3 — Success (Legal Clause)
**Q:** "What is the arbitration agreement?"  
**A:** Accurately retrieves the arbitration clause from Section 19, explaining that users agree to binding arbitration unless they opt out, and that class-action lawsuits are waived.

### ⚠️ Query 4 — Partial Success (Multi-Section Answer)
**Q:** "What are my rights as a buyer on eBay?"  
**A:** Retrieves chunks from Purchase Conditions (Section 8) but may miss some buyer protection details scattered across other sections. The top-5 retrieval captures the primary rules but not all edge cases.  
**Limitation:** Information spread across many sections may not all fit in the top-K retrieval window.

### ❌ Query 5 — Expected Failure (Out-of-Scope)
**Q:** "What is the weather in New York today?"  
**A:** The chatbot correctly responds: "I don't have enough information in the document to answer this." — demonstrating the anti-hallucination guard works as intended.

---

## 5. Limitations & Known Issues

| Issue | Details |
|---|---|
| **No conversation memory** | Each query is independent; the LLM doesn't remember previous Q&A in the session |
| **Retrieval ceiling** | Top-K=5 may miss relevant chunks when information is scattered across many sections |
| **Small chunks for legal text** | Legal documents have cross-referencing sections; chunking may break these connections |
| **API dependency** | Requires active Groq API key and internet connection; no offline fallback |
| **Hallucination risk** | While minimized by prompt design, the LLM may occasionally paraphrase inaccurately for complex legal clauses |
| **Single document type** | Only tested with legal/T&C documents; performance on other document types (technical manuals, research papers) is untested |

---

*Report prepared for Amlgo Labs AI Engineer Assignment.*

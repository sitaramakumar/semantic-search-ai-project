# Semantic Search AI Project: Build Guide

> End-to-end guide to building an LLM + embedding pipeline for semantic document search — from scratch to running prototype.

---

## 1. Project Overview

### What It Does
Searches documents by **meaning** (not keywords). A user asks a natural-language question (e.g., "How do I reset my password?") and the system finds relevant documents even if they don't contain those exact words.

### Simple Workflow (No Vector DB)
```
Documents → Chunk → Embed → Store in JSON
Query → Embed → Cosine Similarity → Top-K → LLM Summarizes Answer
```

---

## 2. Prerequisites

### What You Need
| Item | Requirement |
|------|-------------|
| Python | 3.8+ |
| OpenAI API Key | Get from https://platform.openai.com/api-keys |
| Disk Space | 100MB free (for deps + sample docs) |
| Internet | Required (API calls) |

### Local Setup (No API Costs)
If you want to avoid API costs during development:
```bash
pip install sentence-transformers faiss-cpu # Local embeddings + similarity
```

---

## 3. Project Structure

```
semantic-search-ai-project/
├── .env                          # API keys (DO NOT COMMIT)
├── requirements.txt              # Python dependencies
├── ingest.py                     # Document ingestion + embedding
├── search.py                     # Query + similarity search + LLM answer
├── app.py                        # Streamlit web UI (optional)
├── knowledge_base.json           # Embedded document index (auto-generated)
├── my_documents/                 # Your documents go here
│   ├── faq.txt
│   └── manual.txt
└── README.md                     # This file
```

---

## 4. Step-by-Step Build Guide

### Step 1: Create Project Directory and Set Up

```bash
mkdir semantic-search-ai-project
cd semantic-search-ai-project
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### Step 2: Create .env File

Create `.env` in the project root:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

> **Security note:** Never commit `.env` to version control. Add to `.gitignore`.

### Step 3: Install Dependencies

Create `requirements.txt`:

```txt
openai>=1.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
streamlit>=1.30.0
```

Install:
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Sample Documents

Create `my_documents/faq.txt`:
```
Q: How do I reset my password?
A: Go to the login page, click "Forgot Password", enter your email, and follow the reset link sent to you.

Q: What is the refund policy?
A: We offer full refunds within 30 days of purchase. After 30 days, no refunds are available.

Q: How do I contact support?
A: Email support@company.com or call 1-800-SUPPORT. Response time is within 24 hours.

Q: Can I change my subscription plan?
A: Yes, you can upgrade or downgrade your plan at any time in Account Settings. Changes take effect immediately.

Q: Where do I find my invoice?
A: Invoices are available in Billing History under Account Settings. You can download PDFs from there.
```

Create `my_documents/manual.txt`:
```
# Getting Started Guide

Welcome to our platform. This manual covers the essential features.

## Account Setup
1. Create an account using your email and a strong password.
2. Verify your email by clicking the link in the verification email.
3. Set up two-factor authentication in Security Settings.

## Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one number
- At least one special character

## Troubleshooting Login Issues
If you cannot log in:
- Make sure caps lock is off.
- Try resetting your password using the "Forgot Password" link.
- Clear your browser cache and cookies.
- Ensure you are using a supported browser: Chrome, Firefox, Safari, Edge.

## Billing
All payment methods are managed in the Billing section. You can:
- Add a new credit card
- Update existing payment methods
- View past invoices
- Change your billing cycle

## Support Hours
Monday to Friday, 9 AM to 6 PM (EST). Weekend support is limited.
```

### Step 5: Create the Ingestion Script

Create `ingest.py`:

```python
"""
Document ingestion script.
Loads documents, splits into chunks, embeds them, and saves to JSON.
"""
import json
import os
from dotenv import load_dotenv
import openai

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks
DOCS_DIR = "./my_documents"
OUTPUT_FILE = "knowledge_base.json"

# --- Functions ---
def chunk_text(text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks of approximately max_chars with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            # Try to break at a sentence boundary
            next_period = text.rfind('.', start, end)
            if next_period > start + max_chars // 2:
                end = next_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def embed_chunks(chunks):
    """Embed a list of text chunks using OpenAI."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks
    )
    return [r.embedding for r in response.data]


def ingest_documents():
    """Load all documents from DOCS_DIR, chunk, embed, and save."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set. Check your .env file.")

    knowledge_base = {}

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.startswith('.'):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        print(f"Processing: {filename}")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        knowledge_base[filename] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "num_chunks": len(chunks)
        }
        print(f"  -> {len(chunks)} chunks embedded")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2)

    total_chunks = sum(v["num_chunks"] for v in knowledge_base.values())
    print(f"\nIngestion complete: {len(knowledge_base)} documents, {total_chunks} chunks")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_documents()
```

### Step 6: Create the Search Script

Create `search.py`:

```python
"""
Semantic search script.
Embeds a query, finds top-K similar chunks, and asks an LLM to answer.
"""
import json
import os
import numpy as np
from dotenv import load_dotenv
import openai

load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
TOP_K = 3

# --- Functions ---
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_query(query):
    """Embed a query string."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    return response.data[0].embedding


def similarity_search(query, top_k=TOP_K):
    """Find top-K most similar document chunks to the query."""
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    query_embedding = embed_query(query)
    results = []

    for doc_name, doc_data in kb.items():
        for i, chunk_embedding in enumerate(doc_data["embeddings"]):
            score = cosine_similarity(query_embedding, chunk_embedding)
            results.append({
                "doc": doc_name,
                "chunk": doc_data["chunks"][i],
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def generate_answer(query, context_results):
    """Generate an answer using the LLM with retrieved context."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = "\n\n".join([
        f"[Source: {r['doc']}]\n{r['chunk']}"
        for r in context_results
    ])

    prompt = f"""Use only the context below to answer the question. If the context does not contain enough information, respond with: "I don't have enough information to answer this question."

Context:
{context}

Question: {query}
Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def search_and_answer(query):
    """Main function: search + answer."""
    print(f"Query: {query}\n")
    print("Searching...\n")

    results = similarity_search(query)
    answer = generate_answer(query, results)

    print("--- Results ---\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['doc']}] (score: {r['score']:.4f})")
        print(f"     {r['chunk'][:100]}...\n")

    print(f"--- Answer ---\n{answer}\n")
    return answer, results


if __name__ == "__main__":
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print("Error: Run ingest.py first to build the knowledge base.")
        exit(1)

    query = input("Enter your question: ").strip()
    if not query:
        print("No query entered.")
        exit(0)

    search_and_answer(query)
```

### Step 7: Create the Streamlit UI (Optional)

Create `app.py`:

```python
"""
Streamlit web UI for semantic search.
Run with: streamlit run app.py
"""
import streamlit as st
import os
from search import search_and_answer

st.set_page_config(
    page_title="Semantic Document Search",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Semantic Document Search")
st.markdown("*Search documents by meaning, not just keywords.*")

# Check if knowledge base exists
if not os.path.exists("knowledge_base.json"):
    st.warning("Run `python ingest.py` first to build the knowledge base.")
    st.stop()

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., How do I reset my password?",
    label_visibility="collapsed"
)

# Search button
if st.button("Search", type="primary", use_container_width=True):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and answering..."):
            answer, results = search_and_answer(query)

        st.markdown("---")
        st.subheader("Answer")
        st.markdown(answer)

        st.markdown("---")
        st.subheader("Top Sources")
        for i, r in enumerate(results, 1):
            with st.expander(f"{i}. [{r['doc']}] (Score: {r['score']:.3f})"):
                st.markdown(f"_{r['chunk']}_")

st.markdown("---")
st.caption(
    f"Uses {EMBEDDING_MODEL} for embeddings and {LLM_MODEL} for answering."
)
```

### Step 8: Run the System

```bash
# 1. Ingest documents (creates knowledge_base.json)
python ingest.py

# 2. Test search (CLI)
python search.py

# 3. Start web UI
streamlit run app.py
```

---

## 5. How It Works (Plain Explanation)

### Embedding
An **embedding model** converts text (words, sentences, documents) into a list of numbers (a vector). Similar texts produce similar vectors. For example:

```
"reset password" → [0.2, -0.5, 0.8, ...]
"recover account" → [0.3, -0.4, 0.7, ...]
```
These vectors are close together (high cosine similarity) because they mean similar things.

### Similarity Search
When you ask a question, the same embedding model converts your question into a vector. The system then computes the **cosine similarity** between the query vector and every chunk vector in the knowledge base. The top-K most similar chunks are returned.

### LLM Answer Generation
The retrieved chunks (the "context") plus your question are sent to an LLM (`gpt-4o-mini`). The LLM reads the context and generates a natural-language answer grounded in the retrieved text.

---

## 6. Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (384-dim, fast, cheap) |
| `LLM_MODEL` | `gpt-4o-mini` | Answer-generation model (fast, affordable) |
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between adjacent chunks |
| `TOP_K` | 3 | Number of top results to retrieve |

### Cost Estimates (USD)
| Action | Cost |
|--------|------|
| Embed 1000 chunks (ingest) | ~$0.02 |
| Embed 1 query | ~$0.00002 |
| LLM answer (500 tokens) | ~$0.0003 |
| **100 searches** | ~$0.03 |

---

## 7. Testing the System

### Expected Test Cases

1. **Direct match** — Query: "How do I reset my password?"
   - Should find the FAQ entry about password reset.
   - Answer should include the reset link steps.

2. **Semantic match** — Query: "I forgot my login credentials"
   - Should find password reset and login troubleshooting docs.
   - Answer should redirect to password reset instructions.

3. **Cross-doc inference** — Query: "Are refunds available?"
   - Should find refund policy in FAQ.
   - Answer should state the 30-day refund window.

4. **Out of scope** — Query: "What is the meaning of life?"
   - Should retrieve some documents but answer with "I don't have enough information."

### Running Tests
```bash
# Test with built-in queries
python search.py
# Try these queries:
# 1. How do I reset my password?
# 2. What is the refund policy?
# 3. How do I contact support?
# 4. Can I change my subscription?
# 5. Where are my invoices?
# 6. What are the password requirements?
# 7. How do I enable two-factor authentication?
```

---

## 8. Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY not set` | Check `.env` file exists and key is correct |
| `knowledge_base.json not found` | Run `python ingest.py` first |
| Very slow responses | Reduce `TOP_K` or use a faster LLM model |
| Poor answer quality | Increase `CHUNK_SIZE` or decrease `CHUNK_OVERLAP` |
| High costs | Switch to local embeddings (Sentence-BERT) |

### Local Development (No API Costs)

To use local embeddings instead of OpenAI:

```bash
pip install sentence-transformers
```

Modify `ingest.py` and `search.py`:
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
# Replace embed_chunks() with:
# embeddings = model.encode(chunks).tolist()
```

---

## 9. Next Steps (Level-Up Path)

Once the simple workflow works, level up in this order:

### Level 1: Better Search Quality
- Add **reranking** (Cohere rerank or cross-encoder)
- Increase `TOP_K` to 5-10, then rerank to 2-3

### Level 2: Scale Beyond Memory
- Replace JSON storage with **Chroma** or **Qdrant** vector DB
- Supports millions of documents, persistent storage, faster search

### Level 3: Hybrid Search
- Combine semantic (vector) + keyword (BM25) search
- Use libraries like `rank-bm25` or built-in Chroma hybrid search

### Level 4: Multi-User & Production
- Add authentication
- Deploy with FastAPI backend + React frontend
- Use Redis for query caching
- Add monitoring (latency, cost tracking)

---

## 10. File Summary

| File | Purpose |
|------|---------|
| `.env` | Stores API key |
| `requirements.txt` | Python dependencies |
| `ingest.py` | Loads docs, chunks, embeds, saves index |
| `search.py` | Embeds queries, finds similar chunks, generates answers |
| `app.py` | Streamlit web interface |
| `knowledge_base.json` | Auto-generated document index |
| `my_documents/` | Your documents go here |
| `README.md` | This guide |

---

## Quick Start Checklist

- [ ] Create project directory
- [ ] Set up Python virtualenv
- [ ] Create `.env` with OpenAI API key
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Add documents to `my_documents/`
- [ ] Run `python ingest.py`
- [ ] Run `python search.py` and ask a question
- [ ] (Optional) Run `streamlit run app.py` for web UI

# AGENTS.md - Semantic Search AI Project

## Project Overview
A simple LLM + embedding pipeline for semantic document search. No vector DB required.

## Quick Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
# Step 1: Ingest documents
python ingest.py

# Step 2: Search (CLI)
python search.py

# Step 3: Web UI
streamlit run app.py
```

### Test Queries
```
How do I reset my password?
What is the refund policy?
How do I contact support?
Can I change my subscription?
Where do I find my invoice?
What are the password requirements?
How do I enable two-factor authentication?
```

## Configuration
Edit `.env` for:
- `OPENAI_API_KEY` — your OpenAI API key

## Project Structure
- `ingest.py` — Document loading, chunking, embedding, index creation
- `search.py` — Query embedding, similarity search, LLM answer generation
- `app.py` — Streamlit web UI
- `my_documents/` — Input documents (txt, md, pdf support via langchain)
- `knowledge_base.json` — Generated embedding index

## Dependencies
- openai — Embedding + LLM API
- numpy — Cosine similarity computation
- python-dotenv — Environment variable loading
- streamlit — Web UI

## Model Costs (estimates)
- Embedding: text-embedding-3-small = $0.00002 per call
- LLM: gpt-4o-mini = ~$0.0003 per 500-token answer

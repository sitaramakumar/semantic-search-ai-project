"""
Document ingestion script.
Loads documents, splits into chunks, embeds them, and saves to JSON.
"""
import json
import os
from dotenv import load_dotenv
import openai

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DOCS_DIR = "./my_documents"
OUTPUT_FILE = "knowledge_base.json"


def chunk_text(text, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
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
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks
    )
    return [r.embedding for r in response.data]


def ingest_documents():
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

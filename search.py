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

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
TOP_K = 3


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_query(query):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    return response.data[0].embedding


def similarity_search(query, top_k=TOP_K):
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

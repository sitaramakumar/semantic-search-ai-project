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
SIMILARITY_THRESHOLD = 0.75
DEFAULT_ALLOWED_ROLES = ["READ", "WRITE", "ADMIN"]  # matches ingest.py's fail-open default


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


def similarity_search(query, caller_role, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """
    Role filtering happens BEFORE any similarity math runs, not after: a
    document whose allowed_roles doesn't include caller_role is skipped
    entirely in the loop below, so its chunks are never scored, never ranked,
    and can never end up in context sent to the LLM. This project has no
    vector DB (see README - that's a deliberate choice, not an oversight), so
    there's no Chroma/Qdrant `where` clause to hand this to; the equivalent
    here is simply never computing cosine_similarity for a chunk whose
    document isn't in scope for this caller.

    A near-miss below `threshold` and a real match hidden by the role filter
    produce the exact same empty result, on purpose - telling a low-privilege
    caller "something matched but you can't see it" would leak the existence
    of restricted content through the shape of the response, not just its
    substance.
    """
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    query_embedding = embed_query(query)
    return rank_and_filter(kb, query_embedding, caller_role, top_k, threshold)


def rank_and_filter(kb, query_embedding, caller_role, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """
    The pure decision logic behind similarity_search, split out so it's
    unit-testable with synthetic embeddings instead of a live OpenAI call -
    same "separate the pure decision from the network/IO dependency" pattern
    as DocumentAccessGuard in the interview-poc-sharepoint/-rust services.
    """
    results = []

    for doc_name, doc_data in kb.items():
        allowed_roles = doc_data.get("allowed_roles", DEFAULT_ALLOWED_ROLES)
        if caller_role not in allowed_roles:
            continue
        for i, chunk_embedding in enumerate(doc_data["embeddings"]):
            score = cosine_similarity(query_embedding, chunk_embedding)
            results.append({
                "doc": doc_name,
                "chunk": doc_data["chunks"][i],
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    # Drop every chunk below threshold, not just gate on the best one -
    # otherwise a strong top match could still drag a barely-related chunk
    # into the LLM's context alongside it.
    return [r for r in results if r["score"] >= threshold][:top_k]


NO_RESULTS_MESSAGE = "No relevant documents found for your query."


def generate_answer(query, context_results):
    if not context_results:
        # No LLM call at all: nothing relevant (or nothing this caller's role
        # can see) means there's no context worth spending a paid completion
        # call on - real token/cost savings, distinct from but complementary
        # to the role filter skipping similarity math above.
        return NO_RESULTS_MESSAGE

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


def search_and_answer(query, caller_role):
    print(f"Query: {query} (role: {caller_role})\n")
    print("Searching...\n")

    results = similarity_search(query, caller_role)
    answer = generate_answer(query, results)

    print("--- Results ---\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['doc']}] (score: {r['score']:.4f})")
        print(f"     {r['chunk'][:100]}...\n")

    print(f"--- Answer ---\n{answer}\n")
    return answer, results


if __name__ == "__main__":
    import roles

    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print("Error: Run ingest.py first to build the knowledge base.")
        exit(1)

    token = input("Paste a Bearer token (blank to generate a demo one): ").strip()
    if not token:
        demo_role = input("Demo role [READ/WRITE/ADMIN]: ").strip().upper() or "READ"
        token = roles.generate_demo_token("cli-demo-user", demo_role)
        print(f"Generated demo token for role {demo_role}: {token}\n")

    try:
        role = roles.role_from_token(token)
    except roles.TokenError as e:
        print(f"Access denied: {e}")
        exit(1)

    query = input("Enter your question: ").strip()
    if not query:
        print("No query entered.")
        exit(0)

    search_and_answer(query, role)

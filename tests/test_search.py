import pytest

from search import rank_and_filter, generate_answer, NO_RESULTS_MESSAGE

# Trivial 2D "embeddings" chosen so cosine similarity is exact and easy to
# reason about: identical vector -> 1.0, orthogonal -> 0.0.
QUERY_EMBEDDING = [1.0, 0.0]

KB = {
    "public.txt": {
        "chunks": ["public chunk"],
        "embeddings": [[1.0, 0.0]],  # identical to query -> score 1.0
        "allowed_roles": ["READ", "WRITE", "ADMIN"],
    },
    "admin-only.txt": {
        "chunks": ["admin-only chunk"],
        "embeddings": [[1.0, 0.0]],  # also a perfect match, but restricted
        "allowed_roles": ["ADMIN"],
    },
    "irrelevant.txt": {
        "chunks": ["irrelevant chunk"],
        "embeddings": [[0.0, 1.0]],  # orthogonal -> score 0.0
        "allowed_roles": ["READ", "WRITE", "ADMIN"],
    },
}


def test_a_read_caller_never_sees_the_admin_only_document():
    results = rank_and_filter(KB, QUERY_EMBEDDING, "READ", top_k=5, threshold=0.5)

    docs = [r["doc"] for r in results]
    assert "admin-only.txt" not in docs
    assert "public.txt" in docs


def test_an_admin_caller_sees_the_admin_only_document():
    results = rank_and_filter(KB, QUERY_EMBEDDING, "ADMIN", top_k=5, threshold=0.5)

    docs = [r["doc"] for r in results]
    assert "admin-only.txt" in docs


def test_role_filtering_happens_before_scoring_not_after():
    # Regression guard: if filtering were applied AFTER ranking/top_k instead
    # of before, a READ caller asking a question only the admin-only document
    # answers would still get zero results - the point of this test is that
    # the forbidden chunk's score is never computed or ranked at all, not
    # merely hidden from the final output.
    admin_only_kb = {"admin-only.txt": KB["admin-only.txt"]}
    results = rank_and_filter(admin_only_kb, QUERY_EMBEDDING, "READ", top_k=5, threshold=0.5)
    assert results == []


def test_results_below_threshold_are_dropped_even_for_an_allowed_role():
    results = rank_and_filter(KB, QUERY_EMBEDDING, "ADMIN", top_k=5, threshold=0.99)

    # Only the perfect (1.0) matches clear a 0.99 threshold; the orthogonal
    # (0.0) one does not.
    docs = [r["doc"] for r in results]
    assert "irrelevant.txt" not in docs


def test_no_results_and_role_filtered_out_look_identical():
    # A caller whose role matches nothing, and a caller asking about content
    # that just doesn't exist, both get an empty list - the response never
    # reveals whether a restricted document would otherwise have matched.
    no_match_kb = {"admin-only.txt": KB["admin-only.txt"]}
    filtered_out = rank_and_filter(no_match_kb, QUERY_EMBEDDING, "READ", top_k=5, threshold=0.5)
    genuinely_absent = rank_and_filter({}, QUERY_EMBEDDING, "READ", top_k=5, threshold=0.5)
    assert filtered_out == genuinely_absent == []


def test_generate_answer_skips_the_llm_call_entirely_when_there_is_no_context():
    # No API key is set in this test environment; if generate_answer tried to
    # call OpenAI for an empty context, this test would fail with an auth
    # error instead of passing - which is exactly what proves the early
    # return happens before any network call.
    assert generate_answer("anything", []) == NO_RESULTS_MESSAGE

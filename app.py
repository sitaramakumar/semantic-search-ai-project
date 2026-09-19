"""
Streamlit web UI for semantic search.
Run with: streamlit run app.py
"""
import streamlit as st
import os
from search import search_and_answer, EMBEDDING_MODEL, LLM_MODEL

st.set_page_config(
    page_title="Semantic Document Search",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Semantic Document Search")
st.markdown("*Search documents by meaning, not just keywords.*")

if not os.path.exists("knowledge_base.json"):
    st.warning("Run `python ingest.py` first to build the knowledge base.")
    st.stop()

query = st.text_input(
    "Enter your question:",
    placeholder="e.g., How do I reset my password?",
    label_visibility="collapsed"
)

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

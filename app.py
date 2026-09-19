"""
Streamlit web UI for semantic search.
Run with: streamlit run app.py
"""
import streamlit as st
import os
from search import search_and_answer, EMBEDDING_MODEL, LLM_MODEL
import roles

st.set_page_config(
    page_title="Semantic Document Search",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Semantic Document Search")
st.markdown("*Search documents by meaning, not just keywords — scoped to your role.*")

if not os.path.exists("knowledge_base.json"):
    st.warning("Run `python ingest.py` first to build the knowledge base.")
    st.stop()

with st.sidebar:
    st.subheader("Access token")
    st.caption("Paste a Bearer token from interview-poc-sharepoint/-rust's "
               "/api/tokens, or generate a demo one below (same JWT_SECRET, "
               "same claim shape — decoded by roles.py, not faked).")
    demo_role = st.selectbox("Generate a demo token for role", ["READ", "WRITE", "ADMIN"])
    if st.button("Generate demo token"):
        try:
            st.session_state["token"] = roles.generate_demo_token("streamlit-demo-user", demo_role)
        except roles.TokenError as e:
            st.error(f"Cannot generate token: {e}")
    token = st.text_area("Bearer token", value=st.session_state.get("token", ""), height=100)

query = st.text_input(
    "Enter your question:",
    placeholder="e.g., How do I reset my password?",
    label_visibility="collapsed"
)

if st.button("Search", type="primary", use_container_width=True):
    if not query:
        st.warning("Please enter a question.")
    else:
        try:
            caller_role = roles.role_from_token(token)
        except roles.TokenError as e:
            st.error(f"403 Forbidden — access denied: {e}")
            st.stop()

        st.caption(f"Authenticated as role: **{caller_role}**")

        with st.spinner("Searching and answering..."):
            answer, results = search_and_answer(query, caller_role)

        st.markdown("---")
        st.subheader("Answer")
        st.markdown(answer)

        if results:
            st.markdown("---")
            st.subheader("Top Sources")
            for i, r in enumerate(results, 1):
                with st.expander(f"{i}. [{r['doc']}] (Score: {r['score']:.3f})"):
                    st.markdown(f"_{r['chunk']}_")

st.markdown("---")
st.caption(
    f"Uses {EMBEDDING_MODEL} for embeddings and {LLM_MODEL} for answering."
)

# Streamlit UI — the complete face of DocuMind AI
# RUN : streamlit run app.py

import os
import gc
import shutil
import streamlit as st
from document_loader import load_pdf
from chunker import chunk_documents
from embedder import load_embedding_model
from vector_store import save_to_vectorstore
from retriever import get_retriever
from qa_chain import build_qa_chain, get_answer

# Page settings — must be first Streamlit command
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="centered"
)

# Custom styling to make it look like a real product
st.markdown("""
<style>
.stApp { background-color: #0d1117 !important; }
header { visibility: hidden; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f2e 0%, #1a0a2e 100%) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #ffffff !important; }
[data-testid="stFileUploader"] > div {
    background-color: #1a1a3e !important;
    border: 2px dashed #ec4899 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small { color: #ffffff !important; }
[data-testid="stFileUploader"] button {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    border: 1px solid #ec4899 !important;
    border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #1e3a8a 0%, #9333ea 50%, #ec4899 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}
.answer-box {
    background-color: #111827 !important;
    border-left: 4px solid #ec4899 !important;
    padding: 16px 20px !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #f1f5f9 !important;
    box-shadow: 0 2px 12px rgba(236,72,153,0.2) !important;
}
.source-badge {
    display: inline-block !important;
    background-color: #1e1b4b !important;
    color: #f9a8d4 !important;
    border: 1px solid #ec4899 !important;
    padding: 3px 12px !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    margin-right: 5px !important;
    margin-top: 8px !important;
}
.stChatInput input {
    background-color: #1a2332 !important;
    color: #ffffff !important;
    border: 1px solid #ec4899 !important;
    border-radius: 12px !important;
}
hr { border-color: #2d1b4e !important; }
</style>
""", unsafe_allow_html=True)

# Session state keeps data alive between Streamlit reruns
# Without this, everything resets every time user clicks anything
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores all questions and answers

if "qa_ready" not in st.session_state:
    st.session_state.qa_ready = False   # becomes True after PDF is processed

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""      # stores the name of uploaded PDF


# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocuMind AI")
    st.markdown("Upload any PDF and ask questions about it.")
    st.divider()

    # Multiple PDF uploader — accepts more than one PDF at once
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,  # allows uploading multiple PDFs
        help="Upload one or more PDFs — research papers, reports, contracts."
    )

    # Show process button only after files are uploaded
    if uploaded_files:
        if st.button("🚀 Process Document(s)", use_container_width=True):
            with st.spinner("Reading and indexing your document(s)..."):
                try:
                    # Make sure the upload folder exists
                    os.makedirs("data/uploaded_docs", exist_ok=True)

                    # Save all uploaded files to disk
                    saved_paths = []
                    for file in uploaded_files:
                        save_path = f"data/uploaded_docs/{file.name}"
                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())
                        saved_paths.append(save_path)

                    # Release old ChromaDB connection properly
                    if st.session_state.retriever is not None:
                        try:
                            st.session_state.retriever.vectorstore._client = None
                        except Exception:
                            pass

                    # Clear everything from memory
                    st.session_state.qa_chain = None
                    st.session_state.retriever = None
                    gc.collect()

                    # Wait for Windows to release file locks
                    import time

                    time.sleep(1)

                    # Force delete vectorstore — try 3 times if needed
                    for attempt in range(3):
                        if os.path.exists("vectorstore"):
                            try:
                                shutil.rmtree("vectorstore")
                                log.info("Vectorstore deleted successfully")
                                break  # deleted successfully, stop trying
                            except Exception:
                                time.sleep(1)  # wait 1 more second and try again

                    # Load all pages from all uploaded PDFs
                    all_pages = []
                    for path in saved_paths:
                        pages = load_pdf(path)
                        all_pages.extend(pages)  # combine pages from all PDFs

                    # Chunk all pages together
                    chunks = chunk_documents(all_pages)

                    # Load embedding model
                    embeddings = load_embedding_model()

                    # Save all chunks to ChromaDB
                    vectorstore = save_to_vectorstore(chunks, embeddings)

                    # Create retriever
                    retriever = get_retriever(vectorstore)

                    # Build QA chain
                    qa_chain = build_qa_chain(retriever)

                    # Store everything in session state for later use
                    st.session_state.qa_chain = qa_chain
                    st.session_state.retriever = retriever
                    st.session_state.qa_ready = True
                    st.session_state.chat_history = []  # clear old chat
                    st.session_state.doc_name = ", ".join([f.name for f in uploaded_files])

                    st.success(f"✅ Ready! {len(all_pages)} pages indexed from {len(uploaded_files)} file(s).")

                except Exception as e:
                    # Show exact error so user knows what went wrong
                    st.error(f"❌ Something went wrong: {str(e)}")

    # Show which document is currently loaded
    if st.session_state.qa_ready:
        st.divider()
        st.markdown(f"📂 **Loaded:** {st.session_state.doc_name}")

        # Button to clear and start fresh
        if st.button("🗑️ Clear & Upload New", use_container_width=True):
            st.session_state.qa_ready = False
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.session_state.retriever = None
            st.session_state.doc_name = ""
            gc.collect()
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            st.rerun()  # refresh the page

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")
    st.caption("🎓 DocuMind AI — by Hema Malini Gangumalla")


# ── MAIN CHAT AREA ───────────────────────────────────────
st.markdown("## 💬 DocuMind AI")
st.caption("Upload a PDF on the left, then ask me anything about it.")
st.divider()

# Show all previous chat messages in order
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input only works after PDF is processed
if st.session_state.qa_ready:
    question = st.chat_input("Ask a question about your document...")

    if question:
        # Show user question as chat bubble immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Save user question to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Generate and show the answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = get_answer(
                        question,
                        st.session_state.qa_chain,
                        st.session_state.retriever
                    )

                    # Show answer in styled box
                    st.markdown(
                        f"<div class='answer-box'>{response['answer']}</div>",
                        unsafe_allow_html=True
                    )

                    # Show which pages the answer came from
                    if response["sources"]:
                        badges = "".join([
                            f"<span class='source-badge'>📄 Page {p}</span>"
                            for p in sorted(response["sources"])
                        ])
                        st.markdown(f"**Sources:** {badges}", unsafe_allow_html=True)

                    # Save answer to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })

                except Exception as e:
                    st.error(f"❌ Could not generate answer: {str(e)}")

else:
    # Show this when no PDF is uploaded yet
    st.info("👈 Upload a PDF from the sidebar to get started.")
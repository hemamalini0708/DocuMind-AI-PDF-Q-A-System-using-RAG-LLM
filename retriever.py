# Search ChromaDB and return most relevant chunks for a question

from logger import get_logger

log = get_logger("retriever")

# Number of chunks to retrieve per question
# 3 is the sweet spot — enough context, not too much noise
TOP_K = 3


def get_retriever(vectorstore, top_k: int = TOP_K):
    """
    Converts ChromaDB into a search tool called retriever.
    INPUT : vectorstore → ChromaDB object from vector_store.py
            top_k       → how many chunks to return (default 3)
    OUTPUT: retriever object
    """
    log.info(f"Creating retriever with top_k={top_k}")

    # as_retriever turns ChromaDB into a LangChain search tool
    # search_type="similarity" → finds chunks with most similar meaning to question
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    log.info("Retriever created successfully.")
    return retriever


def retrieve_chunks(question: str, retriever):
    """
    Takes a question and returns most relevant chunks from ChromaDB.
    INPUT : question  → user's question as plain English string
            retriever → retriever object from get_retriever()
    OUTPUT: list of relevant Document chunks
    """
    log.info(f"Searching for: {question[:80]}")

    # invoke() sends question to ChromaDB and returns top matching chunks
    relevant_chunks = retriever.invoke(question)

    log.info(f"Found {len(relevant_chunks)} relevant chunks.")

    # Log a short preview of each chunk found
    for i, chunk in enumerate(relevant_chunks, start=1):
        page = chunk.metadata.get("page", "?")
        preview = chunk.page_content[:100]
        log.debug(f"Chunk {i} | Page {page} | {preview}...")

    return relevant_chunks


# Run this block only when testing this file directly
if __name__ == "__main__":
    from document_loader import load_pdf
    from chunker import chunk_documents
    from embedder import load_embedding_model
    from vector_store import save_to_vectorstore

    pages = load_pdf("data/uploaded_docs/1706.03762v7.pdf")
    chunks = chunk_documents(pages)
    embeddings = load_embedding_model()
    vectorstore = save_to_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    results = retrieve_chunks("What is multi-head attention?", retriever)
    log.info(f"Test passed — {len(results)} chunks retrieved.")
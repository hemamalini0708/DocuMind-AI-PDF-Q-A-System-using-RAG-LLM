# chunk vectors into ChromaDB and load them back

import os
import shutil
from langchain_chroma import Chroma
from logger import get_logger

log = get_logger("vector_store")

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "documind_collection"


def save_to_vectorstore(chunks: list, embeddings):
    """
    Deletes old vectorstore and saves fresh chunks.
    Always builds from scratch — no loading old data.
    """
    # Force delete old vectorstore first — clean start every time
    if os.path.exists(VECTORSTORE_DIR):
        try:
            shutil.rmtree(VECTORSTORE_DIR)
            log.info("Old vectorstore deleted successfully.")
        except Exception as e:
            log.warning(f"Could not delete vectorstore folder: {e}")

    # Create fresh vectorstore from new chunks
    log.info(f"Saving {len(chunks)} chunks to ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR
    )

    # Verify it was created correctly
    if vectorstore is None:
        raise ValueError("ChromaDB failed to create vectorstore.")

    log.info(f"Vectorstore saved successfully at: {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore(embeddings):
    """
    Loads existing ChromaDB from disk.
    Only used when vectorstore already exists.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        log.error("Vectorstore not found on disk.")
        raise FileNotFoundError("No vectorstore found. Please process a PDF first.")

    log.info(f"Loading ChromaDB from: {VECTORSTORE_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIR
    )

    log.info("ChromaDB loaded successfully.")
    return vectorstore


# Run this block only when testing this file directly
if __name__ == "__main__":
    from document_loader import load_pdf
    from chunker import chunk_documents
    from embedder import load_embedding_model

    pages = load_pdf("data/uploaded_docs/1706.03762v7.pdf")
    chunks = chunk_documents(pages)
    embeddings = load_embedding_model()
    vectorstore = save_to_vectorstore(chunks, embeddings)

    log.info("Test passed!")
    log.info(f"Total chunks stored: {len(chunks)}")
#  Convert text into numbers (vectors) so computer can understand meaning

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
# HuggingFaceEmbeddings runs the embedding model locally on your computer
# No API key needed — completely free after first download

from logger import get_logger

# Load API keys from .env file
load_dotenv()

log = get_logger("embedder")

# This model converts text into 384 numbers (dimensions)
# Small, fast, free, and accurate enough for our RAG system
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Tell HuggingFace to use locally cached model — avoids internet timeout
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY", "")


def load_embedding_model():
    """
    Loads the embedding model from local cache.
    First time only: downloads the model (~90MB). After that: instant load.
    OUTPUT: embeddings → the loaded model object
    """
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # run on CPU, works on all computers
        encode_kwargs={"normalize_embeddings": True}  # makes comparisons more accurate
    )

    log.info("Embedding model loaded successfully.")
    return embeddings


# Run this block only when testing this file directly
if __name__ == "__main__":
    embeddings = load_embedding_model()

    test_query = "What is the Transformer model?"
    vector = embeddings.embed_query(test_query)

    log.info(f"Test passed!")
    log.info(f"Query: {test_query}")
    log.info(f"Vector size: {len(vector)} dimensions")
    log.debug(f"First 5 values: {vector[:20]}....")  # just first 5 is enough
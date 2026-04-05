"""
Connect retrieved chunks + question to Groq LLM and generate answer
"""

import os
from dotenv import load_dotenv  # reads our .env file for API keys

# ChatGroq → connects to Groq's free LLaMA 3 model
from langchain_groq import ChatGroq

# These three are the modern way to build chains in LangChain 2025
from langchain_core.prompts import PromptTemplate          # builds our prompt
from langchain_core.output_parsers import StrOutputParser  # extracts text from LLM response
from langchain_core.runnables import RunnablePassthrough   # passes question through pipeline

from logger import get_logger
load_dotenv()  # load GROQ_API_KEY from .env file
log = get_logger("qa_chain")

GROQ_MODEL = "llama-3.3-70b-versatile"  # free and powerful model on Groq

# ============================================================
# PROMPT ENGINEERING
# This tells the LLM exactly how to behave:
# 1. What role it plays (DocuMind AI assistant)
# 3. What to say if answer not found
# 4. How to format the answer
# ============================================================
# 2. Only answer from the given context — no outside knowledge
PROMPT_TEMPLATE = """
You are DocuMind AI, an intelligent document assistant.
Your job is to answer questions ONLY based on the context provided below.
Do NOT use any outside knowledge. If the answer is not in the context,
say exactly: "I could not find this information in the uploaded document."

Context from the document:
---------------------------------
{context}
---------------------------------

User Question: {question}

Instructions:
- Answer clearly and precisely
- Keep the answer concise but complete
- Do NOT make up any information

Answer:
"""


def format_docs(docs):
    # Takes list of chunk documents and joins them into one big string
    # This combined string becomes the {context} in our prompt
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever):
    """
    Builds the full RAG chain — retriever + prompt + LLM all connected.
    INPUT : retriever → object from retriever.py
    OUTPUT: qa_chain → ready to answer any question
    """
    log.info("Building QA chain...")

    # Read Groq API key from .env file
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        log.error("GROQ_API_KEY not found in .env file!")
        raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

    # Create the LLM — this connects to Groq free API
    # temperature=0.2 → low randomness = more factual answers
    # max_tokens=1024 → max length of generated answer
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
        api_key=groq_api_key
    )
    log.info(f"LLM loaded: {GROQ_MODEL}")

    # Create prompt from our template above
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Build the chain using | (pipe) operator — modern LangChain style
    # How it works step by step:
    # 1. retriever searches ChromaDB and returns relevant chunks
    # 2. format_docs joins all chunks into one string → fills {context}
    # 3. RunnablePassthrough passes the question as-is → fills {question}
    # 4. prompt combines context + question into final prompt text
    # 5. llm reads the prompt and generates an answer
    # 6. StrOutputParser extracts plain text from the LLM response object
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    log.info("QA chain built successfully.")
    return qa_chain


def get_answer(question: str, qa_chain, retriever):
    """
    Runs a question through the full RAG pipeline.
    INPUT : question  → user's plain English question
            qa_chain  → built by build_qa_chain()
            retriever → needed to get source page numbers
    OUTPUT: dict with 'answer' and 'sources'
    """
    log.info(f"Processing question: {question}")

    # Run the full chain — returns the answer as a plain string
    answer = qa_chain.invoke(question)

    # Get the source chunks separately to extract page numbers
    source_docs = retriever.invoke(question)

    # Extract page numbers from chunk metadata
    # metadata["page"] is 0-indexed so we add +1 for human-readable pages
    sources = list(set([doc.metadata.get("page", 0) + 1 for doc in source_docs]))

    log.info("Answer generated successfully.")
    log.debug(f"Answer preview: {answer[:]}...")
    log.info(f"Source pages: {sources}")

    return {
        "answer": answer,
        "sources": sources
    }


# ── TEST ──────────────────────────────────────────────────
if __name__ == "__main__":
    from document_loader import load_pdf
    from chunker import chunk_documents
    from embedder import load_embedding_model
    from vector_store import save_to_vectorstore
    from retriever import get_retriever

    # Step 1: Load PDF
    pages = load_pdf("data/uploaded_docs/1706.03762v7.pdf")

    # Step 2: Chunk it
    chunks = chunk_documents(pages)

    # Step 3: Load embedding model
    embeddings = load_embedding_model()

    # Step 4: Save to ChromaDB
    vectorstore = save_to_vectorstore(chunks, embeddings)

    # Step 5: Create retriever
    retriever = get_retriever(vectorstore)

    # Step 6: Build QA chain
    qa_chain = build_qa_chain(retriever)

    # Step 7: Ask a question
    question = "What is the purpose of multi-head attention?"
    response = get_answer(question, qa_chain, retriever)

    log.info(f"Question : {question}")
    log.info(f"Answer   : {response['answer']}")
    log.info(f"Sources  : Page {response['sources']}")
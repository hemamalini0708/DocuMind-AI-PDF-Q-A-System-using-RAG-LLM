# Split large PDF pages into smaller overlapping pieces called chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter splits text smartly
# It tries paragraph first, then sentence, then word
# This keeps meaning intact and never cuts mid-sentence

from logger import get_logger

log = get_logger("chunker")


def chunk_documents(pages: list, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    INPUT : pages        → list of pages from document_loader.py
            chunk_size   → max characters per chunk (1000 = about 150 words)
            chunk_overlap → characters repeated between chunks (keeps context)
    OUTPUT: chunks → a bigger list of smaller Document pieces
    """
    log.info(f"Starting chunking → chunk_size={chunk_size}, overlap={chunk_overlap}")

    # Create the splitter with our settings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len  # measure size using number of characters
    )

    # Split all pages into chunks
    chunks = splitter.split_documents(pages)

    log.info(f"Chunking done → {len(pages)} pages → {len(chunks)} chunks")

    # Show a small preview of the first chunk to verify
    if chunks:
        log.debug(f"First chunk preview: {chunks[0].page_content[:150]}...")

    return chunks


# Run this block only when testing this file directly
if __name__ == "__main__":
    from document_loader import load_pdf

    pages = load_pdf("data/uploaded_docs/1706.03762v7.pdf")
    chunks = chunk_documents(pages)
    log.info(f"Sample metadata: {chunks[0].metadata}")
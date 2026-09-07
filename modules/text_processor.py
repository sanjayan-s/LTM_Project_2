import re


def clean_text(text):
    """
    Clean extracted PDF text before chunking.
    """

    if not text:
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks.

    Chunks are created using a sliding window while
    preserving natural sentence/paragraph boundaries
    whenever possible.
    """

    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    text = text.strip()

    # If the entire document fits inside one chunk,
    # return it directly.
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Try to find a natural boundary only when
        # enough text has already been included.
        if end < text_length:

            minimum_boundary = start + int(chunk_size * 0.5)

            paragraph_break = text.rfind(
                "\n\n",
                minimum_boundary,
                end
            )

            sentence_breaks = [
                text.rfind(". ", minimum_boundary, end),
                text.rfind("? ", minimum_boundary, end),
                text.rfind("! ", minimum_boundary, end)
            ]

            sentence_break = max(sentence_breaks)

            if paragraph_break != -1:
                end = paragraph_break

            elif sentence_break != -1:
                end = sentence_break + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        next_start = end - chunk_overlap

        # Safety check to guarantee forward movement
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

if __name__ == "__main__":

    sample_text = """
    Knowledge Guardian Lite is an AI-powered document intelligence system.

    It allows users to upload organizational documents and ask questions
    about their contents.

    The system uses semantic search and retrieval augmented generation.
    """

    cleaned = clean_text(sample_text)

    chunks = chunk_text(
        cleaned,
        chunk_size=1000,
        chunk_overlap=200
    )

    print("Cleaned Text:")
    print(cleaned)

    print("\nChunks:")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)
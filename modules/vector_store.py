import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Load environment variables
load_dotenv()


# -----------------------------
# Configuration
# -----------------------------

VECTOR_DB_PATH = "vector_db"
COLLECTION_NAME = "knowledge_guardian"


# -----------------------------
# Gemini Embedding Model
# -----------------------------

def get_embedding_model():
    """
    Create and return the Gemini embedding model.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Please check your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )


# -----------------------------
# ChromaDB
# -----------------------------

def get_vector_store():
    """
    Create or load the persistent ChromaDB vector store.
    """

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    return vector_store


# -----------------------------
# Index Document
# -----------------------------

def add_document_to_vector_store(chunks, source):
    """
    Add document chunks to ChromaDB.

    Existing chunks belonging to the same source
    are removed first to prevent duplicate indexing.
    """

    if not chunks:
        return 0

    vector_store = get_vector_store()

    # Remove existing chunks for this document
    try:
        existing = vector_store.get(
            where={"source": source}
        )

        existing_ids = existing.get("ids", [])

        if existing_ids:
            vector_store.delete(
                ids=existing_ids
            )

    except Exception:
        # Continue if the document has not been indexed before
        pass

    documents = []
    ids = []

    for index, chunk in enumerate(chunks):

        document = Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_id": index
            }
        )

        documents.append(document)

        ids.append(
            f"{source}_{index}"
        )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return len(documents)


# -----------------------------
# Semantic Search
# -----------------------------

def search_documents(query, k=5, source=None):
    """
    Perform semantic similarity search.

    If source is provided, search only within
    that document.
    """

    vector_store = get_vector_store()

    if source:

        results = vector_store.similarity_search(
            query,
            k=k,
            filter={"source": source}
        )

    else:

        results = vector_store.similarity_search(
            query,
            k=k
        )

    return results


# -----------------------------
# Check Document Status
# -----------------------------

def is_document_indexed(source):
    """
    Check whether a document already exists
    in ChromaDB.
    """

    vector_store = get_vector_store()

    try:
        existing = vector_store.get(
            where={"source": source}
        )

        ids = existing.get("ids", [])

        return len(ids) > 0

    except Exception:
        return False


# -----------------------------
# Get Chunk Count
# -----------------------------

def get_document_chunk_count(source):
    """
    Return the number of indexed chunks
    for a specific document.
    """

    vector_store = get_vector_store()

    try:
        existing = vector_store.get(
            where={"source": source}
        )

        ids = existing.get("ids", [])

        return len(ids)

    except Exception:
        return 0
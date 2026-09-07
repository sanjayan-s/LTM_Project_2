import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from modules.vector_store import search_documents


load_dotenv()


def get_llm():
    """
    Initialize the Gemini chat model.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Please check your .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key
    )


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Your task is to answer the user's question ONLY using
the provided document context.

STRICT RULES:

1. Use only the information present in the context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the answer cannot be found in the context,
   clearly say:

   "I could not find this information in the selected document."

5. Give a concise and direct answer.
6. When possible, mention the relevant evidence from the document.

DOCUMENT CONTEXT:
{context}
"""
        ),
        (
            "human",
            """
Question:
{question}

Answer based only on the document context above.
"""
        )
    ]
)


def format_context(documents):
    """
    Convert retrieved LangChain documents
    into a clean context string.
    """

    if not documents:
        return ""

    context_parts = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        chunk_id = document.metadata.get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"""
[Document Evidence {index}]
Source: {source}
Section: {chunk_id}

{document.page_content}
"""
        )

    return "\n".join(context_parts)


def ask_question(
    question,
    source,
    k=5
):
    """
    Answer a question using RAG.

    Retrieval is restricted to the selected document.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not source:
        raise ValueError(
            "A source document must be selected."
        )

    # Retrieve relevant chunks only
    # from the selected document.
    documents = search_documents(
        question,
        k=k,
        source=source
    )

    if not documents:
        return {
            "answer": (
                "I could not find this information "
                "in the selected document."
            ),
            "sources": []
        }

    context = format_context(
        documents
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question
        }
    )

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    answer = response.text

    sources = []

    for document in documents:

        source_info = {
            "source": document.metadata.get(
                "source"
            ),
            "chunk_id": document.metadata.get(
                "chunk_id"
            )
        }

        if source_info not in sources:
            sources.append(
                source_info
            )

    return {
        "answer": answer,
        "sources": sources
    }
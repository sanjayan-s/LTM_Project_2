import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from modules.text_processor import chunk_text


load_dotenv()


def get_llm():
    """
    Initialize the Gemini model for document summarization.
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


CHUNK_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Summarize the provided section of a document.

Rules:
1. Use only information contained in the section.
2. Do not add outside knowledge.
3. Do not invent facts.
4. Preserve important technical details.
5. Remove unnecessary repetition.
6. Keep the summary concise.

SECTION:
{section}
"""
        ),
        (
            "human",
            "Summarize this document section."
        )
    ]
)


FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are Knowledge Guardian, an AI document intelligence assistant.

Create a clear executive summary using the provided
section summaries.

Rules:
1. Use only information contained in the summaries.
2. Do not introduce outside knowledge.
3. Do not invent facts.
4. Combine repeated information.
5. Preserve important technical details.
6. Keep the final summary concise and useful.

Use the following structure when applicable:

Overview
Key Points
Important Technical Details
Risks or Concerns
Conclusion

Do not invent information for sections that are
not supported by the document.

SECTION SUMMARIES:
{summaries}
"""
        ),
        (
            "human",
            "Create the final executive summary."
        )
    ]
)


def summarize_section(section):
    """
    Summarize one section of a document.
    """

    llm = get_llm()

    prompt = CHUNK_SUMMARY_PROMPT.invoke(
        {
            "section": section
        }
    )

    response = llm.invoke(prompt)

    return response.text


def combine_summaries(summaries):
    """
    Combine intermediate summaries into
    one final executive summary.
    """

    if not summaries:
        return ""

    combined_text = "\n\n".join(
        f"Section Summary {index + 1}:\n{summary}"
        for index, summary in enumerate(summaries)
    )

    llm = get_llm()

    prompt = FINAL_SUMMARY_PROMPT.invoke(
        {
            "summaries": combined_text
        }
    )

    response = llm.invoke(prompt)

    return response.text


def summarize_document(
    text,
    chunk_size=6000,
    chunk_overlap=500
):
    """
    Generate an executive summary for a document.

    Large documents are processed using a
    map-reduce summarization strategy.
    """

    if not text or not text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    cleaned_text = text.strip()

    sections = chunk_text(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    if not sections:
        raise ValueError(
            "Unable to create document sections."
        )

    # Small document
    if len(sections) == 1:

        return summarize_section(
            sections[0]
        )

    # Large document
    summaries = []

    for section in sections:

        summary = summarize_section(
            section
        )

        summaries.append(summary)

    return combine_summaries(
        summaries
    )
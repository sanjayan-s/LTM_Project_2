import os

import streamlit as st

from modules.auth import login, logout
from modules.document_manager import upload_document
from modules.pdf_loader import extract_text_from_pdf
from modules.text_processor import clean_text, chunk_text

from modules.vector_store import (
    add_document_to_vector_store,
    is_document_indexed,
    get_document_chunk_count
)

from modules.rag_engine import ask_question
from modules.summarizer import summarize_document


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_source" not in st.session_state:
    st.session_state.summary_source = None


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.logged_in:

    login()

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

st.title("📘 Knowledge Guardian Lite")

st.write(
    f"Welcome, **{st.session_state.username}**"
)

logout()

st.divider()


# ============================================================
# 1. UPLOAD DOCUMENT
# ============================================================

st.header("📄 Upload Document")

upload_document()

st.divider()


# ============================================================
# 2. MY DOCUMENTS
# ============================================================

st.header("📁 My Documents")

uploaded_files = [
    file
    for file in os.listdir(UPLOAD_FOLDER)
    if file.lower().endswith(".pdf")
]


if not uploaded_files:

    st.info(
        "No documents uploaded yet. "
        "Upload a PDF to get started."
    )

    st.stop()


# ------------------------------------------------------------
# Document selector
# ------------------------------------------------------------

selected_file = st.selectbox(
    "Select a document",
    uploaded_files
)


# ------------------------------------------------------------
# Clear old summary when document changes
# ------------------------------------------------------------

if st.session_state.summary_source != selected_file:

    st.session_state.summary = None
    st.session_state.summary_source = selected_file


# ============================================================
# 3. DOCUMENT PROCESSING
# ============================================================

st.header("⚙️ Document Processing")


is_indexed = is_document_indexed(
    selected_file
)


if is_indexed:

    chunk_count = get_document_chunk_count(
        selected_file
    )

    st.success(
        "✓ Document is ready for analysis"
    )

    st.caption(
        f"{chunk_count} document sections processed"
    )

else:

    st.warning(
        "This document has not been processed yet."
    )


# ------------------------------------------------------------
# Index button
# ------------------------------------------------------------

if st.button(
    "🔄 Process Document",
    use_container_width=True
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        selected_file
    )

    try:

        with st.spinner(
            "Processing document..."
        ):

            # Extract text
            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()

            # Clean text
            cleaned_text = clean_text(
                extracted_text
            )

            # Create chunks
            chunks = chunk_text(
                cleaned_text
            )

            # Store embeddings
            chunk_count = add_document_to_vector_store(
                chunks,
                selected_file
            )

        st.success(
            "✓ Document processed successfully"
        )

        st.caption(
            f"{chunk_count} sections indexed for search"
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"Document processing failed: {e}"
        )


st.divider()


# ============================================================
# 4. ASK QUESTIONS
# ============================================================

st.header("💬 Ask Questions")

st.write(
    "Ask questions about the selected document."
)


question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What security measures are "
        "defined in this document?"
    ),
    height=100,
    label_visibility="visible"
)


if st.button(
    "🔍 Ask Question",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not is_document_indexed(
        selected_file
    ):

        st.warning(
            "Please process the document before "
            "asking questions."
        )

    else:

        try:

            with st.spinner(
                "Searching the document..."
            ):

                result = ask_question(
                    question=question,
                    source=selected_file,
                    k=5
                )

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.subheader("💡 Answer")

            st.write(
                result["answer"]
            )

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            if result["sources"]:

                st.caption(
                    f"📄 Source: {selected_file}"
                )

        except Exception as e:

            st.error(
                f"Unable to answer the question: {e}"
            )


st.divider()


# ============================================================
# 5. DOCUMENT SUMMARY
# ============================================================

st.header("📝 Document Summary")

st.write(
    "Generate an executive summary of the selected document."
)


if st.button(
    "✨ Generate Summary",
    use_container_width=True
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        selected_file
    )

    try:

        with st.spinner(
            "Generating executive summary..."
        ):

            # Extract document text
            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()

            # Generate summary
            summary = summarize_document(
                extracted_text
            )

            # Store in session state
            st.session_state.summary = summary
            st.session_state.summary_source = selected_file

    except Exception as e:

        st.error(
            f"Unable to generate summary: {e}"
        )


# ------------------------------------------------------------
# Display summary
# ------------------------------------------------------------

if (
    st.session_state.summary
    and
    st.session_state.summary_source == selected_file
):

    st.subheader("📌 Executive Summary")

    st.markdown(
        st.session_state.summary
    )


st.divider()


# ============================================================
# 6. DOCUMENT PREVIEW
# ============================================================

st.header("👁️ Document Preview")

with st.expander(
    "View extracted document text"
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        selected_file
    )

    try:

        extracted_text = extract_text_from_pdf(
            file_path
        )

        if extracted_text:

            st.text_area(
                "Document Text",
                extracted_text[:5000],
                height=350
            )

        else:

            st.warning(
                "No readable text was found."
            )

    except Exception as e:

        st.error(
            f"Unable to preview document: {e}"
        )
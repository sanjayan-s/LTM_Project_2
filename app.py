import os

import streamlit as st

from modules.pdf_loader import extract_text_from_pdf
from modules.auth import login, logout
from modules.document_manager import upload_document
from modules.text_processor import (
    clean_text,
    chunk_text
)
from modules.vector_store import (
    add_document_to_vector_store,
    is_document_indexed,
    get_document_chunk_count
)
from modules.rag_engine import ask_question


UPLOAD_FOLDER = "uploads"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

if not st.session_state.logged_in:

    login()

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

else:

    st.title("📘 Knowledge Guardian Lite")

    st.success(
        f"Welcome, {st.session_state.username}"
    )

    logout()

    st.divider()


    # --------------------------------------------------
    # DOCUMENT UPLOAD
    # --------------------------------------------------

    st.subheader("📄 Upload Document")

    upload_document()

    st.divider()


    # --------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------

    st.subheader("📁 My Documents")

    uploaded_files = os.listdir(
        UPLOAD_FOLDER
    )

    if not uploaded_files:

        st.info(
            "No documents uploaded yet."
        )

    else:

        for file in uploaded_files:

            st.write(
                f"📄 {file}"
            )


        st.divider()


        # --------------------------------------------------
        # DOCUMENT SELECTION
        # --------------------------------------------------

        selected_file = st.selectbox(
            "Select a document",
            uploaded_files
        )


        # --------------------------------------------------
        # DOCUMENT PROCESSING
        # --------------------------------------------------

        st.subheader(
            "⚙️ Document Processing"
        )


        if is_document_indexed(
            selected_file
        ):

            chunk_count = (
                get_document_chunk_count(
                    selected_file
                )
            )

            st.success(
                f"✓ Document indexed ({chunk_count} chunks)"
            )

        else:

            st.info(
                "This document has not been indexed yet."
            )


        if st.button(
            "🔄 Index Document",
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

                    # Extract PDF text
                    extracted_text = (
                        extract_text_from_pdf(
                            file_path
                        )
                    )


                    if not extracted_text.strip():

                        st.error(
                            "No readable text was found in this PDF."
                        )

                    else:

                        # Clean extracted text
                        cleaned_text = (
                            clean_text(
                                extracted_text
                            )
                        )


                        # Create chunks
                        chunks = (
                            chunk_text(
                                cleaned_text
                            )
                        )


                        # Store embeddings in ChromaDB
                        chunk_count = (
                            add_document_to_vector_store(
                                chunks,
                                selected_file
                            )
                        )


                        st.success(
                            f"✓ Document indexed successfully "
                            f"({chunk_count} chunks)"
                        )


                        # Refresh page state
                        st.session_state[
                            "document_indexed"
                        ] = True


            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )


        # --------------------------------------------------
        # RAG QUESTION ANSWERING
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "💬 Ask Questions"
        )


        question = st.text_area(
            "Ask something about this document",
            placeholder=(
                "Example: What security measures are defined?"
            ),
            height=100
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
                    "Please index the document before asking questions."
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


                    # --------------------------------------------------
                    # ANSWER
                    # --------------------------------------------------

                    st.subheader(
                        "💡 Answer"
                    )


                    st.write(
                        result["answer"]
                    )


                    # --------------------------------------------------
                    # SOURCE
                    # --------------------------------------------------

                    if result["sources"]:

                        st.caption(
                            f"📄 Source: {selected_file}"
                        )


                except Exception as e:

                    st.error(
                        f"Unable to answer the question: {e}"
                    )


        # --------------------------------------------------
        # DOCUMENT PREVIEW
        # --------------------------------------------------

        with st.expander(
            "👁️ Preview Document"
        ):

            file_path = os.path.join(
                UPLOAD_FOLDER,
                selected_file
            )


            try:

                extracted_text = (
                    extract_text_from_pdf(
                        file_path
                    )
                )


                if extracted_text:

                    st.text_area(
                        "Extracted Text",
                        extracted_text[:3000],
                        height=300
                    )

                else:

                    st.warning(
                        "No readable text found."
                    )


            except Exception as e:

                st.error(
                    f"Unable to preview document: {e}"
                )
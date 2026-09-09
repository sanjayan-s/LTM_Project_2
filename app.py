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
from modules.action_items import extract_action_items
from modules.gap_analyzer import analyze_gaps


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

if "action_items" not in st.session_state:
    st.session_state.action_items = None

if "action_items_source" not in st.session_state:
    st.session_state.action_items_source = None

if "gap_analysis" not in st.session_state:
    st.session_state.gap_analysis = None

if "gap_analysis_source" not in st.session_state:
    st.session_state.gap_analysis_source = None


# ============================================================
# AUTHENTICATION
# ============================================================

if not st.session_state.logged_in:
    login()
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

st.title("📘 Knowledge Guardian Lite")

st.write(f"Welcome, **{st.session_state.username}**")

logout()

st.divider()


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

st.header("📄 Upload Document")

upload_document()

st.divider()


# ============================================================
# MY DOCUMENTS
# ============================================================

st.header("📁 My Documents")

uploaded_files = [
    file
    for file in os.listdir(UPLOAD_FOLDER)
    if file.lower().endswith(".pdf")
]

if not uploaded_files:
    st.info("No documents uploaded yet. Upload a PDF to get started.")
    st.stop()


selected_file = st.selectbox(
    "Select a document",
    uploaded_files
)


# ============================================================
# RESET ANALYSIS WHEN DOCUMENT CHANGES
# ============================================================

if st.session_state.summary_source != selected_file:
    st.session_state.summary = None
    st.session_state.summary_source = selected_file


if st.session_state.action_items_source != selected_file:
    st.session_state.action_items = None
    st.session_state.action_items_source = selected_file


if st.session_state.gap_analysis_source != selected_file:
    st.session_state.gap_analysis = None
    st.session_state.gap_analysis_source = selected_file


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

st.header("⚙️ Document Processing")

is_indexed = is_document_indexed(selected_file)

if is_indexed:

    chunk_count = get_document_chunk_count(selected_file)

    st.success("✓ Document is ready for analysis")

    st.caption(
        f"{chunk_count} document sections processed"
    )

else:

    st.warning(
        "This document has not been processed yet."
    )


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

            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()


            cleaned_text = clean_text(
                extracted_text
            )

            chunks = chunk_text(
                cleaned_text
            )

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
# ASK QUESTIONS
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

    elif not is_document_indexed(selected_file):

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


            st.subheader("💡 Answer")

            st.write(
                result["answer"]
            )


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
# DOCUMENT SUMMARY
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

            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()


            summary = summarize_document(
                extracted_text
            )


            st.session_state.summary = summary

            st.session_state.summary_source = selected_file


    except Exception as e:

        st.error(
            f"Unable to generate summary: {e}"
        )


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
# ACTION ITEMS
# ============================================================

st.header("📋 Action Items")

st.write(
    "Extract actionable tasks from the selected document."
)


if st.button(
    "✨ Extract Action Items",
    use_container_width=True
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        selected_file
    )

    try:

        with st.spinner(
            "Extracting action items..."
        ):

            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()


            action_items = extract_action_items(
                extracted_text
            )


            st.session_state.action_items = action_items

            st.session_state.action_items_source = selected_file


    except Exception as e:

        st.error(
            f"Unable to extract action items: {e}"
        )


if (
    st.session_state.action_items is not None
    and
    st.session_state.action_items_source == selected_file
):

    action_items = st.session_state.action_items


    if not action_items:

        st.info(
            "ℹ️ No action items were identified "
            "in this document."
        )


    else:

        st.subheader(
            "📌 Identified Action Items"
        )


        for index, item in enumerate(
            action_items,
            start=1
        ):

            st.markdown(
                f"### {index}. {item['action']}"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.caption("Owner")

                st.write(
                    item["owner"]
                )


            with col2:

                st.caption("Deadline")

                st.write(
                    item["deadline"]
                )


            with col3:

                st.caption("Priority")

                st.write(
                    item["priority"]
                )


            st.divider()


st.divider()


# ============================================================
# KNOWLEDGE GAP ANALYSIS
# ============================================================

st.header("🧠 Knowledge Gap Analysis")

st.write(
    "Identify missing or incomplete sections "
    "and assess document risk."
)


if st.button(
    "🔎 Analyze Knowledge Gaps",
    use_container_width=True
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        selected_file
    )

    try:

        with st.spinner(
            "Analyzing document completeness..."
        ):

            extracted_text = extract_text_from_pdf(
                file_path
            )

            if not extracted_text.strip():

                st.error(
                    "No readable text was found in this PDF."
                )

                st.stop()


            gap_analysis = analyze_gaps(
                extracted_text
            )


            st.session_state.gap_analysis = (
                gap_analysis
            )

            st.session_state.gap_analysis_source = (
                selected_file
            )


    except Exception as e:

        st.error(
            f"Unable to analyze knowledge gaps: {e}"
        )


# ============================================================
# DISPLAY KNOWLEDGE GAP RESULTS
# ============================================================

if (
    st.session_state.gap_analysis is not None
    and
    st.session_state.gap_analysis_source == selected_file
):

    result = st.session_state.gap_analysis


    # --------------------------------------------------------
    # DOCUMENT TYPE
    # --------------------------------------------------------

    st.subheader("📄 Document Type")

    st.info(
        result["document_type"]
    )


    # --------------------------------------------------------
    # OVERALL RISK
    # --------------------------------------------------------

    st.subheader("⚠️ Overall Risk")

    overall_risk = result["overall_risk"]


    if overall_risk == "High":

        st.error(
            "🔴 High Risk"
        )

    elif overall_risk == "Medium":

        st.warning(
            "🟡 Medium Risk"
        )

    else:

        st.success(
            "🟢 Low Risk"
        )


    # --------------------------------------------------------
    # STATUS SUMMARY
    # --------------------------------------------------------

    sections = result["sections"]

    present_count = sum(
        1
        for item in sections
        if item["status"] == "Present"
    )

    partial_count = sum(
        1
        for item in sections
        if item["status"] == "Partial"
    )

    missing_count = sum(
        1
        for item in sections
        if item["status"] == "Missing"
    )


    st.subheader("📊 Completeness Overview")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Present",
            present_count
        )


    with col2:

        st.metric(
            "Partial",
            partial_count
        )


    with col3:

        st.metric(
            "Missing",
            missing_count
        )


    # --------------------------------------------------------
    # SECTION ANALYSIS
    # --------------------------------------------------------

    st.subheader(
        "📋 Section Analysis"
    )


    for item in sections:

        section = item["section"]

        status = item["status"]

        evidence = item["evidence"]

        risk = item["risk"]

        recommendation = item["recommendation"]


        if status == "Present":

            status_icon = "✅"

        elif status == "Partial":

            status_icon = "🟡"

        else:

            status_icon = "❌"


        with st.expander(
            f"{status_icon} {section} — {status}"
        ):

            st.write(
                f"**Evidence:** {evidence}"
            )

            st.write(
                f"**Risk:** {risk}"
            )


            if recommendation and recommendation != "No major gap identified":

                st.info(
                    f"💡 **Recommendation:** {recommendation}"
                )

            else:

                st.caption(
                    "✓ No major gap identified"
                )


st.divider()


# ============================================================
# DOCUMENT PREVIEW
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
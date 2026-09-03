import streamlit as st
import os
from modules.pdf_loader import extract_text_from_pdf
from modules.auth import login, logout
from modules.document_manager import (
    upload_document,
    show_uploaded_files
)

# Create session variable if it doesn't exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Show login page if not authenticated
if not st.session_state.logged_in:
    login()

# Dashboard after login
else:

    st.title("📘 Knowledge Guardian Lite")

    st.success(
        f"Welcome, {st.session_state.username}"
    )

    logout()

    upload_document()

    st.divider()

    show_uploaded_files()
    st.divider()

st.subheader("Document Preview")

uploaded_files = os.listdir("uploads")

if uploaded_files:

    selected_file = st.selectbox(
        "Select a document",
        uploaded_files
    )

    if st.button("Preview Content"):

        file_path = os.path.join(
            "uploads",
            selected_file
        )

        extracted_text = extract_text_from_pdf(file_path)

        st.text_area(
            "Extracted Text",
            extracted_text[:3000],
            height=300
        )

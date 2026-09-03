import os
import streamlit as st

UPLOAD_FOLDER = "uploads"


def save_uploaded_file(uploaded_file):
    file_path = os.path.join(
        UPLOAD_FOLDER,
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def upload_document():
    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file:

        file_path = os.path.join(
            UPLOAD_FOLDER,
            uploaded_file.name
        )

        if os.path.exists(file_path):
            st.warning(
                "File already exists."
            )
        else:
            save_uploaded_file(uploaded_file)
            st.success(
                f"{uploaded_file.name} uploaded successfully."
            )


def show_uploaded_files():
    st.subheader("📁 Uploaded Documents")

    files = os.listdir(UPLOAD_FOLDER)

    if not files:
        st.info("No documents uploaded.")

    for file in files:
        st.write(f"📄 {file}")
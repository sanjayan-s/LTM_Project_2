import streamlit as st
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
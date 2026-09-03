import json
import streamlit as st

def authenticate(username, password):
    try:
        with open("users.json", "r") as file:
            users = json.load(file)

        return users.get(username) == password

    except Exception:
        return False


def login():
    st.title("🔐 Knowledge Guardian Lite")

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")


def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
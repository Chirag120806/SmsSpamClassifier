"""Streamlit front-end. Run from the project root:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from spam_classifier import SpamClassifier


@st.cache_resource
def load_classifier() -> SpamClassifier:
    return SpamClassifier()


st.title("Email / SMS Spam Classifier")
message = st.text_area("Enter the message")

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message first.")
    elif load_classifier().is_spam(message):
        st.header("Spam")
    else:
        st.header("Not Spam")

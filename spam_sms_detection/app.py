import streamlit as st
import pickle
import os

st.set_page_config(page_title="SMS Spam Detection", page_icon="📧")


def transform_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch == " ")
    words = text.split()

    stop_words = {
        "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "it", "its",
        "they", "them", "their", "a", "an", "and",
        "or", "for", "to", "of", "in", "on", "with",
        "is", "am", "are", "was", "were", "be",
        "have", "has", "had", "the", "that"
    }

    words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)


@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    with open(os.path.join(base_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    return tfidf, model


tfidf, model = load_models()

st.title("📧 SMS Spam Detection")
st.caption("Developed by Sium Ahameed")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean = transform_text(msg)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

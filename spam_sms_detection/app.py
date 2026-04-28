import streamlit as st
import pickle
import os


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📧",
    layout="centered"
)


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Keep only letters, numbers, spaces
    text = "".join(char for char in text if char.isalnum() or char == " ")

    # Split into words
    words = text.split()

    # Basic stopwords
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her',
        'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'a', 'an', 'and', 'but', 'or', 'for', 'so', 'yet',
        'at', 'by', 'in', 'into', 'of', 'on', 'to', 'with',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'the', 'that'
    }

    # Remove stopwords + short words
    words = [word for word in words if word not in stop_words and len(word) > 2]

    return " ".join(words)


# -------------------------------
# LOAD MODEL FILES
# -------------------------------
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
    model_path = os.path.join(base_dir, "model.pkl")

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return tfidf, model


# -------------------------------
# APP HEADER
# -------------------------------
st.title("📧 SMS Spam Detection")
st.caption("Developed by Sium Ahameed")
st.markdown("---")


# -------------------------------
# LOAD FILES
# -------------------------------
with st.spinner("Loading model..."):
    tfidf, model = load_models()


# -------------------------------
# USER INPUT
# -------------------------------
input_sms = st.text_area(
    "Enter your message:",
    placeholder="Type your SMS here...",
    height=150
)


# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # Clean text
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Probability
        probability = model.predict_proba(vector_input)[0][1] * 100

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"🚨 Spam Message ({probability:.1f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({100 - probability:.1f}% confidence)")

        st.progress(probability / 100)


# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Powered by Machine Learning")

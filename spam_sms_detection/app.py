import streamlit as st
import pickle
import string
import os

# --- SIMPLE TEXT CLEANING (NO NLTK) ---
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and keep only letters/numbers
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    
    # Split into words
    words = text.split()
    
    # Basic stopwords (most common ones only)
    stop_words = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves',
        'a', 'an', 'and', 'but', 'or', 'for', 'so', 'yet',
        'at', 'by', 'in', 'into', 'of', 'on', 'to', 'with',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did',
        'doing', 'the', 'that'
    }
    
    # Remove stopwords and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return " ".join(words)


# --- PAGE SETUP ---
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 SMS Spam Detection")
st.caption("Developed by Sium Ahameed")
st.markdown("---")


# --- LOAD MODELS (FAST + NO LOADING UI) ---
@st.cache_resource(show_spinner=False)
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")
    model_path = os.path.join(base_dir, "model.pkl")

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return tfidf, model


tfidf, model = load_models()


# --- INPUT BOX ---
input_sms = st.text_area(
    "Enter the message below:",
    placeholder="Type here...",
    height=150
)


# --- PREDICTION ---
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter some text first!")

    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Probability
        prob = model.predict_proba(vector_input)[0]
        spam_probability = prob[1] * 100

        # 5. Output
        st.subheader("Result:")

        if result == 1:
            st.error(f"🚨 SPAM - {spam_probability:.1f}% confidence")
        else:
            st.success(f"✅ NOT SPAM - {100 - spam_probability:.1f}% confidence")

        st.progress(spam_probability / 100)


st.markdown("---")
st.caption("Powered by Machine Learning")

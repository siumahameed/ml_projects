import streamlit as st
import pickle
import string

# --- SIMPLE TEXT CLEANING (NO NLTK) ---
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and keep only letters/numbers
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    
    # Split into words
    words = text.split()
    
    # Basic stopwords (most common ones only)
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
                  'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
                  'they', 'them', 'their', 'theirs', 'themselves', 'a', 'an', 'and', 'but', 'or', 'for', 
                  'so', 'yet', 'at', 'by', 'in', 'into', 'of', 'on', 'to', 'with', 'is', 'am', 'are', 
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                  'did', 'doing', 'the', 'and', 'to', 'of', 'a', 'in', 'for', 'on', 'with', 'that'}
    
    # Remove stopwords and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return " ".join(words)

# --- LOAD MODELS (CACHED) ---
@st.cache_resource
def load_models():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

# --- LOAD EVERYTHING AT START ---
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 SMS Spam Detection")
st.caption("Developed by Sium Ahameed")
st.markdown("---")

# Load models once when app starts
with st.spinner("Loading models..."):
    tfidf, model = load_models()

input_sms = st.text_area("Enter the message below:", placeholder="Type here...", height=150)

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
        
        # 4. Get probability
        prob = model.predict_proba(vector_input)[0]
        spam_probability = prob[1] * 100

        # 5. Display Results
        st.subheader("Result:")
        
        if result == 1:
            st.error(f"🚨 SPAM - {spam_probability:.1f}% confidence")
        else:
            st.success(f"✅ NOT SPAM - {100-spam_probability:.1f}% confidence")
        
        st.progress(spam_probability / 100)

st.markdown("---")
st.caption("Powered by Machine Learning")




# ---------------------------------------( less faster but more detailed) ----------------------------
# for this you can use NLTK library and do more detailed text preprocessing like removing stopwords, stemming, lemmatization etc. But for simplicity and speed we are doing basic cleaning only.

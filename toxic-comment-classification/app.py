import streamlit as st
import pickle
import re
import string
import emoji
import numpy as np
from bs4 import BeautifulSoup

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    search_paths = [
        os.path.join(BASE_DIR, 'models', filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(os.getcwd(), 'models', filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return search_paths[0]

@st.cache_resource
def load_model():
    model_path = find_file('model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_labels():
    labels_path = find_file('labels.pkl')
    with open(labels_path, 'rb') as f:
        return pickle.load(f)

def clean_txt(txt):
    if not isinstance(txt, str):
        return " "
    txt = txt.lower()
    txt = re.sub(r"what's", "what is ", txt)
    txt = re.sub(r"\'ve", " have ", txt)
    txt = re.sub(r"can't", "can not ", txt)
    txt = re.sub(r"n't", " not ", txt)
    txt = re.sub(r"i'm", "i am ", txt)
    txt = re.sub(r"\'re", " are ", txt)
    txt = re.sub(r"\'d", " would ", txt)
    txt = re.sub(r"\'ll", " will ", txt)
    txt = re.sub(r'http\S+|www\S+', '', txt)
    txt = emoji.demojize(txt)
    txt = re.sub(r':\w+:', ' ', txt)
    txt = BeautifulSoup(txt, 'html.parser').get_text()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r'\d+', '', txt)
    return " ".join(txt.split()).strip()

def preprocess(text):
    return clean_txt(str(text)) if isinstance(text, str) else " "

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️")

st.title("🛡️ Toxic Comment Classifier")
st.caption("Multi-label NLP classification for detecting toxic content in comments")

st.divider()

try:
    model = load_model()
    labels = load_labels()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Run `python train_model.py` first.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Enter Comment")
    text_input = st.text_area(
        "Type or paste a comment:",
        height=100,
        label_visibility="collapsed"
    )
    analyze_btn = st.button("Analyze", type="primary")

with col2:
    st.subheader("📊 Labels")
    st.write(f"**6 Categories:**")
    for label in labels:
        st.write(f"• {label}")

if analyze_btn and text_input:
    with st.spinner("Analyzing..."):
        processed_text = preprocess(text_input)
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])
        
        toxic_labels = [labels[i] for i, p in enumerate(prediction) if p == 1]
    
    st.divider()
    
    col_result, col_conf = st.columns([1, 1])
    
    with col_result:
        if toxic_labels:
            st.error(f"⚠️ **TOXIC** - {len(toxic_labels)} category(s)")
            for label in toxic_labels:
                st.write(f"🚫 {label.replace('_', ' ').title()}")
        else:
            st.success("✅ **SAFE** - No toxic content detected")
    
    with col_conf:
        st.subheader("Confidence Scores")
        for i, label in enumerate(labels):
            prob = float(probabilities[0][i])
            color = "🔴" if prob > 0.5 else "🟡" if prob > 0.2 else "🟢"
            st.write(f"{color} {label}: {prob:.1%}")

    with st.expander("Processed Text"):
        st.text(processed_text)

st.divider()
st.caption("Built with Streamlit | NLP Multi-label Classification")
st.caption("Developed by **Sium Ahameed Bhuyn**")
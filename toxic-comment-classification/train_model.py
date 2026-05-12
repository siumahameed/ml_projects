import pandas as pd
import numpy as np
import pickle
import re
import string
import emoji
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

nltk.download('stopwords', quiet=True)

def remove_stopwords(txt):
    stopwords_set = set(stopwords.words('english'))
    return " ".join([w for w in txt.split() if w not in stopwords_set])

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

stemmer = SnowballStemmer('english')

def stemming(sentence):
    return " ".join([stemmer.stem(word) for word in sentence.split()])

def preprocess(text):
    text = remove_stopwords(text)
    text = clean_txt(text)
    text = stemming(text)
    return text

print("Loading data...")
df = pd.read_csv("train.csv")
df = df.drop('id', axis=1)

print(f"Using FULL dataset: {len(df)} samples")
# No sampling - use all data

print("Preprocessing text (optimized)...")
df['comment_text'] = df['comment_text'].apply(lambda x: clean_txt(str(x)) if isinstance(x, str) else " ")

X = df['comment_text']
y = df.drop("comment_text", axis=1)
labels = list(y.columns)

print("Building optimized model...")
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=15000, ngram_range=(1, 2))),
    ('classifier', OneVsRestClassifier(SGDClassifier(loss='log_loss', max_iter=300, random_state=42, alpha=0.0001, early_stopping=True, n_iter_no_change=5)))
])

print("Training model...")
model.fit(X, y)

print("Saving model...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print("Model saved successfully!")
print(f"Labels: {labels}")
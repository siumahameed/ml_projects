# 🚫 Toxic Comment Classifier

A machine learning web application built with Streamlit that predicts toxic content in comments. The model is trained using TF-IDF vectorization with SGDClassifier and deployed as an interactive web app.

---

## 🚀 Live Demo
Add your deployed link here:

[https://your-app-link.streamlit.app](https://toxiccommentclassifierbysium.streamlit.app/)

---

## 📌 Project Overview

This project detects toxic comments and classifies them into multiple categories such as toxic, severe_toxic, obscene, threat, insult, and identity_hate.

It demonstrates a full machine learning workflow:
- Data preprocessing (text cleaning, emoji handling)
- Feature engineering (TF-IDF with bigrams)
- Model training (SGDClassifier with OneVsRest)
- Model saving using Pickle
- Interactive web app deployment using Streamlit

---

## 🧠 Machine Learning Model

- Algorithm: SGDClassifier with OneVsRest wrapper
- Vectorization: TF-IDF (max_features=15000, ngram_range=(1,2))
- Scaling: Built into pipeline
- Task: Multi-label classification (6 toxicity categories)

---

## 📊 Labels

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- BeautifulSoup
- Emoji
- Streamlit

---

## 📁 Project Structure
Toxic-Comment-Classification/
│
├── app.py # Streamlit application
├── train_model.py # Model training script
├── models/
│   ├── model.pkl # Trained ML model
│   └── labels.pkl # Label names
├── train.csv # Training dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation

---

## 📥 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Toxic-Comment-Classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - model already trained):
```bash
python train_model.py
```

4. Run the app:
```bash
streamlit run app.py
```

---

## 🔧 Usage

1. Open the app in your browser
2. Enter or paste a comment in the text area
3. Click "Analyze" button
4. View the prediction results and confidence scores

---

## 👨‍💻 Developer

Developed by **Sium Ahameed Bhuyn**

---

## 📜 License

This project is for educational purposes.

# ❤️ Heart Disease Prediction App

A machine learning web application built with Streamlit that predicts the likelihood of heart disease based on patient health parameters. The model is trained using a K-Nearest Neighbors (KNN) algorithm and deployed as an interactive web app.

---

## 🚀 Live Demo
Add your deployed link here:

https://your-app-link.streamlit.app

---

## 📌 Project Overview

This project predicts whether a person is at risk of heart disease based on medical inputs such as age, cholesterol, blood pressure, and ECG results.

It demonstrates a full machine learning workflow:
- Data preprocessing
- Feature engineering (one-hot encoding)
- Model training (KNN classifier)
- Model saving using Joblib
- Interactive web app deployment using Streamlit

---

## 🧠 Machine Learning Model

- Algorithm: K-Nearest Neighbors (KNN)
- Scaling: StandardScaler
- Encoding: One-hot encoding
- Task: Binary classification (Heart Disease Risk)

---

## 📊 Input Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak (ST Depression)
- ST Slope

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

---

## 📁 Project Structure

Heart-Disease-Prediction/
│
├── app.py # Streamlit application
├── knn_heart_model.pkl # Trained ML model
├── heart_scaler.pkl # Feature scaler
├── heart_columns.pkl # Training feature columns
├── requirements.txt # Dependencies
└── README.md # Project documentation

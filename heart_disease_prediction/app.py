import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Files (Updated Paths)
# -----------------------------
# I add the folder name to the path so Streamlit can find them
model = joblib.load("heart_disease_prediction/knn_heart_model.pkl")
scaler = joblib.load("heart_disease_prediction/heart_scaler.pkl")
expected_columns = joblib.load("heart_disease_prediction/heart_columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Developed by SIUM")
st.markdown("Enter patient information below to predict heart disease risk.")

st.divider()

# -----------------------------
# Input Section
# -----------------------------
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=0,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)",
    0.0,
    6.5,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Result"):

    # Raw Input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,

        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add Missing Columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Correct Order
    input_df = input_df[expected_columns]

    # Scale Data
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(scaled_input)[0][1]
    else:
        probability = None

    # -----------------------------
    # Output
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    if probability is not None:
        st.write(f"Risk Probability: {probability:.2%}")

    # Show User Input
    with st.expander("See Input Data"):
        st.dataframe(input_df)

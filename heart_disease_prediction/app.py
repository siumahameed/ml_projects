import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="HeartCare AI | Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Custom CSS for Eye-Catchy UI
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
        color: white;
    }
    .name-text {
        font-family: 'Courier New', Courier, monospace;
        color: #ff4b4b;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        border: 2px solid #ff4b4b;
        padding: 5px;
        border-radius: 10px;
        width: fit-content;
        margin: auto;
    }
    </style>
    """, unsafe_allow_index=True)

# -----------------------------
# Load Files (Updated Paths)
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("heart_disease_prediction/knn_heart_model.pkl")
    scaler = joblib.load("heart_disease_prediction/heart_scaler.pkl")
    expected_columns = joblib.load("heart_disease_prediction/heart_columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------
# Header Section
# -----------------------------
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=100)
with col2:
    st.title("Heart Disease Prediction AI")
    st.markdown("##### Precision Analytics for Cardiovascular Health")

# Eye-catchy name tag
st.markdown('<div class="name-text">Developed by SIUM</div>', unsafe_allow_index=True)
st.write("---")

# -----------------------------
# Input Section (Two-Column Layout)
# -----------------------------
st.subheader("📋 Patient Clinical Information")
col_a, col_b = st.columns(2)

with col_a:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 0, 700, 200)

with col_b:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.5, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("") # Spacer

# -----------------------------
# Prediction Button & Logic
# -----------------------------
if st.button("RUN DIAGNOSTIC PREDICTION"):
    # Raw Input Dictionary
    raw_input = {
        "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "MaxHR": max_hr, "Oldpeak": oldpeak,
        "Sex_" + sex: 1, "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1, "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Process Data
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    
    # Scale & Predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if hasattr(model, "predict_proba"):
        prob_val = model.predict_proba(scaled_input)[0][1]
    else:
        prob_val = float(prediction)

    # -----------------------------
    # Realistic Results UI
    # -----------------------------
    st.divider()
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        st.subheader("Diagnostic Assessment")
        if prob_val > 0.7:
            st.error("### ⚠️ HIGH RISK DETECTED")
            st.write("The model indicates clinical markers consistent with heart disease. Immediate consultation with a cardiologist is advised.")
        elif 0.35 <= prob_val <= 0.7:
            st.warning("### 🟠 MODERATE RISK")
            st.write("Some markers are elevated. We recommend reviewing lifestyle factors and scheduling a routine check-up.")
        else:
            st.success("### ✅ LOW RISK")
            st.write("Current indicators suggest a healthy cardiovascular profile. Continue maintaining a balanced diet and regular exercise.")

    with res_col2:
        st.metric(label="Risk Probability", value=f"{prob_val:.1%}")
        st.progress(prob_val)

    # Technical Details
    with st.expander("🔍 View Technical Metadata"):
        st.write("This prediction is based on a K-Nearest Neighbors (KNN) algorithm trained on the UCI Heart Disease Dataset.")
        st.dataframe(input_df)

st.write("---")
st.caption("Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.")

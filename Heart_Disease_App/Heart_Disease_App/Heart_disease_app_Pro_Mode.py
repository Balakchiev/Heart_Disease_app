import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------
# 🧠 Load the real trained model and scaler
# ----------------------------------------
model = joblib.load("model_pro.pkl")

scaler = joblib.load("model_pro.pkl")


# ----------------------------------------
# 🖥️ Streamlit App Interface - Pro Mode
# ----------------------------------------
st.set_page_config(page_title="Heart Disease Predictor - Pro Mode", layout="centered")
st.title("💉 Heart Disease Risk Estimator (Pro Mode)")
st.markdown("Enter full medical metrics to estimate heart disease risk using a trained machine learning model.")

# Sidebar Input
st.sidebar.header("🔧 Input Medical Parameters")

age = st.sidebar.slider("Age", 30, 80, 50)
sex = st.sidebar.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL? (1 = true, 0 = false)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment (0 = up, 1 = flat, 2 = down)", [0, 1, 2])
ca = st.sidebar.slider("Number of Major Vessels Colored (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Combine into feature array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

input_scaled = scaler.transform(input_data)

# Predict
if st.button("🧠 Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📋 Prediction Result")
    st.write("**Risk Classification:**", "🟥 Positive" if prediction == 1 else "🟩 Negative")
    st.write(f"**Estimated Probability of Heart Disease:** {prob:.2%}")

    st.progress(min(int(prob * 100), 100))
    st.bar_chart({"Probability": [prob], "Inverse": [1 - prob]})

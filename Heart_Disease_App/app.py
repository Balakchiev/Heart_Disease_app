import streamlit as st
import numpy as np
import joblib

# ----------------------------------------
# ✅ Load model and scaler (MUST be in same folder)
# ----------------------------------------
model = joblib.load("model_pro.pkl")
scaler = joblib.load("scaler_pro.pkl")

# ----------------------------------------
# 🖥️ Streamlit App - Pro Mode (Full Features)
# ----------------------------------------
st.set_page_config(page_title="Heart Disease Predictor - Pro Mode", layout="centered")
st.title("💉 Heart Disease Risk Estimator (Pro Mode)")
st.markdown("Enter full clinical data to estimate heart disease risk.")

st.sidebar.header("📋 Enter Parameters")

# User Inputs
age = st.sidebar.slider("Age", 30, 80, 50)
sex = st.sidebar.selectbox("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL?", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.sidebar.slider("Number of Major Vessels Colored (ca)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia (thal)", [1, 2, 3])

# Feature Vector
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🧠 Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction")
    st.write("🟥 High Risk" if pred == 1 else "🟩 Low Risk")
    st.write(f"**Estimated Probability:** {prob:.2%}")
    st.progress(min(int(prob * 100), 100))
    st.bar_chart({"Probability": [prob], "Inverse": [1 - prob]})

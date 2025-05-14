import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# 🧠 Load model and scaler (placeholder if saved model is needed)
# ----------------------------------------
# For this example, we'll train a basic logistic regression inline
from sklearn.datasets import load_iris
model = LogisticRegression()
scaler = StandardScaler()

# Dummy train to prevent error (replace with real model in production)
X_dummy = np.random.rand(100, 5)
y_dummy = np.random.randint(0, 2, 100)
X_scaled = scaler.fit_transform(X_dummy)
model.fit(X_scaled, y_dummy)

# ----------------------------------------
# 🖥️ Streamlit App Interface
# ----------------------------------------
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("💓 Heart Disease Risk Estimator")
st.markdown("Enter the details below to estimate the probability of heart disease.")

# Sidebar input
age = st.slider("Age", 30, 80, 50)
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
chol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 200)
thalach = st.slider("Max Heart Rate Achieved (thalach)", 70, 210, 150)
oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.0, 1.0)

# Create input array
input_data = np.array([[age, cp, chol, thalach, oldpeak]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🩺 Prediction Result")
    st.write("**Risk of Heart Disease:**", "Positive" if prediction == 1 else "Negative")
    st.write(f"**Estimated Probability:** {prob:.2%}")

    # Visualization
    st.progress(min(int(prob * 100), 100))
    st.bar_chart({"Probability": [prob], "Inverse": [1 - prob]})

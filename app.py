import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model('diabetes_model.h5')
scaler = joblib.load('scaler.pkl')

# Load dataset
df = pd.read_csv('diabetes.csv')

# Streamlit App
st.set_page_config(page_title="Diabetes Detection App", layout="centered")
st.title("🩺 Diabetes Detection using Pima Indians Dataset")

# Sidebar navigation
section = st.sidebar.radio("Choose Section", ["EDA", "Diabetes Prediction"])

# --- EDA Section ---
if section == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Data Sample")
    st.dataframe(df.head())

    st.subheader("Distribution of Outcome")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# --- Prediction Section ---
elif section == "Diabetes Prediction":
    st.header("🔍 Enter Your Health Info")

    # Input form
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
    glucose = st.number_input("Glucose", min_value=0, value=85)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=66)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=29)
    insulin = st.number_input("Insulin", min_value=0, value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f", value=26.6)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.351)
    age = st.number_input("Age", min_value=21, max_value=120, step=1, value=21)


    if st.button("Predict"):
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])
        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0][0]

        if prediction > 0.5:
            st.error(f"🚨 High Risk of Diabetes! (Probability: {prediction:.2f})")
        else:
            st.success(f"✅ Low Risk of Diabetes (Probability: {prediction:.2f})")

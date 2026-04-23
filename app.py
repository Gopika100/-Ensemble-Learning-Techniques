import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("random_forest_model.pkl")

st.title("Ensemble Learning - Random Forest App")

st.write("Enter feature values:")

# Example inputs (change based on your dataset)
f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")
f3 = st.number_input("Feature 3")

if st.button("Predict"):
    data = np.array([[f1, f2, f3]])
    result = model.predict(data)

    st.success(f"Prediction: {result[0]}")
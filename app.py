import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Define the input fields
st.title("Breast Cancer Prediction")

# User input fields for feature values
radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.2f")
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.2f")
area_mean = st.number_input("Area Mean", min_value=0.0, format="%.2f")
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.2f")
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.2f")
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.2f")
radius_se = st.number_input("Radius SE", min_value=0.0, format="%.2f")
perimeter_se = st.number_input("Perimeter SE", min_value=0.0, format="%.2f")
area_se = st.number_input("Area SE", min_value=0.0, format="%.2f")
radius_worst = st.number_input("Radius Worst", min_value=0.0, format="%.2f")
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, format="%.2f")
compactness_worst = st.number_input("Compactness Worst", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict"):
    # Prepare the input data
    new_patient_data = np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean, 
                                   concavity_mean, concave_points_mean, radius_se, perimeter_se, 
                                   area_se, radius_worst, perimeter_worst, compactness_worst]])
    
    # Make prediction
    prediction = model.predict(new_patient_data)

    # Display result
    if prediction[0] == 1:
        st.error("The model predicts the tumor is **Malignant (Cancerous)**.")
    else:
        st.success("The model predicts the tumor is **Benign (Non-cancerous)**.")


import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("kmeans_model.pkl")

# Streamlit UI
st.title("K-Means Clustering App 🎯")
st.write("Enter the Math & Science scores to predict the student's cluster.")

# Input fields
math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
science_score = st.number_input("Science Score", min_value=0, max_value=100, value=50)

if st.button("Predict Cluster"):
    input_data = np.array([[math_score, science_score]])
    cluster = model.predict(input_data)[0]
    
    # Meaningful interpretation
    cluster_label = "High Scoring Student" if cluster == 0 else "Low Scoring Student"
    
    st.success(f"The student belongs to: {cluster_label} (Cluster {cluster}) 🎯")


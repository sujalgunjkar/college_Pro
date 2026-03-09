import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Student Performance Prediction")

st.markdown("<h1 style='text-align: center;'>🎓 Student Performance Prediction System</h1>", unsafe_allow_html=True)
st.write("Predict whether a student will Pass or Fail")
# st.markdown("This project predicts whether a student will pass or fail using a Random Forest Machine Learning model.")


# Input Fields
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks", 0, 100, 50)
study_hours = st.slider("Study Hours per Day", 0, 10, 4)
assignments = st.slider("Assignments Completed", 0, 10, 4)

if st.button("Predict Result"):

    input_data = np.array([[attendance, internal_marks, study_hours, assignments]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)


    if prediction[0] == 1:
        st.success("✅ The student is likely to PASS")
    else:
        st.error("❌ The student is likely to FAIL")

    st.write("Pass Probability:", round(probability[0][1]*100,2), "%")


importance = model.feature_importances_
features = ["Attendance","Internal Marks","Study Hours","Assignments"]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.subheader("Feature Importance")
    st.bar_chart(importance_df.set_index("Feature"))



st.write("Model Accuracy: ~90% (trained model)")
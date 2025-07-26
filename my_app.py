import streamlit as st
import numpy as np

st.title("Salary Prediction App")

st.divider()

st.write("With this app, you can get an estimate for the salaries of the company employees.")

years = st.number_input("Years", value=1, step=1, min_value=0)
jobrate = st.number_input("Job Rate (e.g., performance rating)", value=3.5, step=0.5, min_value=0.0)

# Create the input feature array for the model
# The model expects a 2D array, even for a single prediction
X = np.array([[years, jobrate]]) 

# Load the trained model
try:
    model = joblib.load("linearmodel.pkl")
except FileNotFoundError:
    st.error("Error: 'linearmodel.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop() # Stop execution if the model isn't found

st.divider()

predict_button = st.button("Predict Salary")

st.divider()

if predict_button:
    st.balloons()
    try:
        prediction = model.predict(X)
        # Display the prediction clearly, rounding if necessary for better readability
        st.success(f"The predicted salary is: ${prediction[0]:,.2f}") 
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Please enter the details and click the 'Predict Salary' button to get an estimate.")

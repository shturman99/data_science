import stremlit as st
import joblib 
import numpy as np


st.title("Salary Estimantion App")

st.divider()

yr_at_company = st.number_input("Enter years at company", min_value=0, max_value=20)
satisfaction_level = st.number_input("Enter satisfaction level", min_value=0.0, max_value=1.0)
average_monthly_hours = st.number_input("Average monthly hours", min_value=120, max_value=400)



X =[yr_at_company, satisfaction_level, average_monthly_hours]

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

predict_button = st.button("Press for prediction the salary")

st.divider()


if predict_button:
    
    st.balloons()

    X1 = np.array(X)

    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    st.write(f"Salary prediction is {prediction}")

else:
    st.write("Please enter the values and press to predict button")
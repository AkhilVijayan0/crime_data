import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input
model = joblib.load("Case Status.pkl")
st.title("Crimes Data")
AREA = st.number_input("Enter AREA")
Part = st.number_input("Enter Part 1-2")
Crm Cd = st.number_input("Enter Crm Cd")
Crm Cd Desc = st.number_input("Enter Crm Cd Desc")
Vict Sex = st.number_input("Enter Vict Sex")
Vict Descent = st.number_input("Enter Vict Descent")
Premis Desc = st.number_input("Enter Premis Desc")
LOCATION = st.number_input("Enter LOCATION")
Day of Week = st.number_input("Enter Day of Week")

manual_input = pd.DataFrame({
  'AREA' = ['AREA'],
  'Part 1-2' = ['Part'],
  'Crm Cd' = ['Crm Cd'],
  'Crm Cd Desc' = ['Crm Cd Desc'],
  'Vict Sex' = ['Vict Sex'],
  'Vict Descent' = ['Vict Descent'],
  'Premis Desc' = ['Premis Desc'],
  'LOCATION' = ['LOCATION'],
  'Day of Week' = ['Day of Week']
})
st.write("Input Summary",manual_input)
if st.button("Predict case"):
        processed_manual = preprocess_input(manual_input)
        predictions = model.predict(processed_manual)
        
        prediction_text = "Closed" if predictions[0] == 1 else "Not Closed"
        
        st.write("Fraud Prediction:")
        st.success(prediction_text) 



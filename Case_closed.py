import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input
model = joblib.load("Case Status.pkl")
st.title("Crimes Data")
area= st.number_input("Enter AREA")
Part = st.number_input("Enter Part 1-2")
Crm_Cd = st.number_input("Enter Crm Cd")
Crm_Cd_Desc = st.number_input("Enter Crm Cd Desc")
Vict_Sex = st.number_input("Enter Vict Sex")
Vict_Descent = st.number_input("Enter Vict Descent")
Premis_Desc = st.number_input("Enter Premis Desc")
location = st.number_input("Enter LOCATION")
Day_of_Week = st.number_input("Enter Day of Week")

manual_input = pd.DataFrame({
  'AREA': [area],
  'Part 1-2': [Part],
  'Crm Cd': [Crm_Cd],
  'Crm Cd Desc': [Crm_Cd_Desc],
  'Vict Sex': [Vict_Sex],
  'Vict Descent': [Vict_Descent],
  'Premis Desc': [Premis_Desc],
  'LOCATION': [location],
  'Day of Week': [Day_of_Week]
})

st.write("Input Summary",manual_input)
if st.button("Predict case"):
        processed_manual = preprocess_input(manual_input)
        predictions = model.predict(processed_manual)
        
        prediction_text = "Closed" if predictions[0] == 1 else "Not Closed"
        
        st.write("Fraud Prediction:")
        st.success(prediction_text) 



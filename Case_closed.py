import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input
model = joblib.load("Case Status.pkl")
st.title("Crimes Data")
AREA = st.number_input("Enter AREA")
Part 1-2 = st.number_input("Enter Part 1-2")
Crm Cd = st.number_input("Enter Crm Cd")
Crm Cd Desc = st.number_input("Enter Crm Cd Desc")
Vict Sex = st.number_input("Enter Vict Sex")
Vict Descent = st.number_input("Enter Vict Descent")
Premis Desc = st.number_input("Enter Premis Desc")
LOCATION = st.number_input("Enter LOCATION")
Day of Week = st.number_input("Enter Day of Week")

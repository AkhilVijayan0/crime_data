import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
lab=joblib.load('Label_Encoder.pkl')
def preprocess_input(df):
#   df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
#   df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], errors='coerce')
  selected_columns = [
    'AREA', 'Part 1-2', 'Crm Cd', 'Crm Cd Desc',
    'Vict Sex', 'Vict Descent', 'Premis Desc',
    'LOCATION', 'Day of Week'
      ]

  # Create a new DataFrame with only the selected columns
  df_selected = df[selected_columns]
  return df_selected
  


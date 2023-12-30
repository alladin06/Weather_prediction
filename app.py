import streamlit as st
import joblib
from joblib import load
import pandas
import numpy
# from sklearn import *
from datetime import datetime
st.title("Weather Prediction App")
selected_date = st.date_input("Select a date", key='date_picker')
select_city = st.selectbox("Select a city", ["Delhi", "Mumbai", "Kolkata","Lucknow","Patna","Ahmadabad","Hyderabad","Chennai","Srinagar","Bhopal","Ranchi","Jaipur","Bhubhaneshwar"])

with open(f"models/{select_city}_temperature.joblib", 'rb') as file:
    model_t = load(file)
with open(f"models/{select_city}_humidity.joblib", 'rb') as file:
    model_h = load(file)
# custom_date_str = str(selected_date)
# custom_date = datetime.strptime(custom_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
# day_of_year = custom_date.timetuple().tm_yday
# custom_feature = [[day_of_year]]
# Parsing the selected date directly without converting it to string and back
custom_date = selected_date
day_of_year = custom_date.timetuple().tm_yday
custom_feature = [[day_of_year]]
if st.button("Get the Weather forecast"):
    predicted_temperature = model_t.predict(custom_feature)
    predicted_humidity = model_h.predict(custom_feature)
    st.write(f"Temperature: {predicted_temperature[0]}")
    st.write(f"Humidity: {predicted_humidity[0]}")

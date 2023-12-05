import streamlit as st
import pandas as pd
from models import LogisiticRegression


st.markdown("# Logistic Regression")
st.sidebar.header("Logistic Regression")

# Access the dictionaries
airport_codes = config.airport_codes
airline_codes = config.airline_codes
months = config.months
    
airports = {"ATL":10397,  
            "CLT":11057,  
            "DEN":11292,  
            "DFW":11298,  
            "LAS":12889,  
            "LAX":12892,  
            "LGA":12953,  
            "ORD":13930,  
            "PHX":14107,  
            "SEA":14747,}

carrier =   {
    "9E",
    "NK",
    "YV",
    "WN",
    "UA",
    "QX",
    "OO",
    "OH",
    "MQ",
    "AA",
    "HA",
    "G4",
    "F9",
    "DL",
    "B6",
    "AS",
    "YX" 
}
##Input
#Enter Month
origin = st.sidebar.selectbox('Origin Airport', airports.keys())
#Airport Code
dest = st.sidebar.selectbox('Dest Airport', airports.keys())
#Airline Code
carrier = st.sidebar.selectbox('Airline Carrier', carrier)
#Airline Code
delay = st.sidebar.number_input('Delay In Departure', value=None, placeholder="Type a number...")

#Generate Statistics
submit = st.sidebar.button('Generate Statistics')
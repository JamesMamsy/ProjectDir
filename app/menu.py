import streamlit as st

#import models

kmeans, apriori, regression = False,False,False #initModels()

# def initModels():
#     return False,False,False

st.session_state['airport_codes'] = {
    "New York, NY: John F. Kennedy International":"JFK",
    "Los Angeles, CA: Los Angeles International":"LAX",
    "Oklahoma City, OK: Will Rogers World":"OKC"
}

st.session_state['airline_codes'] = {
    "American Airlines Inc.":"AA",
    "Delta Air Lines Inc.":"DL",
    "Southwest Airlines Co.":"WN",
    "United Air Lines Inc.":"UA"
}


st.session_state['months'] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

days = range(31)

st.set_page_config(
    page_title="Airline Predictions",
    page_icon="✈️",
)

st.write("# Airline Delay Predictior")

st.sidebar.success("Select a model to use for prediction.")

st.markdown(
    """
    This application uses data mining and public flight data to predict the outcome of flight delay
    
    **👈 Select a model from the sidebar** to see some examples
    # Models
    
    ### Apriori
    - Input
    - Output
    ### K-means
    - Input
    - Output
    ### Logistic Regression
    - Input
    - Output
"""
)
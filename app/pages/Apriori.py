import streamlit as st
import altair as alt
import pandas as pd
from models.apriori import FlightDelayApriori

airport_codes = {
    "New York, NY: John F. Kennedy International": "JFK",
    "Los Angeles, CA: Los Angeles International": "LAX",
    "Oklahoma City, OK: Will Rogers World": "OKC"
}

airline_codes = {
    "American Airlines Inc.": "AA",
    "Delta Air Lines Inc.": "DL",
    "Southwest Airlines Co.": "WN",
    "United Air Lines Inc.": "UA"
}

months = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# create instance
delay_analysis = FlightDelayApriori()
# delay_analysis = FlightDelayApriori(st.session_state['data'])

st.markdown("# Apriori")
st.sidebar.header("Apriori")

# Enter Month
month_options = list(months.items())  # Creates a list of tuples
month = st.sidebar.selectbox('Select Month of flight', options=month_options, format_func=lambda x: x[0])
selected_month = month[1]  # This will be the key from the months dictionary

# Airport Code
airport_options = list(airport_codes.items())  # Creates a list of tuples
airport = st.sidebar.selectbox('Airport of Departure', options=airport_options, format_func=lambda x: x[0])
selected_airport = airport[1]  # This will be the key from the airport_codes dictionary

# Airline Code
airline_options = list(airline_codes.items())  # Creates a list of tuples
airline = st.sidebar.selectbox('Airline', options=airline_options, format_func=lambda x: x[0])
selected_airline = airline[1]  # This will be the key from the airline_codes dictionary

#Input
#Generate Statistics
submit = st.sidebar.button('Generate Statistics')

#Display on right
if(submit):
    # Filter and analyze data based on user input
    filtered_data = delay_analysis.filter_data(selected_month, selected_airline, selected_airport)
    delay_proportion, highest_support_value = delay_analysis.calculate_probabilities(filtered_data)


    data = pd.DataFrame({
        'Event': ['Probability'],
        'Probability': str(round(delay_proportion, 2)) + '%'
    })

    # Create a chart
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Probability', scale=alt.Scale(domain=[0, 1]), axis = alt.Axis(labels=False, title=None)),
        y='Event'
    )

    col1, col2 = st.columns([1, 1])
    col1.markdown(str(round(delay_proportion, 2)) + '%')
    col1.write("Probability of Delay")
    col1.altair_chart(chart, use_container_width=True)

    col2.markdown(highest_support_value)
    col2.markdown("Most likely delay type and length")

else:
    st.write("Please enter in your flight information on the side bar to the left")
    
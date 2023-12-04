import streamlit as st
import altair as alt
import pandas as pd
from models.apriori import FlightDelayApriori

# create instance
delay_analysis = FlightDelayApriori()

st.markdown("# Apriori")
st.sidebar.header("Apriori")

##Input
#Enter Month
month = st.sidebar.selectbox('Select Month of flight',st.session_state['months'].values())
#Airport Code
airport = st.sidebar.selectbox('Airport of Departure', st.session_state['airport_codes'])
#Airline Code
airline = st.sidebar.selectbox('Airline', st.session_state['airline_codes'])
#Generate Statistics
submit = st.sidebar.button('Generate Statistics')

#Display on right
if(submit):
    # Filter and analyze data based on user input
    filtered_data = delay_analysis.filter_data(month, airline, airport)
    delay_proportion, highest_support_value = delay_analysis.calculate_probabilities(filtered_data)


    data = pd.DataFrame({
        'Event': ['Probability'],
        'Probability': [delay_proportion / 100]
    })

    # Create a chart
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Probability', scale=alt.Scale(domain=[0, 1]), axis = alt.Axis(labels=False, title=None)),
        y='Event'
    )

    col1,col2 = st.columns([1,1])
    col1.markdown("# 50%")
    col1.write("Probability of Delay")
    col1.altair_chart(chart,use_container_width=True )

    col2.markdown("# 12 minutes")
    col2.markdown("Most likely delay length")
    col2.divider()
    col2.markdown("# Carrier")
    col2.markdown("Most likely delay type")
else:
    st.write("Please enter in your flight information on the side bar to the left")
    
import streamlit as st
import altair as alt
import pandas as pd

st.markdown("# Apriori")
st.sidebar.header("Apriori")



##Input
#Enter Month
month = st.sidebar.selectbox('Select Month of flight',st.session_state['months'].values())
#Airport Code
airport = st.sidebar.selectbox('Airport of Departure', st.session_state['airport_codes'])
#Generate Statistics
submit = st.sidebar.button('Generate Statistics')

#Display on right
if(submit):
    # test_data = {'Probability of Delay': 50}
    # st.bar_chart(test_data)

    #Convert input

    #Run model

    data = pd.DataFrame({
        'Event': ['Probability'],
        'Probability': [.50]
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
    
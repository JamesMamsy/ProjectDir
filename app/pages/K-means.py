import streamlit as st
st.markdown("# K-Means")
st.sidebar.header("K-means")

#Carrier Delay
carrier_delay = st.sidebar.number_input("Carrier Delay (Minutes)", value = None, placeholder = "Enter a number")

#Arrival Delay
dept_delay = st.sidebar.number_input("Arrival Delay (Minutes)", value = None, placeholder="Enter a number")

#Submit 
cluster = st.sidebar.button("Estimate Total Delay")

if(cluster and (carrier_delay == None or dept_delay == None) ):
    st.sidebar.write("Please enter in a value")
elif(cluster):

    #avg_delay = ???
    #
    st.markdown("## Estimated Total Delay:")
    st.markdown("(Average Cluster Delay )")
    st.markdown("### 35 Minutes")
    st.markdown("## Cluster: ")
    st.markdown("### Carrier:")
    st.markdown("### Arrival:")



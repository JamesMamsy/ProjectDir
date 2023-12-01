import streamlit as st
import pandas as pd
from models.kmeans import KMeans
import numpy as np


# Function to load and preprocess data
def load_data():
    # Load the flights DataFrame
    flights_df = pd.read_csv("flights.csv")

    # Select only the "DEP_DELAY" and "ARR_DELAY" columns and drop NaN values
    flights_df = flights_df[["DEP_DELAY", "ARR_DELAY"]].dropna()

    # Take only 10% of the DataFrame for testing
    flights_df = flights_df.sample(frac=0.01, random_state=42)

    return flights_df


def main():
    st.title("KMeans Clustering App")

    df = load_data()

    # Get user input for DEP_DELAY and ARR_DELAY
    dep_delay = st.number_input("Enter DEP_DELAY:", min_value=0.0)
    arr_delay = st.number_input("Enter ARR_DELAY:", min_value=0.0)

    kmeans = KMeans(k=4)

    # Check if the user has pressed the "PREDICT CLUSTER" button
    if st.button("PREDICT CLUSTER"):
        kmeans.fit(df)

        # Predict cluster for the user input
        predicted_cluster = kmeans.predict(dep_delay, arr_delay, df)

        st.success(f"Predicted Cluster: {predicted_cluster}")

    # Check if the user has pressed the "SHOW CLUSTERS" button
    if kmeans is not None and st.button("SHOW CLUSTERS"):
        kmeans.fit(df)
        # Show cluster information without fitting again
        cluster_info_output = kmeans.cluster_info(df)
        for info_line in cluster_info_output:
            st.write(info_line)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from models.kmeans import KMeans


# Function to load and preprocess data for clustering
def load_data():
    # load the df
    flights_df = pd.read_csv(r"data\flights.csv")

    # select columns to do clustering on and drop missing values
    flights_df = flights_df[["DEP_DELAY", "ARR_DELAY"]].dropna()

    return flights_df


def main():
    st.title("Airline delay app")

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
    if kmeans is not None and st.button("SHOW CLUSTERS INFO"):
        kmeans.fit(df)
        # Show cluster information without fitting again
        cluster_info_output = kmeans.cluster_info(df)
        for info_line in cluster_info_output:
            st.write(info_line)


if __name__ == "__main__":
    main()

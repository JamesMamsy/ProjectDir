import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import streamlit as st


# months_pattern=r'C:\Users\al105\OneDrive\Desktop\OU\Data Mining\hehe\Data Files-20231127T233339Z-001\Data Files\*.csv'
# all_months=glob.glob(months_pattern)

# flights_df=[]

# for month in all_months:
#     df=pd.read_csv(month)
#     print("1 file is done")
#     flights_df.append(df)


# flights_df=pd.concat(flights_df,ignore_index=True)

# flights_df.to_csv(r'C:\Users\al105\OneDrive\Desktop\OU\Data Mining\hehe\flights.csv',index=False)

# print(flights_df.columns)


class KMeans:
    def __init__(self, k, max_iters=25):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, data):
        centroids = data.sample(self.k)
        return centroids.values

    def assign_to_clusters(self, data):
        # calc distances from centroids
        distances = np.linalg.norm(
            data.values[:, np.newaxis, :] - self.centroids, axis=2
        )
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, labels):
        new_centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            new_centroids[i] = np.mean(data[labels == i], axis=0)
        return new_centroids

    def fit(self, data):
        self.centroids = self.initialize_centroids(data)

        for _ in range(self.max_iters):
            labels = self.assign_to_clusters(data)
            new_centroids = self.update_centroids(data.values, labels)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels

    def calculate_inertia(self, data):
        # calculate the internal distances and later used for elbow method testing
        labels = self.assign_to_clusters(data)
        distances = np.linalg.norm(data.values - self.centroids[labels], axis=1)
        return np.sum(distances**2)

    def calculate_silhouette(self, data):
        # calc silhouette score for similarity test of clusters
        labels = self.assign_to_clusters(data)

        # internal cluster distance
        a_values = np.zeros(data.shape[0])
        for i in range(self.k):
            cluster_points = data[labels == i]
            for j in range(data.shape[0]):
                if labels[j] == i:
                    a_values[j] = np.mean(
                        np.linalg.norm(cluster_points - data.iloc[j].values, axis=1)
                    )

        # cluster x to cluster y distance
        b_values = np.zeros(data.shape[0])
        for i in range(self.k):
            for j in range(data.shape[0]):
                if labels[j] != i:
                    cluster_points = data[labels == i]
                    b_values[j] = np.min(
                        np.linalg.norm(cluster_points - data.iloc[j].values, axis=1)
                    )

        # Calculate silhouette score for each sample
        silhouette_values = (b_values - a_values) / np.maximum(a_values, b_values)

        # Return the mean silhouette score for the entire dataset
        return np.mean(silhouette_values)

    def cluster_info(self, data):
        if self.centroids is None:
            raise ValueError(
                "KMeans model has not been fitted. Please call fit() before using this method."
            )

        labels = self.assign_to_clusters(data)

        # to make streamlit formatting better
        cluster_info_output = []

        for i in range(self.k):
            cluster_points = data[labels == i]
            centroid = self.centroids[i]

            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            cluster_info_output.append(f"Cluster {i + 1} Info:")
            cluster_info_output.append(f"Centroid: {centroid}")
            cluster_info_output.append(
                f"Average Distance to Centroid: {np.mean(distances)}"
            )
            cluster_info_output.append("")

        return cluster_info_output

    def predict(self, dep_delay, arr_delay, dataframe):
        # Assign data points in the dataframe to clusters
        self.assign_to_clusters(dataframe)
        # Create a new array with DEP_DELAY and ARR_DELAY for prediction
        point = np.array([dep_delay, arr_delay])

        # Calculate distances from centroids for the new point
        distances = np.linalg.norm(point - self.centroids, axis=1)

        # Assign the new point to the closest cluster
        predicted_cluster = np.argmin(distances)

        return predicted_cluster + 1


# flights_df = pd.read_csv(
#     r"C:\Users\al105\OneDrive\Desktop\OU\Data Mining\hehe\flights.csv"
# )

# flights_df = flights_df[["DEP_DELAY", "ARR_DELAY"]].dropna()
# flights_df = flights_df.sample(frac=0.01, random_state=42)
# # print(flights_df.shape)
# kmeans = KMeans(k=4)
# kmeans.fit(flights_df)
# print(kmeans.cluster_info(flights_df))

# cluster_index = kmeans.predict(0, 0, flights_df)
# print(f"The point belongs to cluster: {cluster_index}")

# testing for best K, K=4 was the best
# k_values = range(2, 8)

# # Calculate inertia for each K
# inertia_values = []
# for k in k_values:
#     kmeans_model = KMeans(k)
#     labels = kmeans_model.fit(flights_df)
#     inertia = kmeans_model.calculate_inertia(flights_df)
#     inertia_values.append(inertia)

# # Plot the elbow curve
# plt.plot(k_values, inertia_values, marker="*")
# plt.title("Elbow method")
# plt.xlabel("K")
# plt.ylabel("SSE")
# plt.show()

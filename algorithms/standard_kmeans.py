import time
import random
import numpy as np
from utils.math_utils import euclidean_distance
from utils.plot_utils import plot_clusters_pca, plot_convergence, plot_reassignments



class StandardKMeans:
    """
    Implements the standard K-Means clustering algorithm.
    """

    def __init__(self, data, k, max_iter, epsilon):
        """
        Initializes the StandardKMeans instance with the given parameters.
        :param data: The dataset to be clustered, expected to be a list of data points (each data point is a list of features).
        :param k: The number of clusters to form.
        :param max_iter: The maximum number of iterations to run the algorithm.
        :param epsilon: The convergence threshold for centroid movement. If the maximum shift of any centroid is less than epsilon, the algorithm will stop.
        """
        self.data = np.array(data)
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.centroids = None
        self.assignments = [-1] * len(data)
        self.sse_history = []
        self.reassignment_history = []

    def run(self):
        """
        Runs the K-Means algorithm, performing the steps of initializing centroids, assigning points to clusters,
        updating centroids, and checking for convergence. It also tracks the sum of squared errors (SSE) and the number of
        reassigned points at each iteration for visualization purposes.
        """
        self.initialize_centroids()
        start_time = time.time()

        # Initial assignment of points to clusters
        for iteration in range(self.max_iter):
            reassigned = self.assign_points()
            shift = self.update_centroids()
            sse = self.compute_sse()

            self.sse_history.append(sse)
            self.reassignment_history.append(reassigned)

            print(f"Iteration {iteration} SSE = {sse}")

            # Check for convergence based on centroid shift
            if shift < self.epsilon:
                print(f"Converged at iteration {iteration}")
                break

        # Calculate total runtime for the algorithm
        total_runtime = time.time() - start_time
        print(f"Total runtime for k={self.k}: {total_runtime:.4f} seconds")

        # Visualize results
        plot_clusters_pca(self.data, self.assignments, self.centroids, f"Standard K-Means Clusters (k={self.k})")
        plot_convergence(self.sse_history, f"Standard K-Means Convergence (k={self.k})")
        plot_reassignments(self.reassignment_history, f"Standard K-Means Reassignments (k={self.k})")

    def initialize_centroids(self):
        """
        Initializes the centroids by randomly selecting k data points from the dataset.
        """
        indices = random.sample(range(len(self.data)), self.k)
        self.centroids = np.array([self.data[i].copy() for i in indices])

    def assign_points(self):
        """
        Assigns each data point to the nearest centroid, updating the cluster assignments. It also counts how many points
        were reassigned to a different cluster compared to the previous iteration.
        :return: The number of points that were reassigned to a different cluster in this iteration.
        """
        reassigned_count = 0

        for i in range(len(self.data)):
            min_dist = float("inf")
            best_cluster = -1

            # Find the nearest centroid for the current data point
            for j in range(self.k):
                dist = euclidean_distance(self.data[i], self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j

            # Check if the assignment has changed from the previous iteration
            if self.assignments[i] != best_cluster:
                reassigned_count += 1

            # Update the assignment for the current data point
            self.assignments[i] = best_cluster

        return reassigned_count

    def update_centroids(self):
        """
        Updates the centroids by calculating the mean of the data points assigned to each cluster. It also computes the
        maximum shift of any centroid from its previous position to determine if the algorithm has converged.
        :return: The maximum shift of any centroid from its previous position after the update.
        """
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.k)

        # Sum up the data points for each cluster and count how many points are assigned to each cluster
        for i in range(len(self.data)):
            c = self.assignments[i]
            counts[c] += 1
            new_centroids[c] += self.data[i]

        # Calculate the mean for each cluster to get the new centroids
        for i in range(self.k):
            if counts[i] > 0:
                new_centroids[i] /= counts[i]

        # Calculate the maximum shift of any centroid from its previous position
        max_shift = 0.0
        for i in range(self.k):
            shift = euclidean_distance(self.centroids[i], new_centroids[i])
            max_shift = max(max_shift, shift)

        # Update the centroids to the new positions
        self.centroids = new_centroids
        return max_shift

    def compute_sse(self):
        """
        Computes the sum of squared errors (SSE) for the current cluster assignments.
        :return: The computed SSE value, rounded to 4 decimal places.
        """
        sse = 0.0
        for i in range(len(self.data)):
            c = self.assignments[i]
            dist = euclidean_distance(self.data[i], self.centroids[c])
            sse += dist * dist
        return round(sse, 4)

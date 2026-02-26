import time
import random
import numpy as np
from utils.math_utils import euclidean_distance
from utils.plot_utils import plot_clusters_pca, plot_reassignments, plot_convergence


class AlternateKMeans:
    """
    Implements an alternate K-Means clustering algorithm that, in addition to the standard steps of centroid initialization,
    assignment, and update, also includes a step to reassign the point furthest from its assigned centroid to a different
    cluster if it is closer to another centroid.
    """

    def __init__(self, data, k, max_iter, epsilon):
        """
        Initializes the AlternateKMeans instance with the given parameters.
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
        Runs the alternate K-Means algorithm, which includes the standard steps of centroid initialization, assignment, and update,
        as well as an additional step to reassign the point furthest from its assigned centroid to a different cluster
        if it is closer to another centroid. The method tracks the sum of squared errors (SSE) and the number of reassigned
        points at each iteration for visualization purposes.
        """
        start_time = time.time()
        self.initialize_centroids()
        self.assign_initial_points()

        # Initial assignment of points to clusters and iterative optimization
        for iteration in range(self.max_iter):
            shift = self.update_centroids()
            sse = self.compute_sse()
            reassigned = self.assign_farthest_point()
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
        plot_clusters_pca(self.data, self.assignments, self.centroids, f"Alternate K-Means Clusters (k={self.k})")
        plot_convergence(self.sse_history, f"Alternate K-Means Convergence (k={self.k})")
        plot_reassignments(self.reassignment_history, f"Alternate K-Means Reassignments (k={self.k})")

    def initialize_centroids(self):
        """
        Initializes centroids by randomly selecting k unique data points from the dataset.
        """
        indices = random.sample(range(len(self.data)), self.k)
        self.centroids = np.array([self.data[i].copy() for i in indices])

    def assign_initial_points(self):
        """
        Assigns each data point to the nearest centroid to establish the initial cluster assignments before the iterative optimization process begins.
        """
        for i in range(len(self.data)):
            min_dist = float("inf")
            best_cluster = -1

            # Find the nearest centroid for each data point
            for j in range(self.k):
                dist = euclidean_distance(self.data[i], self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j

            self.assignments[i] = best_cluster

    def update_centroids(self):
        """
        Updates the centroids by calculating the mean of the data points assigned to each cluster. It also calculates
        the maximum shift of any centroid from its previous position to determine if the algorithm has converged.
        :return: The maximum shift of any centroid from its previous position, which is used to check for convergence.
        """
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.k)

        # Accumulate the sum of points in each cluster and count the number of points in each cluster
        for i in range(len(self.data)):
            c = self.assignments[i]
            counts[c] += 1
            new_centroids[c] += self.data[i]

        # Calculate the mean for each cluster to update the centroids
        for i in range(self.k):
            if counts[i] > 0:
                new_centroids[i] /= counts[i]

        # Calculate the maximum shift of any centroid from its previous position to determine if the algorithm has converged
        max_shift = 0.0
        for i in range(self.k):
            shift = euclidean_distance(self.centroids[i], new_centroids[i])
            max_shift = max(max_shift, shift)

        self.centroids = new_centroids
        return max_shift

    def assign_farthest_point(self):
        """
        Reassigns the point that is furthest from its assigned centroid to a different cluster if it is closer to another centroid.
        :return: The number of points that were reassigned to a different cluster in this step.
        """
        reassigned_count = 0

        for cluster_idx in range(self.k):
            # Get indices of points in this cluster
            points_in_cluster = [i for i, c in enumerate(self.assignments) if c == cluster_idx]
            if not points_in_cluster:
                continue

            # Find the point furthest from the centroid
            max_dist = -1
            worst_point_idx = -1
            for i in points_in_cluster:
                dist = euclidean_distance(self.data[i], self.centroids[cluster_idx])
                if dist > max_dist:
                    max_dist = dist
                    worst_point_idx = i

            # Check if this point is closer to another centroid
            best_cluster = cluster_idx
            best_dist = max_dist
            for j in range(self.k):
                if j == cluster_idx:
                    continue
                dist_to_other = euclidean_distance(self.data[worst_point_idx], self.centroids[j])
                if dist_to_other < best_dist:
                    best_dist = dist_to_other
                    best_cluster = j

            # Reassign the point if a better cluster is found
            if best_cluster != cluster_idx:
                self.assignments[worst_point_idx] = best_cluster
                reassigned_count += 1

        return reassigned_count

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
import time
import numpy as np
from utils.math_utils import euclidean_distance
from utils.plot_utils import plot_clusters_pca, plot_convergence, plot_reassignments


class OptimizedKMeans:
    """
    Implements an optimized K-Means clustering algorithm that uses density-based initialization and efficient centroid updates.
    """

    def __init__(self, data, k, max_iter, epsilon, density_radius):
        """
        Initializes the OptimizedKMeans instance with the given parameters.
        :param data: The dataset to be clustered.
        :param k: The number of clusters to form.
        :param max_iter: The maximum number of iterations to run the algorithm.
        :param epsilon: The convergence threshold for centroid movement.
        :param density_radius: The radius used to calculate the density of points for centroid initialization.
        """
        self.data = np.array(data)
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.density_radius = density_radius  # radius for density calculation

        self.centroids = None
        self.assignments = [-1] * len(data)

        # Optimization structures
        self.cluster_sums = None
        self.cluster_counts = None
        self.point_density = None
        self.sse_history = []
        self.reassignment_history = []

    def run(self):
        """
        Runs the optimized K-Means algorithm, which includes density-based centroid initialization and efficient
        updates of centroids and cluster assignments.
        :return:
        """
        start_time = time.time()

        self.compute_point_density()
        self.initialize_centroids()
        self.initialize_clusters()

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
        plot_clusters_pca(self.data, self.assignments, self.centroids, f"Optimized K-Means Clusters (k={self.k})")
        plot_convergence(self.sse_history, f"Optimized K-Means Convergence (k={self.k})")
        plot_reassignments(self.reassignment_history, f"Optimized K-Means Reassignments (k={self.k})")

    def compute_point_density(self):
        """
        Computes the density of each point in the dataset based on the number of neighboring points within a specified radius.
        """
        n = len(self.data)
        self.point_density = np.zeros(n)
        for i in range(n):
            count = 0
            # Count neighbors within density_radius
            for j in range(n):
                if i != j and euclidean_distance(self.data[i], self.data[j]) <= self.density_radius:
                    count += 1
            self.point_density[i] = count

    def initialize_centroids(self):
        """
        Initializes the centroids using a density-based approach.
        """
        n = len(self.data)
        chosen = []

        # First centroid: point with highest density
        first_index = int(np.argmax(self.point_density))
        chosen.append(first_index)

        # Remaining centroids
        while len(chosen) < self.k:
            best_score = -1
            best_index = -1
            for i in range(n):
                if i in chosen:
                    continue
                # minimum distance to already chosen centroids
                min_dist = min(euclidean_distance(self.data[i], self.data[j]) for j in chosen)
                score = self.point_density[i] * min_dist
                if score > best_score:
                    best_score = score
                    best_index = i
            chosen.append(best_index)

        # Assign centroids
        self.centroids = np.array([self.data[i].copy() for i in chosen])

    def initialize_clusters(self):
        """
        Initializes the cluster sums and counts based on the initial centroids.
        Each data point is assigned to the nearest centroid, and the sums and counts for each cluster are updated accordingly.
        """
        d = self.data.shape[1]
        self.cluster_sums = np.zeros((self.k, d))
        self.cluster_counts = np.zeros(self.k)

        # Assign points to nearest centroid initially
        for i in range(len(self.data)):
            # Find nearest centroid
            distances = [euclidean_distance(self.data[i], c) for c in self.centroids]
            c = int(np.argmin(distances))
            self.assignments[i] = c
            self.cluster_sums[c] += self.data[i]
            self.cluster_counts[c] += 1

        # Compute initial centroids (already chosen, but correct sums/counts)
        for j in range(self.k):
            if self.cluster_counts[j] > 0:
                self.centroids[j] = self.cluster_sums[j] / self.cluster_counts[j]

    def assign_points(self):
        """
        Assigns each data point to the nearest centroid, updating the cluster assignments. It also counts how many points
        were reassigned to a different cluster compared to the previous iteration.
        :return: The number of points that were reassigned to a different cluster in this iteration.
        """
        reassigned_count = 0

        for i in range(len(self.data)):
            # Find nearest centroid
            min_dist = float("inf")
            best_cluster = -1
            for j in range(self.k):
                dist = euclidean_distance(self.data[i], self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j

            old_cluster = self.assignments[i]

            # If the assignment has changed, update the cluster sums and counts accordingly
            if old_cluster != best_cluster:
                reassigned_count += 1

                # Remove from old cluster
                if old_cluster != -1:
                    self.cluster_sums[old_cluster] -= self.data[i]
                    self.cluster_counts[old_cluster] -= 1

                # Add to new cluster
                self.cluster_sums[best_cluster] += self.data[i]
                self.cluster_counts[best_cluster] += 1

                self.assignments[i] = best_cluster

        return reassigned_count

    def update_centroids(self):
        """
        Updates the centroids by calculating the mean of the data points assigned to each cluster. It also computes the
        maximum shift of any centroid from its previous position to determine if the algorithm has converged.
        :return: The maximum shift of any centroid from its previous position after the update.
        """
        max_shift = 0.0
        for j in range(self.k):
            # Only update centroids for clusters that have points assigned to them
            if self.cluster_counts[j] > 0:
                new_centroid = self.cluster_sums[j] / self.cluster_counts[j]
                shift = euclidean_distance(self.centroids[j], new_centroid)
                max_shift = max(max_shift, shift)
                self.centroids[j] = new_centroid
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

import time
import random
import numpy as np
from utils.math_utils import euclidean_distance
from utils.plot_utils import plot_clusters_pca, plot_reassignments, plot_convergence


class AlternateKMeans:

    def __init__(self, data, k, max_iter, epsilon):
        self.data = np.array(data)
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.centroids = None
        self.assignments = [-1] * len(data)
        self.sse_history = []
        self.reassignment_history = []

    def run(self):
        start_time = time.time()
        self.initialize_centroids()
        self.assign_initial_points()

        for iteration in range(self.max_iter):
            shift = self.update_centroids()
            sse = self.compute_sse()
            reassigned = self.assign_farthest_point()
            self.sse_history.append(sse)
            self.reassignment_history.append(reassigned)

            print(f"Iteration {iteration} SSE = {sse}")

            if shift < self.epsilon:
                print(f"Converged at iteration {iteration}")
                break

        total_runtime = time.time() - start_time
        print(f"Total runtime for k={self.k}: {total_runtime:.4f} seconds")

        plot_clusters_pca(self.data, self.assignments, self.centroids, "Alternate K-Means Clusters")
        plot_convergence(self.sse_history, f"Alternate K-Means Convergence (k={self.k})")
        plot_reassignments(self.reassignment_history, f"Alternate K-Means Reassignments (k={self.k})")

    def initialize_centroids(self):
        indices = random.sample(range(len(self.data)), self.k)
        self.centroids = np.array([self.data[i].copy() for i in indices])

    def assign_initial_points(self):
        for i in range(len(self.data)):
            min_dist = float("inf")
            best_cluster = -1

            for j in range(self.k):
                dist = euclidean_distance(self.data[i], self.centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = j

            self.assignments[i] = best_cluster

    def update_centroids(self):
        new_centroids = np.zeros_like(self.centroids)
        counts = np.zeros(self.k)

        for i in range(len(self.data)):
            c = self.assignments[i]
            counts[c] += 1
            new_centroids[c] += self.data[i]

        for i in range(self.k):
            if counts[i] > 0:
                new_centroids[i] /= counts[i]

        max_shift = 0.0
        for i in range(self.k):
            shift = euclidean_distance(self.centroids[i], new_centroids[i])
            max_shift = max(max_shift, shift)

        self.centroids = new_centroids
        return max_shift

    def assign_farthest_point(self):
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
        sse = 0.0
        for i in range(len(self.data)):
            c = self.assignments[i]
            dist = euclidean_distance(self.data[i], self.centroids[c])
            sse += dist * dist
        return round(sse, 4)
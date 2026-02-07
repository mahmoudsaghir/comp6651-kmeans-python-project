import numpy as np
from utils.math_utils import euclidean_distance


class OptimizedKMeans:

    def __init__(self, data, k, max_iter, epsilon, density_radius):
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

    def run(self):
        self.compute_point_density()
        self.initialize_centroids()
        self.initialize_clusters()

        for iteration in range(self.max_iter):
            self.assign_points()
            shift = self.update_centroids()
            sse = self.compute_sse()

            print(f"Iteration {iteration} SSE = {sse}")

            if shift < self.epsilon:
                print(f"Converged at iteration {iteration}")
                break

    def compute_point_density(self):
        n = len(self.data)
        self.point_density = np.zeros(n)
        for i in range(n):
            count = 0
            for j in range(n):
                if i != j and euclidean_distance(self.data[i], self.data[j]) <= self.density_radius:
                    count += 1
            self.point_density[i] = count

    def initialize_centroids(self):
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
            if old_cluster != best_cluster:
                # Remove from old cluster
                if old_cluster != -1:
                    self.cluster_sums[old_cluster] -= self.data[i]
                    self.cluster_counts[old_cluster] -= 1
                # Add to new cluster
                self.cluster_sums[best_cluster] += self.data[i]
                self.cluster_counts[best_cluster] += 1
                self.assignments[i] = best_cluster

    def update_centroids(self):
        max_shift = 0.0
        for j in range(self.k):
            if self.cluster_counts[j] > 0:
                new_centroid = self.cluster_sums[j] / self.cluster_counts[j]
                shift = euclidean_distance(self.centroids[j], new_centroid)
                max_shift = max(max_shift, shift)
                self.centroids[j] = new_centroid
        return max_shift

    def compute_sse(self):
        sse = 0.0
        for i in range(len(self.data)):
            c = self.assignments[i]
            dist = euclidean_distance(self.data[i], self.centroids[c])
            sse += dist * dist
        return round(sse, 4)
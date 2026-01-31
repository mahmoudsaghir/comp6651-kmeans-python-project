import random
import numpy as np
from utils.math_utils import euclidean_distance


class StandardKMeans:

    def __init__(self, data, k, max_iter, epsilon):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.epsilon = epsilon

        self.centroids = None
        self.assignments = [-1] * len(data)

    def run(self):
        self.initialize_centroids()

        for iteration in range(self.max_iter):
            self.assign_points()
            shift = self.update_centroids()
            sse = self.compute_sse()

            print(f"Iteration {iteration} SSE = {sse}")

            if shift < self.epsilon:
                print(f"Converged at iteration {iteration}")
                break

    def initialize_centroids(self):
        indices = random.sample(range(len(self.data)), self.k)
        self.centroids = np.array([self.data[i].copy() for i in indices])

    def assign_points(self):
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

    def compute_sse(self):
        sse = 0.0
        for i in range(len(self.data)):
            c = self.assignments[i]
            dist = euclidean_distance(self.data[i], self.centroids[c])
            sse += dist * dist
        return round(sse, 4)
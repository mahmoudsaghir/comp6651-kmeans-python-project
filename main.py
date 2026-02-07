import time

from algorithms.optimized_kmeans import OptimizedKMeans
from config.config import Config
from data.dataset_loader import load_csv
from algorithms.standard_kmeans import StandardKMeans

def main():
    config = Config("config.txt")

    dataset_path = config.get_string("dataset")
    k = config.get_int("k")
    max_iter = config.get_int("maxIter")
    epsilon = config.get_double("epsilon")
    density_radius = config.get_double("densityRadius")

    data = load_csv(dataset_path)

    standard_kmeans = StandardKMeans(data, k, max_iter, epsilon)
    print("Running Standard K-Means:")
    standard_kmeans.run()

    optimized_kmeans = OptimizedKMeans(data, k, max_iter, epsilon, density_radius)
    print("\nRunning Optimized K-Means:")
    optimized_kmeans.run()

if __name__ == "__main__":
    main()
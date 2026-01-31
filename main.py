from config.config import Config
from data.dataset_loader import load_csv
from algorithms.standard_kmeans import StandardKMeans

def main():
    config = Config("config.txt")

    dataset_path = config.get_string("dataset")
    k = config.get_int("k")
    max_iter = config.get_int("maxIter")
    epsilon = config.get_double("epsilon")

    data = load_csv(dataset_path)

    kmeans = StandardKMeans(data, k, max_iter, epsilon)
    kmeans.run()

if __name__ == "__main__":
    main()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_clusters_pca(data, assignments, centroids, title):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids)

    plt.figure()

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=assignments)

    plt.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        marker='x',
        s=200
    )

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def plot_convergence(sse_history, title):
    plt.figure()

    plt.plot(range(len(sse_history)), sse_history, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Clustering Objective (SSE)")
    plt.title(title)
    plt.grid()

    plt.show()

def plot_reassignments(reassignment_history, title):
    plt.figure()

    plt.plot(range(len(reassignment_history)), reassignment_history, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Number of Reassigned Points")
    plt.title(title)
    plt.grid()

    plt.show()
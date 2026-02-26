from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_clusters_pca(data, assignments, centroids, title):
    """
    Plots the clusters in a 2D space using PCA for dimensionality reduction.
    :param data: The original high-dimensional data points.
    :param assignments: The cluster assignments for each data point, which will be used to color the points in the plot.
    :param centroids: The centroids of the clusters, which will be plotted as 'x' markers on the graph.
    :param title: The title for the plot, which will be displayed at the top of the graph.
    """
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
    """
    Plots the convergence of the clustering algorithm by showing the sum of squared errors (SSE) over iterations.
    :param sse_history: A list of SSE values recorded at each iteration of the clustering algorithm, which will be plotted on the y-axis.
    :param title: The title for the plot, which will be displayed at the top of the graph. This should indicate which algorithm and parameters were used for clarity.
    """
    plt.figure()

    plt.plot(range(len(sse_history)), sse_history, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Clustering Objective (SSE)")
    plt.title(title)
    plt.grid()

    plt.show()

def plot_reassignments(reassignment_history, title):
    """
    Plots the number of reassigned points at each iteration of the clustering algorithm to visualize how the algorithm
    is progressing towards convergence.
    :param reassignment_history: A list of integers representing the number of data points that were reassigned to different clusters at each iteration.
    :param title: The title for the plot, which will be displayed at the top of the graph.
    """
    plt.figure()

    plt.plot(range(len(reassignment_history)), reassignment_history, marker="o")

    plt.xlabel("Iteration")
    plt.ylabel("Number of Reassigned Points")
    plt.title(title)
    plt.grid()

    plt.show()
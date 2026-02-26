import math
import numpy as np

def euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two points a and b
    :param a: First point
    :param b: Second point
    :return: The Euclidean distance between points a and b, calculated as the square root of the sum of squared differences between corresponding features.
    """
    sum_sq = 0.0
    for i in range(len(a)):
        if np.isnan(a[i]) or np.isnan(b[i]):
            continue
        diff = a[i] - b[i]
        sum_sq += diff * diff
    return math.sqrt(sum_sq)
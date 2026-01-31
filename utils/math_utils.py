import math
import numpy as np

def euclidean_distance(a, b):
    sum_sq = 0.0
    for i in range(len(a)):
        if np.isnan(a[i]) or np.isnan(b[i]):
            continue
        diff = a[i] - b[i]
        sum_sq += diff * diff
    return math.sqrt(sum_sq)
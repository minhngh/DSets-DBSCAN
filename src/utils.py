import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from contextlib import contextmanager
from time import time

@contextmanager
def timer():
    start = time()
    yield
    end = time()
    print(f'Elapsed time: {end - start} s')

def get_pair_distance(a, b):
    return np.sqrt(np.sum((a[:-1] - b[:-1])**2))

def get_distances(X, metric = 'euclidean'):
    X = X[:, :-1]
    return pairwise_distances(X, metric = metric)

def get_similarity(x, distance_fn = pairwise_distances):
    x = x[:, :-1]
    dist = pairwise_distances(x, metric = 'sqeuclidean')
    sigma = np.median(dist)
    return np.exp(-dist / sigma)

def visualize(data, clusters):
    if data.shape[1] != 2:
        print("Only visualize 2d-data")
        return
    plt.figure(figsize = (15, 15))
    clusters = sorted(clusters, key = lambda x: x[1])
    for i, (x, _ ) in enumerate(clusters):
        x = x.astype(np.int)
        plt.scatter(data[x, 0], data[x, 1], marker = '^' if i == 0 else None)
    plt.show()

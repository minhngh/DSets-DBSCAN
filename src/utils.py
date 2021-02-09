import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import contextmanager
from time import time

@contextmanager
def timer():
    start = time()
    yield
    end = time()
    print(f'Elapsed time: {end - start} s')

def get_pair_distance(a, b, metric = 'euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.sum((a[:-1] - b[:-1])**2))
    elif metric == 'cosine':
        return np.sum(a[:-1] * b[:-1]) / (np.linalg.norm(a[:-1]) * np.linalg.norm(b[:-1]))

def get_distances(X, metric = 'euclidean'):
    X = X[:, :-1]
    if metric == 'euclidean':
        return pairwise_distances(X, metric = metric)
    elif metric == 'cosine':
        return cosine_similarity(X)

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

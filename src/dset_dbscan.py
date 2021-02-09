import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity

from .components import equalize_hist, dominant_set, expand_cluster
from .utils import get_distances, get_similarity
def get_eps_in_dset(dset, min_points, metric = 'euclidean'):
    dists = get_distances(dset, metric = metric)
    if dset.shape[0] <= min_points:
        if metric == 'euclidean': return dists.max()
        elif metric == 'cosine': return dists.min()
    if metric == 'euclidean':
        return reduce(lambda eps, ele: max(eps, np.sort(ele)[min_points]), dists, float('-inf'))
    elif metric == 'cosine':
        return reduce(lambda eps, ele: min(eps, np.sort(ele)[::-1][min_points]), dists, 1)
        

def dset_dbscan(X, min_points, cut_off = 2e-4, metric = 'euclidean'):
    X = np.hstack((X, np.arange(X.shape[0]).reshape(-1, 1)))
    if metric == 'euclidean':
        A = get_similarity(X)
        A = equalize_hist(A)
    elif metric == 'cosine':
        A = cosine_similarity(X[:, :-1])
    else:
        raise ValueError("This metric isn't supported")
    results = []
    cluster_id = 0
    while A.size > 0:
        x = dominant_set(A)
        dset_idxs = np.where(x > cut_off)[0]
        epsilon = get_eps_in_dset(X[dset_idxs], min_points, metric = metric)
        # print the number of the dominant set
        print('thresh:', epsilon, ',', 'dominant set:', len(dset_idxs))
        cluster = expand_cluster(X, dset_idxs[0], epsilon, min_points, metric = metric)
        if not cluster:
            cluster = dset_idxs.tolist()
            current_cluster_id = -1
        else:
            current_cluster_id = cluster_id
            cluster_id += 1
            idxs = X[cluster, -1]
            results.append((idxs, current_cluster_id))        
        A = np.delete(np.delete(A, cluster, axis = 0), cluster, axis = 1)
        if current_cluster_id != -1:
            X = np.delete(X, cluster, axis = 0)
    return results
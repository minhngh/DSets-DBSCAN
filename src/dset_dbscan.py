import numpy as np
from functools import reduce

from .components import equalize_hist, dominant_set, expand_cluster
from .utils import get_distances, get_similarity
def get_eps_in_dset(dset, min_points, metric = 'euclidean'):
    dists = get_distances(dset, metric = metric)
    if dset.shape[0] <= min_points:
        return dists.max()
    return reduce(lambda eps, env: max(eps, np.sort(env[1][:env[0]])[min_points] if env[0] > min_points else env[1].max()), enumerate(dists), float('-inf'))
        

def dset_dbscan(X, min_points, cut_off = 2e-4, metric = 'euclidean'):
    X = np.hstack((X, np.arange(X.shape[0]).reshape(-1, 1)))
    results = []
    A = get_similarity(X)
    A = equalize_hist(A)
    cluster_id = 0
    while A.size > 0:
        x = dominant_set(A)
        dset_idxs = np.where(x > cut_off)[0]
        epsilon = get_eps_in_dset(X[dset_idxs], min_points, metric = metric)
        cluster = expand_cluster(X, dset_idxs[0], epsilon, min_points)
        print(epsilon, len(cluster))
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
import numpy as np
import math
import cv2 as cv
from .utils import get_pair_distance

def equalize_hist(A):
    A = (A * 255).astype(np.uint8)
    A = cv.equalizeHist(A)
    return A / 255.
def dominant_set(A, x = None, epsilon = 1e-4):
    if not x:
        x = np.ones(A.shape[0]) / A.shape[0]
    dist = float('inf')
    while dist > epsilon:
        x_old = x
        x = x * A.dot(x)
        x = x / x.sum()
        dist = np.linalg.norm(x - x_old)
    return x

def is_neighbor(p, q, epsilon):
    return get_pair_distance(p, q) < epsilon
def region_query(x, point_id, epsilon):
    return [i for i in range(x.shape[0]) if is_neighbor(x[point_id], x[i], epsilon) and point_id != i]
def expand_cluster(x, point_id, epsilon, min_points):
    seeds = region_query(x, point_id, epsilon)
    if len(seeds) < min_points:
        return []
    explored = [0] * x.shape[0]
    result = []
    for seed in seeds:
        explored[seed] = 1
    result.extend(seeds)
    while seeds:
        current_point = seeds[0]
        neighbors = region_query(x, current_point, epsilon)
        if len(neighbors) >= min_points:
            for neighbor in neighbors:
                if explored[neighbor] == 0:
                    seeds.append(neighbor)
                    result.append(neighbor)
                    explored[neighbor] = 1
        seeds = seeds[1:]
    return result


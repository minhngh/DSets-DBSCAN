import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from src import dset_dbscan, timer, visualize

def get_configs():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type = str, required = True, help = 'path of the dataset')
    ap.add_argument('--min-pts', type = int, default = 3, help = 'min points')
    ap.add_argument('--cut-off', type = float, default = 2e-4)
    ap.add_argument('--metric', type = str, default = 'euclidean', help = 'metric to calculate the distance')
    ap.add_argument('--visualize', type = bool, default = True, help = 'visualize the result or not')
    return ap.parse_args()
if __name__ == "__main__":
    args = get_configs()
    data_path = args.dataset
    assert os.path.exists(data_path), 'File doesn\'t exist'
    assert os.path.basename(data_path).split('.')[-1] == 'npy', "Don't support this format"

    data = np.load(data_path)
    with timer():
        result = dset_dbscan(data, args.min_pts, args.cut_off, args.metric)
    if args.visualize:
        visualize(data, result)

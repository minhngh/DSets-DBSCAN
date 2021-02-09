### DSets-DBSCAN Algorithm
***
This is an implementation described in the paper ["DSets-DBSCAN: A Parameter-Free Clustering Algorithm"](https://ieeexplore.ieee.org/document/7460951). 
***
#### Environment
The code uses Python 3 and some libraries: sklearn, numpy, matplotlib, opencv
#### Usage
Clone this repository and then run with:
```
    python main.py --dataset [dataset-path]
                   --min-pts [default 3]
                   --cut-off [default 2e-4]
                   --metric [default euclidean]
                   --visualize [default True]
```
#### Documentation
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
|   dset_dbscan(X, min_points, cut_off = 2e-4, metric = 'euclidean')                      |
|   For more details, you should read the paper                                           |
|     Inputs:                                                                             |
|       X: array of the points needing clustering                                         |
|       min_points: minimum number of the points in epsilon-neighborhood of a core point  |
|       cut-off: a threshold to define a dominant set                                     |
|       metric: use when calculating the distance. It must be 'euclidean' or 'cosine'.                                              |
└─────────────────────────────────────────────────────────────────────────────────────────┘
```
#### Tests
Code in **main.ipynb**
```
    1. cut_off = 2e-4
    Size of dominant set: 225, 99, 3, 3, 2, 2
    Clustering result:
        cluster_id | the number of the elements
        0          | 1153
        1          | 1337
        2          | 3796
        3          | 173
    2. cut_off = 1e-4
    Size of dominant set: 335, 192, 195, 103, 3, 3, 1, 2
    Clustering result:
        cluster_id | the number of the elements
        0          | 1966
        1          | 98
        2          | 95
        3          | 4064
        4          | 231
    3. cut_off = 1e-5
    Size of dominant set: 414, 293, 399, 202, 203, 204, 105, 3, 2
    Clustering result:
        cluster_id | the number of the elements
        0          | 1159
        1          | 190
        2          | 197
        3          | 308
        4          | 104
        5          | 212
        6          | 4268
        7          | 21
```
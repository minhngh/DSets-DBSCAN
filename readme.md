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
|       metric: for calculating the distance                                              |
└─────────────────────────────────────────────────────────────────────────────────────────┘
```
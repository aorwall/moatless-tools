**0 - /tmp/repos/scikit-learn/sklearn/cluster/optics_.py**:
```python
# -*- coding: utf-8 -*-
"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.

Authors: Shane Grigsby <refuge@rocktalus.com>
         Amy X. Zhang <axz@mit.edu>
License: BSD 3 clause
"""

from __future__ import division
import warnings
import numpy as np

from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator, ClusterMixin
from ..metrics.pairwise import pairwise_distances
from ._optics_inner import quick_scan


def optics(X, min_samples=5, max_bound=np.inf, metric='euclidean',
           p=2, metric_params=None, maxima_ratio=.75,
           rejection_ratio=.7, similarity_threshold=0.4,
           significant_min=.003, min_cluster_size_ratio=.005,
           min_maxima_ratio=0.001, algorithm='ball_tree',
           leaf_size=30, n_jobs=1):
    """Perform OPTICS clustering from vector array

    OPTICS: Ordering Points To Identify the Clustering Structure
    Equivalent to DBSCAN, finds core sample of high density and expands
    clusters from them. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Optimized for usage on large point datasets.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data.

    min_samples : int
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    max_bound : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood. This is also the largest object size
        expected within the dataset. Default value of "np.inf" will identify
        clusters across all scales; reducing `max_bound` will result in
        shorter run times.

    metric : string or callable, optional
        The distance metric to use for neighborhood lookups. Default is
        "minkowski". Other options include "euclidean", "manhattan",
        "chebyshev", "haversine", "seuclidean", "hamming", "canberra",
        and "braycurtis". The "wminkowski" and "mahalanobis" metrics are
        also valid with an additional argument.

    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        :ref:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function.

    maxima_ratio : float, optional
        The maximum ratio we allow of average height of clusters on the
        right and left to the local maxima in question. The higher the
        ratio, the more generous the algorithm is to preserving local
        minima, and the more cuts the resulting tree will have.

    rejection_ratio : float, optional
        Adjusts the fitness of the clustering. When the maxima_ratio is
        exceeded, determine which of the clusters to the left and right to
        reject based on rejection_ratio. Higher values will result in points
        being more readily classified as noise; conversely, lower values will
        result in more points being clustered.

    similarity_threshold : float, optional
        Used to check if nodes can be moved up one level, that is, if the
        new cluster created is too "similar" to its parent, given the
        similarity threshold. Similarity can be determined by 1) the size
        of the new cluster relative to the size of the parent node or
        2) the average of the reachability values of the new cluster
        relative to the average of the reachability values of the parent
        node. A lower value for the similarity threshold means less levels
        in the tree.

    significant_min : float, optional
        Sets a lower threshold on how small a significant maxima can be.

    min_cluster_size_ratio : float, optional
        Minimum percentage of dataset expected for cluster membership.

    min_maxima_ratio : float, optional
        Used to determine neighborhood size for minimum cluster membership.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default=30)
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    n_jobs : int, optional (default=1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Returns
    -------
    core_sample_indices_ : array, shape (n_core_samples,)
        The indices of the core samples.

    labels_ : array, shape (n_samples,)
        The estimated labels.

    References
    ----------
    Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander.
    "OPTICS: ordering points to identify the clustering structure." ACM SIGMOD
    Record 28, no. 2 (1999): 49-60.
    
```

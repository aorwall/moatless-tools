# scikit-learn__scikit-learn-25752

| **scikit-learn/scikit-learn** | `b397b8f2d952a26344cc062ff912c663f4afa6d5` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 50311 |
| **Any found context length** | 792 |
| **Avg pos** | 288.3333333333333 |
| **Min pos** | 2 |
| **Max pos** | 97 |
| **Top file pos** | 1 |
| **Missing snippets** | 26 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/cluster/_bicluster.py b/sklearn/cluster/_bicluster.py
--- a/sklearn/cluster/_bicluster.py
+++ b/sklearn/cluster/_bicluster.py
@@ -487,7 +487,7 @@ class SpectralBiclustering(BaseSpectral):
     >>> clustering.row_labels_
     array([1, 1, 1, 0, 0, 0], dtype=int32)
     >>> clustering.column_labels_
-    array([0, 1], dtype=int32)
+    array([1, 0], dtype=int32)
     >>> clustering
     SpectralBiclustering(n_clusters=2, random_state=0)
     """
diff --git a/sklearn/cluster/_bisect_k_means.py b/sklearn/cluster/_bisect_k_means.py
--- a/sklearn/cluster/_bisect_k_means.py
+++ b/sklearn/cluster/_bisect_k_means.py
@@ -190,18 +190,18 @@ class BisectingKMeans(_BaseKMeans):
     --------
     >>> from sklearn.cluster import BisectingKMeans
     >>> import numpy as np
-    >>> X = np.array([[1, 2], [1, 4], [1, 0],
-    ...               [10, 2], [10, 4], [10, 0],
-    ...               [10, 6], [10, 8], [10, 10]])
+    >>> X = np.array([[1, 1], [10, 1], [3, 1],
+    ...               [10, 0], [2, 1], [10, 2],
+    ...               [10, 8], [10, 9], [10, 10]])
     >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
     >>> bisect_means.labels_
-    array([2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=int32)
+    array([0, 2, 0, 2, 0, 2, 1, 1, 1], dtype=int32)
     >>> bisect_means.predict([[0, 0], [12, 3]])
-    array([2, 0], dtype=int32)
+    array([0, 2], dtype=int32)
     >>> bisect_means.cluster_centers_
-    array([[10.,  2.],
-           [10.,  8.],
-           [ 1., 2.]])
+    array([[ 2., 1.],
+           [10., 9.],
+           [10., 1.]])
     """
 
     _parameter_constraints: dict = {
@@ -309,7 +309,12 @@ def _bisect(self, X, x_squared_norms, sample_weight, cluster_to_bisect):
         # Repeating `n_init` times to obtain best clusters
         for _ in range(self.n_init):
             centers_init = self._init_centroids(
-                X, x_squared_norms, self.init, self._random_state, n_centroids=2
+                X,
+                x_squared_norms=x_squared_norms,
+                init=self.init,
+                random_state=self._random_state,
+                n_centroids=2,
+                sample_weight=sample_weight,
             )
 
             labels, inertia, centers, _ = self._kmeans_single(
@@ -361,7 +366,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable.
 
         Returns
         -------
diff --git a/sklearn/cluster/_kmeans.py b/sklearn/cluster/_kmeans.py
--- a/sklearn/cluster/_kmeans.py
+++ b/sklearn/cluster/_kmeans.py
@@ -63,13 +63,20 @@
     {
         "X": ["array-like", "sparse matrix"],
         "n_clusters": [Interval(Integral, 1, None, closed="left")],
+        "sample_weight": ["array-like", None],
         "x_squared_norms": ["array-like", None],
         "random_state": ["random_state"],
         "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
     }
 )
 def kmeans_plusplus(
-    X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None
+    X,
+    n_clusters,
+    *,
+    sample_weight=None,
+    x_squared_norms=None,
+    random_state=None,
+    n_local_trials=None,
 ):
     """Init n_clusters seeds according to k-means++.
 
@@ -83,6 +90,13 @@ def kmeans_plusplus(
     n_clusters : int
         The number of centroids to initialize.
 
+    sample_weight : array-like of shape (n_samples,), default=None
+        The weights for each observation in `X`. If `None`, all observations
+        are assigned equal weight. `sample_weight` is ignored if `init`
+        is a callable or a user provided array.
+
+        .. versionadded:: 1.3
+
     x_squared_norms : array-like of shape (n_samples,), default=None
         Squared Euclidean norm of each data point.
 
@@ -125,13 +139,14 @@ def kmeans_plusplus(
     ...               [10, 2], [10, 4], [10, 0]])
     >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
     >>> centers
-    array([[10,  4],
+    array([[10,  2],
            [ 1,  0]])
     >>> indices
-    array([4, 2])
+    array([3, 2])
     """
     # Check data
     check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
+    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
 
     if X.shape[0] < n_clusters:
         raise ValueError(
@@ -154,13 +169,15 @@ def kmeans_plusplus(
 
     # Call private k-means++
     centers, indices = _kmeans_plusplus(
-        X, n_clusters, x_squared_norms, random_state, n_local_trials
+        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials
     )
 
     return centers, indices
 
 
-def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
+def _kmeans_plusplus(
+    X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None
+):
     """Computational component for initialization of n_clusters by
     k-means++. Prior validation of data is assumed.
 
@@ -172,6 +189,9 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
     n_clusters : int
         The number of seeds to choose.
 
+    sample_weight : ndarray of shape (n_samples,)
+        The weights for each observation in `X`.
+
     x_squared_norms : ndarray of shape (n_samples,)
         Squared Euclidean norm of each data point.
 
@@ -206,7 +226,7 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
         n_local_trials = 2 + int(np.log(n_clusters))
 
     # Pick first center randomly and track index of point
-    center_id = random_state.randint(n_samples)
+    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
     indices = np.full(n_clusters, -1, dtype=int)
     if sp.issparse(X):
         centers[0] = X[center_id].toarray()
@@ -218,14 +238,16 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
     closest_dist_sq = _euclidean_distances(
         centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
     )
-    current_pot = closest_dist_sq.sum()
+    current_pot = closest_dist_sq @ sample_weight
 
     # Pick the remaining n_clusters-1 points
     for c in range(1, n_clusters):
         # Choose center candidates by sampling with probability proportional
         # to the squared distance to the closest existing center
         rand_vals = random_state.uniform(size=n_local_trials) * current_pot
-        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
+        candidate_ids = np.searchsorted(
+            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
+        )
         # XXX: numerical imprecision can result in a candidate_id out of range
         np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
 
@@ -236,7 +258,7 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
 
         # update closest distances squared and potential for each candidate
         np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
-        candidates_pot = distance_to_candidates.sum(axis=1)
+        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)
 
         # Decide which candidate is the best
         best_candidate = np.argmin(candidates_pot)
@@ -323,7 +345,8 @@ def k_means(
 
     sample_weight : array-like of shape (n_samples,), default=None
         The weights for each observation in `X`. If `None`, all observations
-        are assigned equal weight.
+        are assigned equal weight. `sample_weight` is not used during
+        initialization if `init` is a callable or a user provided array.
 
     init : {'k-means++', 'random'}, callable or array-like of shape \
             (n_clusters, n_features), default='k-means++'
@@ -939,7 +962,14 @@ def _check_test_data(self, X):
         return X
 
     def _init_centroids(
-        self, X, x_squared_norms, init, random_state, init_size=None, n_centroids=None
+        self,
+        X,
+        x_squared_norms,
+        init,
+        random_state,
+        init_size=None,
+        n_centroids=None,
+        sample_weight=None,
     ):
         """Compute the initial centroids.
 
@@ -969,6 +999,11 @@ def _init_centroids(
             If left to 'None' the number of centroids will be equal to
             number of clusters to form (self.n_clusters)
 
+        sample_weight : ndarray of shape (n_samples,), default=None
+            The weights for each observation in X. If None, all observations
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
+
         Returns
         -------
         centers : ndarray of shape (n_clusters, n_features)
@@ -981,6 +1016,7 @@ def _init_centroids(
             X = X[init_indices]
             x_squared_norms = x_squared_norms[init_indices]
             n_samples = X.shape[0]
+            sample_weight = sample_weight[init_indices]
 
         if isinstance(init, str) and init == "k-means++":
             centers, _ = _kmeans_plusplus(
@@ -988,9 +1024,15 @@ def _init_centroids(
                 n_clusters,
                 random_state=random_state,
                 x_squared_norms=x_squared_norms,
+                sample_weight=sample_weight,
             )
         elif isinstance(init, str) and init == "random":
-            seeds = random_state.permutation(n_samples)[:n_clusters]
+            seeds = random_state.choice(
+                n_samples,
+                size=n_clusters,
+                replace=False,
+                p=sample_weight / sample_weight.sum(),
+            )
             centers = X[seeds]
         elif _is_arraylike_not_scalar(self.init):
             centers = init
@@ -1412,7 +1454,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
             .. versionadded:: 0.20
 
@@ -1468,7 +1511,11 @@ def fit(self, X, y=None, sample_weight=None):
         for i in range(self._n_init):
             # Initialize centers
             centers_init = self._init_centroids(
-                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
+                X,
+                x_squared_norms=x_squared_norms,
+                init=init,
+                random_state=random_state,
+                sample_weight=sample_weight,
             )
             if self.verbose:
                 print("Initialization complete")
@@ -1545,7 +1592,7 @@ def _mini_batch_step(
         Squared euclidean norm of each data point.
 
     sample_weight : ndarray of shape (n_samples,)
-        The weights for each observation in X.
+        The weights for each observation in `X`.
 
     centers : ndarray of shape (n_clusters, n_features)
         The cluster centers before the current iteration
@@ -1818,10 +1865,10 @@ class MiniBatchKMeans(_BaseKMeans):
     >>> kmeans = kmeans.partial_fit(X[0:6,:])
     >>> kmeans = kmeans.partial_fit(X[6:12,:])
     >>> kmeans.cluster_centers_
-    array([[2. , 1. ],
-           [3.5, 4.5]])
+    array([[3.375, 3.  ],
+           [0.75 , 0.5 ]])
     >>> kmeans.predict([[0, 0], [4, 4]])
-    array([0, 1], dtype=int32)
+    array([1, 0], dtype=int32)
     >>> # fit on the whole data
     >>> kmeans = MiniBatchKMeans(n_clusters=2,
     ...                          random_state=0,
@@ -1829,8 +1876,8 @@ class MiniBatchKMeans(_BaseKMeans):
     ...                          max_iter=10,
     ...                          n_init="auto").fit(X)
     >>> kmeans.cluster_centers_
-    array([[3.97727273, 2.43181818],
-           [1.125     , 1.6       ]])
+    array([[3.55102041, 2.48979592],
+           [1.06896552, 1.        ]])
     >>> kmeans.predict([[0, 0], [4, 4]])
     array([1, 0], dtype=int32)
     """
@@ -2015,7 +2062,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
             .. versionadded:: 0.20
 
@@ -2070,6 +2118,7 @@ def fit(self, X, y=None, sample_weight=None):
                 init=init,
                 random_state=random_state,
                 init_size=self._init_size,
+                sample_weight=sample_weight,
             )
 
             # Compute inertia on a validation set.
@@ -2170,7 +2219,8 @@ def partial_fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
         Returns
         -------
@@ -2220,6 +2270,7 @@ def partial_fit(self, X, y=None, sample_weight=None):
                 init=init,
                 random_state=self._random_state,
                 init_size=self._init_size,
+                sample_weight=sample_weight,
             )
 
             # Initialize counts

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/cluster/_bicluster.py | 490 | 490 | - | 25 | -
| sklearn/cluster/_bisect_k_means.py | 193 | 204 | 39 | 7 | 21846
| sklearn/cluster/_bisect_k_means.py | 312 | 312 | 97 | 7 | 50311
| sklearn/cluster/_bisect_k_means.py | 364 | 364 | 53 | 7 | 29012
| sklearn/cluster/_kmeans.py | 66 | 66 | 29 | 1 | 16072
| sklearn/cluster/_kmeans.py | 86 | 86 | 29 | 1 | 16072
| sklearn/cluster/_kmeans.py | 128 | 131 | 29 | 1 | 16072
| sklearn/cluster/_kmeans.py | 157 | 163 | - | 1 | -
| sklearn/cluster/_kmeans.py | 175 | 175 | 30 | 1 | 16609
| sklearn/cluster/_kmeans.py | 209 | 209 | 30 | 1 | 16609
| sklearn/cluster/_kmeans.py | 221 | 228 | - | 1 | -
| sklearn/cluster/_kmeans.py | 239 | 239 | 54 | 1 | 29439
| sklearn/cluster/_kmeans.py | 326 | 326 | 24 | 1 | 13822
| sklearn/cluster/_kmeans.py | 942 | 942 | 15 | 1 | 7565
| sklearn/cluster/_kmeans.py | 972 | 972 | 15 | 1 | 7565
| sklearn/cluster/_kmeans.py | 984 | 984 | 15 | 1 | 7565
| sklearn/cluster/_kmeans.py | 991 | 991 | 15 | 1 | 7565
| sklearn/cluster/_kmeans.py | 1415 | 1415 | 27 | 1 | 14975
| sklearn/cluster/_kmeans.py | 1471 | 1471 | 27 | 1 | 14975
| sklearn/cluster/_kmeans.py | 1548 | 1548 | 33 | 1 | 18054
| sklearn/cluster/_kmeans.py | 1821 | 1824 | 20 | 1 | 11381
| sklearn/cluster/_kmeans.py | 1832 | 1833 | 20 | 1 | 11381
| sklearn/cluster/_kmeans.py | 2018 | 2018 | 37 | 1 | 20287
| sklearn/cluster/_kmeans.py | 2073 | 2073 | 37 | 1 | 20287
| sklearn/cluster/_kmeans.py | 2173 | 2173 | 89 | 1 | 47641
| sklearn/cluster/_kmeans.py | 2223 | 2223 | 89 | 1 | 47641


## Problem Statement

```
KMeans initialization does not use sample weights
### Describe the bug

Clustering by KMeans does not weight the input data.

### Steps/Code to Reproduce

\`\`\`py
import numpy as np
from sklearn.cluster import KMeans
x = np.array([1, 1, 5, 5, 100, 100])
w = 10**np.array([8.,8,8,8,-8,-8]) # large weights for 1 and 5, small weights for 100
x=x.reshape(-1,1)# reshape to a 2-dimensional array requested for KMeans
centers_with_weight = KMeans(n_clusters=2, random_state=0,n_init=10).fit(x,sample_weight=w).cluster_centers_
centers_no_weight = KMeans(n_clusters=2, random_state=0,n_init=10).fit(x).cluster_centers_
\`\`\`

### Expected Results

centers_with_weight=[[1.],[5.]]
centers_no_weight=[[100.],[3.]]

### Actual Results

centers_with_weight=[[100.],[3.]]
centers_no_weight=[[100.],[3.]]

### Versions

\`\`\`shell
System:
    python: 3.10.4 (tags/v3.10.4:9d38120, Mar 23 2022, 23:13:41) [MSC v.1929 64 bit (AMD64)]
executable: E:\WPy64-31040\python-3.10.4.amd64\python.exe
   machine: Windows-10-10.0.19045-SP0

Python dependencies:
      sklearn: 1.2.1
          pip: 22.3.1
   setuptools: 62.1.0
        numpy: 1.23.3
        scipy: 1.8.1
       Cython: 0.29.28
       pandas: 1.4.2
   matplotlib: 3.5.1
       joblib: 1.2.0
threadpoolctl: 3.1.0

Built with OpenMP: True

threadpoolctl info:
       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: E:\WPy64-31040\python-3.10.4.amd64\Lib\site-packages\numpy\.libs\libopenblas.FB5AE2TYXYH2IJRDKGDGQ3XBKLKTF43H.gfortran-win_amd64.dll
        version: 0.3.20
threading_layer: pthreads
   architecture: Haswell
    num_threads: 12

       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: E:\WPy64-31040\python-3.10.4.amd64\Lib\site-packages\scipy\.libs\libopenblas.XWYDX2IKJW2NMTWSFYNGFUWKQU3LYTCZ.gfortran-win_amd64.dll
        version: 0.3.17
threading_layer: pthreads
   architecture: Haswell
    num_threads: 12

       user_api: openmp
   internal_api: openmp
         prefix: vcomp
       filepath: E:\WPy64-31040\python-3.10.4.amd64\Lib\site-packages\sklearn\.libs\vcomp140.dll
        version: None
    num_threads: 12
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/cluster/_kmeans.py** | 419 | 433| 407 | 407 | 18021 | 
| **-> 2 <-** | **1 sklearn/cluster/_kmeans.py** | 1610 | 2256| 385 | 792 | 18021 | 
| 3 | **1 sklearn/cluster/_kmeans.py** | 1171 | 1334| 1622 | 2414 | 18021 | 
| 4 | **1 sklearn/cluster/_kmeans.py** | 1370 | 1396| 256 | 2670 | 18021 | 
| 5 | **1 sklearn/cluster/_kmeans.py** | 530 | 598| 467 | 3137 | 18021 | 
| 6 | **1 sklearn/cluster/_kmeans.py** | 1 | 55| 392 | 3529 | 18021 | 
| 7 | **1 sklearn/cluster/_kmeans.py** | 853 | 888| 336 | 3865 | 18021 | 
| 8 | 2 examples/cluster/plot_kmeans_assumptions.py | 1 | 85| 759 | 4624 | 19877 | 
| 9 | 3 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 69 | 135| 543 | 5167 | 20968 | 
| 10 | **3 sklearn/cluster/_kmeans.py** | 1507 | 1521| 148 | 5315 | 20968 | 
| 11 | 3 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 1 | 52| 388 | 5703 | 20968 | 
| 12 | **3 sklearn/cluster/_kmeans.py** | 678 | 732| 381 | 6084 | 20968 | 
| 13 | 4 examples/cluster/plot_cluster_iris.py | 1 | 91| 737 | 6821 | 21713 | 
| 14 | **4 sklearn/cluster/_kmeans.py** | 1336 | 1368| 193 | 7014 | 21713 | 
| **-> 15 <-** | **4 sklearn/cluster/_kmeans.py** | 941 | 1005| 551 | 7565 | 21713 | 
| 16 | 4 examples/cluster/plot_kmeans_assumptions.py | 87 | 154| 743 | 8308 | 21713 | 
| 17 | 5 examples/cluster/plot_kmeans_plusplus.py | 1 | 42| 301 | 8609 | 22014 | 
| 18 | 6 examples/cluster/plot_kmeans_digits.py | 111 | 183| 752 | 9361 | 23636 | 
| 19 | **6 sklearn/cluster/_kmeans.py** | 147 | 160| 235 | 9596 | 23636 | 
| **-> 20 <-** | **6 sklearn/cluster/_kmeans.py** | 1646 | 1836| 1785 | 11381 | 23636 | 
| 21 | 6 examples/cluster/plot_kmeans_digits.py | 61 | 108| 379 | 11760 | 23636 | 
| 22 | **6 sklearn/cluster/_kmeans.py** | 1880 | 1907| 257 | 12017 | 23636 | 
| 23 | **6 sklearn/cluster/_kmeans.py** | 2105 | 2154| 412 | 12429 | 23636 | 
| **-> 24 <-** | **6 sklearn/cluster/_kmeans.py** | 272 | 418| 1393 | 13822 | 23636 | 
| 25 | **6 sklearn/cluster/_kmeans.py** | 917 | 939| 209 | 14031 | 23636 | 
| 26 | 6 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 55 | 66| 134 | 14165 | 23636 | 
| **-> 27 <-** | **6 sklearn/cluster/_kmeans.py** | 1398 | 1506| 810 | 14975 | 23636 | 
| 28 | **7 sklearn/cluster/_bisect_k_means.py** | 207 | 251| 336 | 15311 | 27848 | 
| **-> 29 <-** | **7 sklearn/cluster/_kmeans.py** | 58 | 145| 761 | 16072 | 27848 | 
| **-> 30 <-** | **7 sklearn/cluster/_kmeans.py** | 163 | 223| 537 | 16609 | 27848 | 
| 31 | **7 sklearn/cluster/_kmeans.py** | 735 | 802| 466 | 17075 | 27848 | 
| 32 | 8 sklearn/utils/class_weight.py | 149 | 195| 354 | 17429 | 29477 | 
| **-> 33 <-** | **8 sklearn/cluster/_kmeans.py** | 1524 | 1609| 625 | 18054 | 29477 | 
| 34 | **8 sklearn/cluster/_bisect_k_means.py** | 253 | 283| 246 | 18300 | 29477 | 
| 35 | 8 examples/cluster/plot_kmeans_assumptions.py | 155 | 179| 251 | 18551 | 29477 | 
| 36 | 9 examples/cluster/plot_kmeans_silhouette_analysis.py | 57 | 160| 932 | 19483 | 30912 | 
| **-> 37 <-** | **9 sklearn/cluster/_kmeans.py** | 2001 | 2103| 804 | 20287 | 30912 | 
| 38 | **9 sklearn/cluster/_kmeans.py** | 1838 | 1878| 307 | 20594 | 30912 | 
| **-> 39 <-** | **9 sklearn/cluster/_bisect_k_means.py** | 76 | 205| 1252 | 21846 | 30912 | 
| 40 | 9 examples/cluster/plot_kmeans_digits.py | 1 | 58| 389 | 22235 | 30912 | 
| 41 | 10 sklearn/datasets/_samples_generator.py | 237 | 276| 804 | 23039 | 46902 | 
| 42 | 11 examples/cluster/plot_mini_batch_kmeans.py | 111 | 143| 311 | 23350 | 47988 | 
| 43 | 12 examples/cluster/plot_birch_vs_minibatchkmeans.py | 85 | 110| 257 | 23607 | 49058 | 
| 44 | 13 examples/text/plot_document_clustering.py | 334 | 451| 1226 | 24833 | 53224 | 
| 45 | 14 examples/release_highlights/plot_release_highlights_1_1_0.py | 178 | 227| 513 | 25346 | 55361 | 
| 46 | 15 examples/svm/plot_weighted_samples.py | 45 | 72| 248 | 25594 | 55892 | 
| 47 | **15 sklearn/cluster/_kmeans.py** | 1985 | 1999| 130 | 25724 | 55892 | 
| 48 | 16 benchmarks/bench_plot_fastkmeans.py | 100 | 143| 495 | 26219 | 57110 | 
| 49 | 16 examples/text/plot_document_clustering.py | 153 | 234| 775 | 26994 | 57110 | 
| 50 | 16 examples/cluster/plot_kmeans_silhouette_analysis.py | 1 | 55| 503 | 27497 | 57110 | 
| 51 | 17 examples/mixture/plot_gmm_init.py | 1 | 75| 619 | 28116 | 58062 | 
| 52 | **17 sklearn/cluster/_bisect_k_means.py** | 1 | 20| 157 | 28273 | 58062 | 
| **-> 53 <-** | **17 sklearn/cluster/_bisect_k_means.py** | 346 | 437| 739 | 29012 | 58062 | 
| **-> 54 <-** | **17 sklearn/cluster/_kmeans.py** | 224 | 269| 427 | 29439 | 58062 | 
| 55 | 17 examples/cluster/plot_mini_batch_kmeans.py | 1 | 110| 775 | 30214 | 58062 | 
| 56 | 18 examples/cluster/plot_agglomerative_clustering.py | 1 | 84| 681 | 30895 | 58767 | 
| 57 | 19 examples/cluster/plot_ward_structured_vs_unstructured.py | 1 | 135| 854 | 31749 | 59661 | 
| 58 | 20 examples/cluster/plot_cluster_comparison.py | 130 | 265| 964 | 32713 | 61532 | 
| 59 | **20 sklearn/cluster/_kmeans.py** | 1007 | 1030| 196 | 32909 | 61532 | 
| 60 | **20 sklearn/cluster/_kmeans.py** | 436 | 528| 695 | 33604 | 61532 | 
| 61 | 21 sklearn/cluster/_mean_shift.py | 1 | 29| 191 | 33795 | 65971 | 
| 62 | **21 sklearn/cluster/_kmeans.py** | 805 | 851| 354 | 34149 | 65971 | 
| 63 | 21 examples/text/plot_document_clustering.py | 1 | 111| 675 | 34824 | 65971 | 
| 64 | 21 examples/cluster/plot_cluster_comparison.py | 1 | 82| 632 | 35456 | 65971 | 
| 65 | 21 benchmarks/bench_plot_fastkmeans.py | 1 | 54| 391 | 35847 | 65971 | 
| 66 | **21 sklearn/cluster/_kmeans.py** | 890 | 915| 304 | 36151 | 65971 | 
| 67 | 21 examples/cluster/plot_kmeans_digits.py | 184 | 202| 101 | 36252 | 65971 | 
| 68 | 22 examples/release_highlights/plot_release_highlights_0_23_0.py | 98 | 175| 789 | 37041 | 67753 | 
| 69 | 23 sklearn/cluster/_agglomerative.py | 10 | 37| 215 | 37256 | 78938 | 
| 70 | 24 examples/cluster/plot_optics.py | 83 | 108| 380 | 37636 | 80115 | 
| 71 | **24 sklearn/cluster/_kmeans.py** | 1032 | 1078| 371 | 38007 | 80115 | 
| 72 | **25 sklearn/cluster/_bicluster.py** | 181 | 199| 129 | 38136 | 85469 | 
| 73 | 26 examples/cluster/plot_mean_shift.py | 1 | 66| 410 | 38546 | 85879 | 
| 74 | 27 examples/cluster/plot_bisect_kmeans.py | 1 | 66| 496 | 39042 | 86375 | 
| 75 | 27 sklearn/cluster/_mean_shift.py | 275 | 407| 1117 | 40159 | 86375 | 
| 76 | **27 sklearn/cluster/_kmeans.py** | 601 | 677| 599 | 40758 | 86375 | 
| 77 | 28 sklearn/metrics/_pairwise_distances_reduction/_dispatcher.py | 584 | 621| 335 | 41093 | 91030 | 
| 78 | 29 asv_benchmarks/benchmarks/cluster.py | 1 | 54| 356 | 41449 | 91686 | 
| 79 | 29 sklearn/utils/class_weight.py | 79 | 148| 701 | 42150 | 91686 | 
| 80 | 29 sklearn/cluster/_mean_shift.py | 438 | 530| 758 | 42908 | 91686 | 
| 81 | 30 examples/cluster/plot_affinity_propagation.py | 1 | 75| 554 | 43462 | 92240 | 
| 82 | **30 sklearn/cluster/_kmeans.py** | 1909 | 1918| 120 | 43582 | 92240 | 
| 83 | 31 examples/cluster/plot_dbscan.py | 1 | 95| 768 | 44350 | 93233 | 
| 84 | 31 benchmarks/bench_plot_fastkmeans.py | 57 | 97| 330 | 44680 | 93233 | 
| 85 | 31 examples/cluster/plot_cluster_comparison.py | 84 | 128| 275 | 44955 | 93233 | 
| 86 | 32 examples/cluster/plot_linkage_comparison.py | 1 | 80| 591 | 45546 | 94427 | 
| 87 | 32 examples/cluster/plot_linkage_comparison.py | 82 | 176| 603 | 46149 | 94427 | 
| 88 | 32 examples/cluster/plot_birch_vs_minibatchkmeans.py | 1 | 84| 768 | 46917 | 94427 | 
| **-> 89 <-** | **32 sklearn/cluster/_kmeans.py** | 2156 | 2257| 724 | 47641 | 94427 | 
| 90 | 33 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 100 | 115| 176 | 47817 | 95414 | 
| 91 | **33 sklearn/cluster/_kmeans.py** | 1080 | 1102| 195 | 48012 | 95414 | 
| 92 | 34 examples/cluster/plot_adjusted_for_chance_measures.py | 104 | 205| 781 | 48793 | 97372 | 
| 93 | 35 sklearn/cluster/__init__.py | 1 | 55| 381 | 49174 | 97753 | 
| 94 | 36 sklearn/cluster/_birch.py | 641 | 670| 203 | 49377 | 103627 | 
| 95 | 36 examples/mixture/plot_gmm_init.py | 77 | 110| 298 | 49675 | 103627 | 
| 96 | 36 sklearn/cluster/_birch.py | 6 | 25| 123 | 49798 | 103627 | 
| **-> 97 <-** | **36 sklearn/cluster/_bisect_k_means.py** | 285 | 344| 513 | 50311 | 103627 | 
| 98 | 36 sklearn/cluster/_agglomerative.py | 972 | 1090| 910 | 51221 | 103627 | 
| 99 | 36 sklearn/cluster/_agglomerative.py | 907 | 948| 341 | 51562 | 103627 | 
| 100 | 36 sklearn/cluster/_mean_shift.py | 409 | 436| 228 | 51790 | 103627 | 
| 101 | 37 sklearn/utils/validation.py | 1716 | 1788| 530 | 52320 | 120572 | 
| 102 | 38 examples/neighbors/plot_kde_1d.py | 1 | 69| 745 | 53065 | 122125 | 
| 103 | 39 sklearn/dummy.py | 6 | 22| 136 | 53201 | 127315 | 
| 104 | 40 examples/cluster/plot_agglomerative_clustering_metrics.py | 97 | 147| 430 | 53631 | 128537 | 
| 105 | 40 examples/cluster/plot_adjusted_for_chance_measures.py | 1 | 81| 290 | 53921 | 128537 | 
| 106 | 41 sklearn/cluster/_affinity_propagation.py | 294 | 427| 1166 | 55087 | 133016 | 
| 107 | 42 asv_benchmarks/benchmarks/utils.py | 1 | 48| 333 | 55420 | 133349 | 
| 108 | 42 examples/cluster/plot_agglomerative_clustering_metrics.py | 1 | 96| 771 | 56191 | 133349 | 
| 109 | 43 sklearn/metrics/pairwise.py | 10 | 36| 204 | 56395 | 152242 | 
| 110 | 43 examples/cluster/plot_dbscan.py | 96 | 129| 225 | 56620 | 152242 | 
| 111 | 43 sklearn/cluster/_mean_shift.py | 122 | 219| 773 | 57393 | 152242 | 
| 112 | 44 examples/neighbors/plot_regression.py | 1 | 51| 285 | 57678 | 152587 | 
| 113 | 45 examples/release_highlights/plot_release_highlights_0_22_0.py | 90 | 193| 896 | 58574 | 154997 | 
| 114 | 45 sklearn/utils/class_weight.py | 5 | 76| 576 | 59150 | 154997 | 
| 115 | 46 sklearn/impute/_knn.py | 5 | 17| 111 | 59261 | 158134 | 
| 116 | 47 benchmarks/bench_plot_hierarchical.py | 1 | 37| 249 | 59510 | 158750 | 
| 117 | **47 sklearn/cluster/_kmeans.py** | 1104 | 1128| 197 | 59707 | 158750 | 
| 118 | 48 sklearn/neighbors/_classification.py | 177 | 204| 172 | 59879 | 165157 | 
| 119 | **48 sklearn/cluster/_bisect_k_means.py** | 471 | 524| 403 | 60282 | 165157 | 
| 120 | 49 examples/bicluster/plot_bicluster_newsgroups.py | 104 | 126| 231 | 60513 | 166541 | 
| 121 | 49 asv_benchmarks/benchmarks/cluster.py | 57 | 105| 300 | 60813 | 166541 | 
| 122 | 50 examples/cluster/plot_inductive_clustering.py | 76 | 130| 394 | 61207 | 167453 | 
| 123 | 50 examples/text/plot_document_clustering.py | 114 | 150| 356 | 61563 | 167453 | 
| 124 | **50 sklearn/cluster/_kmeans.py** | 1920 | 1983| 566 | 62129 | 167453 | 
| 125 | 50 sklearn/neighbors/_classification.py | 35 | 175| 1326 | 63455 | 167453 | 
| 126 | 50 examples/neighbors/plot_kde_1d.py | 71 | 157| 789 | 64244 | 167453 | 
| 127 | 51 sklearn/linear_model/_stochastic_gradient.py | 9 | 56| 405 | 64649 | 187298 | 
| 128 | 51 examples/cluster/plot_optics.py | 1 | 82| 757 | 65406 | 187298 | 
| 129 | 51 examples/cluster/plot_adjusted_for_chance_measures.py | 206 | 231| 273 | 65679 | 187298 | 
| 130 | 51 sklearn/cluster/_agglomerative.py | 268 | 349| 734 | 66413 | 187298 | 
| 131 | 51 sklearn/neighbors/_classification.py | 367 | 388| 214 | 66627 | 187298 | 
| 132 | 52 examples/neural_networks/plot_mnist_filters.py | 1 | 72| 663 | 67290 | 187961 | 
| 133 | 52 sklearn/cluster/_mean_shift.py | 98 | 119| 232 | 67522 | 187961 | 
| 134 | 53 examples/cluster/plot_color_quantization.py | 1 | 98| 762 | 68284 | 188777 | 
| 135 | 54 sklearn/neighbors/_nearest_centroid.py | 186 | 210| 323 | 68607 | 190764 | 


### Hint

```
Thanks for the reproducible example.

`KMeans` **does** weight the data, but your example is an extreme case. Because `Kmeans` is a non-convex problem, the algorithm can get stuck in a local minimum, and not find the true minimum  of the optimization landscape. This is the reason why the code proposes to use multiple initializations, hoping that some initializations will not get stuck in poor local minima.

Importantly, **the initialization does not use sample weights**. So when using `init="k-means++"` (default), it gets a centroid on the outliers, and `Kmeans` cannot escape this local minimum, even with strong sample weights.

Instead, the optimization does not get stuck if we do one of the following changes:
- the outliers are less extremes (e.g. 10 instead of 100), because it allows the sample weights to remove the local minimum
- the low samples weights are exactly equal to zero, because it completely discard the outliers
- using `init="random"`, because some of the different init are able to avoid the local minimum
If we have less extreme weighting then \`\`\`init='random'\`\`\` does not work at all as shown in this example. The centers are always [[3.],[40.]] instead of circa [[1.],[5.]] for the weighted case.
\`\`\`py
import numpy as np
from sklearn.cluster import KMeans
x = np.array([1, 1, 5, 5, 40, 40])
w = np.array([100,100,100,100,1,1]) # large weights for 1 and 5, small weights for 40
x=x.reshape(-1,1)# reshape to a 2-dimensional array requested for KMeans
centers_with_weight = KMeans(n_clusters=2, random_state=0,n_init=100,init='random').fit(x,sample_weight=w).cluster_centers_
centers_no_weight = KMeans(n_clusters=2, random_state=0,n_init=100).fit(x).cluster_centers_
\`\`\`
Sure, you can find examples where `init="random"` also fails.
My point is that `KMeans` is non-convex, so it is never guaranteed to find the global minimum.

That being said, **it would be nice to have an initialization that uses the sample weights**. I don't know about `init="k-means++"`, but for `init="random"` we could use a non-uniform sampling relative to the sample weights, instead of the current uniform sampling.

Do you want to submit a pull-request?
yes
I want to work on this issue
Was this bug already resolved?
Not resolved yet, you are welcome to submit a pull-request if you'd like.
I already answered this question by "yes" 3 weeks ago. 
```

## Patch

```diff
diff --git a/sklearn/cluster/_bicluster.py b/sklearn/cluster/_bicluster.py
--- a/sklearn/cluster/_bicluster.py
+++ b/sklearn/cluster/_bicluster.py
@@ -487,7 +487,7 @@ class SpectralBiclustering(BaseSpectral):
     >>> clustering.row_labels_
     array([1, 1, 1, 0, 0, 0], dtype=int32)
     >>> clustering.column_labels_
-    array([0, 1], dtype=int32)
+    array([1, 0], dtype=int32)
     >>> clustering
     SpectralBiclustering(n_clusters=2, random_state=0)
     """
diff --git a/sklearn/cluster/_bisect_k_means.py b/sklearn/cluster/_bisect_k_means.py
--- a/sklearn/cluster/_bisect_k_means.py
+++ b/sklearn/cluster/_bisect_k_means.py
@@ -190,18 +190,18 @@ class BisectingKMeans(_BaseKMeans):
     --------
     >>> from sklearn.cluster import BisectingKMeans
     >>> import numpy as np
-    >>> X = np.array([[1, 2], [1, 4], [1, 0],
-    ...               [10, 2], [10, 4], [10, 0],
-    ...               [10, 6], [10, 8], [10, 10]])
+    >>> X = np.array([[1, 1], [10, 1], [3, 1],
+    ...               [10, 0], [2, 1], [10, 2],
+    ...               [10, 8], [10, 9], [10, 10]])
     >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
     >>> bisect_means.labels_
-    array([2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=int32)
+    array([0, 2, 0, 2, 0, 2, 1, 1, 1], dtype=int32)
     >>> bisect_means.predict([[0, 0], [12, 3]])
-    array([2, 0], dtype=int32)
+    array([0, 2], dtype=int32)
     >>> bisect_means.cluster_centers_
-    array([[10.,  2.],
-           [10.,  8.],
-           [ 1., 2.]])
+    array([[ 2., 1.],
+           [10., 9.],
+           [10., 1.]])
     """
 
     _parameter_constraints: dict = {
@@ -309,7 +309,12 @@ def _bisect(self, X, x_squared_norms, sample_weight, cluster_to_bisect):
         # Repeating `n_init` times to obtain best clusters
         for _ in range(self.n_init):
             centers_init = self._init_centroids(
-                X, x_squared_norms, self.init, self._random_state, n_centroids=2
+                X,
+                x_squared_norms=x_squared_norms,
+                init=self.init,
+                random_state=self._random_state,
+                n_centroids=2,
+                sample_weight=sample_weight,
             )
 
             labels, inertia, centers, _ = self._kmeans_single(
@@ -361,7 +366,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable.
 
         Returns
         -------
diff --git a/sklearn/cluster/_kmeans.py b/sklearn/cluster/_kmeans.py
--- a/sklearn/cluster/_kmeans.py
+++ b/sklearn/cluster/_kmeans.py
@@ -63,13 +63,20 @@
     {
         "X": ["array-like", "sparse matrix"],
         "n_clusters": [Interval(Integral, 1, None, closed="left")],
+        "sample_weight": ["array-like", None],
         "x_squared_norms": ["array-like", None],
         "random_state": ["random_state"],
         "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
     }
 )
 def kmeans_plusplus(
-    X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None
+    X,
+    n_clusters,
+    *,
+    sample_weight=None,
+    x_squared_norms=None,
+    random_state=None,
+    n_local_trials=None,
 ):
     """Init n_clusters seeds according to k-means++.
 
@@ -83,6 +90,13 @@ def kmeans_plusplus(
     n_clusters : int
         The number of centroids to initialize.
 
+    sample_weight : array-like of shape (n_samples,), default=None
+        The weights for each observation in `X`. If `None`, all observations
+        are assigned equal weight. `sample_weight` is ignored if `init`
+        is a callable or a user provided array.
+
+        .. versionadded:: 1.3
+
     x_squared_norms : array-like of shape (n_samples,), default=None
         Squared Euclidean norm of each data point.
 
@@ -125,13 +139,14 @@ def kmeans_plusplus(
     ...               [10, 2], [10, 4], [10, 0]])
     >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
     >>> centers
-    array([[10,  4],
+    array([[10,  2],
            [ 1,  0]])
     >>> indices
-    array([4, 2])
+    array([3, 2])
     """
     # Check data
     check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])
+    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
 
     if X.shape[0] < n_clusters:
         raise ValueError(
@@ -154,13 +169,15 @@ def kmeans_plusplus(
 
     # Call private k-means++
     centers, indices = _kmeans_plusplus(
-        X, n_clusters, x_squared_norms, random_state, n_local_trials
+        X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials
     )
 
     return centers, indices
 
 
-def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
+def _kmeans_plusplus(
+    X, n_clusters, x_squared_norms, sample_weight, random_state, n_local_trials=None
+):
     """Computational component for initialization of n_clusters by
     k-means++. Prior validation of data is assumed.
 
@@ -172,6 +189,9 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
     n_clusters : int
         The number of seeds to choose.
 
+    sample_weight : ndarray of shape (n_samples,)
+        The weights for each observation in `X`.
+
     x_squared_norms : ndarray of shape (n_samples,)
         Squared Euclidean norm of each data point.
 
@@ -206,7 +226,7 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
         n_local_trials = 2 + int(np.log(n_clusters))
 
     # Pick first center randomly and track index of point
-    center_id = random_state.randint(n_samples)
+    center_id = random_state.choice(n_samples, p=sample_weight / sample_weight.sum())
     indices = np.full(n_clusters, -1, dtype=int)
     if sp.issparse(X):
         centers[0] = X[center_id].toarray()
@@ -218,14 +238,16 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
     closest_dist_sq = _euclidean_distances(
         centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
     )
-    current_pot = closest_dist_sq.sum()
+    current_pot = closest_dist_sq @ sample_weight
 
     # Pick the remaining n_clusters-1 points
     for c in range(1, n_clusters):
         # Choose center candidates by sampling with probability proportional
         # to the squared distance to the closest existing center
         rand_vals = random_state.uniform(size=n_local_trials) * current_pot
-        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
+        candidate_ids = np.searchsorted(
+            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
+        )
         # XXX: numerical imprecision can result in a candidate_id out of range
         np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
 
@@ -236,7 +258,7 @@ def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trial
 
         # update closest distances squared and potential for each candidate
         np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
-        candidates_pot = distance_to_candidates.sum(axis=1)
+        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)
 
         # Decide which candidate is the best
         best_candidate = np.argmin(candidates_pot)
@@ -323,7 +345,8 @@ def k_means(
 
     sample_weight : array-like of shape (n_samples,), default=None
         The weights for each observation in `X`. If `None`, all observations
-        are assigned equal weight.
+        are assigned equal weight. `sample_weight` is not used during
+        initialization if `init` is a callable or a user provided array.
 
     init : {'k-means++', 'random'}, callable or array-like of shape \
             (n_clusters, n_features), default='k-means++'
@@ -939,7 +962,14 @@ def _check_test_data(self, X):
         return X
 
     def _init_centroids(
-        self, X, x_squared_norms, init, random_state, init_size=None, n_centroids=None
+        self,
+        X,
+        x_squared_norms,
+        init,
+        random_state,
+        init_size=None,
+        n_centroids=None,
+        sample_weight=None,
     ):
         """Compute the initial centroids.
 
@@ -969,6 +999,11 @@ def _init_centroids(
             If left to 'None' the number of centroids will be equal to
             number of clusters to form (self.n_clusters)
 
+        sample_weight : ndarray of shape (n_samples,), default=None
+            The weights for each observation in X. If None, all observations
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
+
         Returns
         -------
         centers : ndarray of shape (n_clusters, n_features)
@@ -981,6 +1016,7 @@ def _init_centroids(
             X = X[init_indices]
             x_squared_norms = x_squared_norms[init_indices]
             n_samples = X.shape[0]
+            sample_weight = sample_weight[init_indices]
 
         if isinstance(init, str) and init == "k-means++":
             centers, _ = _kmeans_plusplus(
@@ -988,9 +1024,15 @@ def _init_centroids(
                 n_clusters,
                 random_state=random_state,
                 x_squared_norms=x_squared_norms,
+                sample_weight=sample_weight,
             )
         elif isinstance(init, str) and init == "random":
-            seeds = random_state.permutation(n_samples)[:n_clusters]
+            seeds = random_state.choice(
+                n_samples,
+                size=n_clusters,
+                replace=False,
+                p=sample_weight / sample_weight.sum(),
+            )
             centers = X[seeds]
         elif _is_arraylike_not_scalar(self.init):
             centers = init
@@ -1412,7 +1454,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
             .. versionadded:: 0.20
 
@@ -1468,7 +1511,11 @@ def fit(self, X, y=None, sample_weight=None):
         for i in range(self._n_init):
             # Initialize centers
             centers_init = self._init_centroids(
-                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
+                X,
+                x_squared_norms=x_squared_norms,
+                init=init,
+                random_state=random_state,
+                sample_weight=sample_weight,
             )
             if self.verbose:
                 print("Initialization complete")
@@ -1545,7 +1592,7 @@ def _mini_batch_step(
         Squared euclidean norm of each data point.
 
     sample_weight : ndarray of shape (n_samples,)
-        The weights for each observation in X.
+        The weights for each observation in `X`.
 
     centers : ndarray of shape (n_clusters, n_features)
         The cluster centers before the current iteration
@@ -1818,10 +1865,10 @@ class MiniBatchKMeans(_BaseKMeans):
     >>> kmeans = kmeans.partial_fit(X[0:6,:])
     >>> kmeans = kmeans.partial_fit(X[6:12,:])
     >>> kmeans.cluster_centers_
-    array([[2. , 1. ],
-           [3.5, 4.5]])
+    array([[3.375, 3.  ],
+           [0.75 , 0.5 ]])
     >>> kmeans.predict([[0, 0], [4, 4]])
-    array([0, 1], dtype=int32)
+    array([1, 0], dtype=int32)
     >>> # fit on the whole data
     >>> kmeans = MiniBatchKMeans(n_clusters=2,
     ...                          random_state=0,
@@ -1829,8 +1876,8 @@ class MiniBatchKMeans(_BaseKMeans):
     ...                          max_iter=10,
     ...                          n_init="auto").fit(X)
     >>> kmeans.cluster_centers_
-    array([[3.97727273, 2.43181818],
-           [1.125     , 1.6       ]])
+    array([[3.55102041, 2.48979592],
+           [1.06896552, 1.        ]])
     >>> kmeans.predict([[0, 0], [4, 4]])
     array([1, 0], dtype=int32)
     """
@@ -2015,7 +2062,8 @@ def fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
             .. versionadded:: 0.20
 
@@ -2070,6 +2118,7 @@ def fit(self, X, y=None, sample_weight=None):
                 init=init,
                 random_state=random_state,
                 init_size=self._init_size,
+                sample_weight=sample_weight,
             )
 
             # Compute inertia on a validation set.
@@ -2170,7 +2219,8 @@ def partial_fit(self, X, y=None, sample_weight=None):
 
         sample_weight : array-like of shape (n_samples,), default=None
             The weights for each observation in X. If None, all observations
-            are assigned equal weight.
+            are assigned equal weight. `sample_weight` is not used during
+            initialization if `init` is a callable or a user provided array.
 
         Returns
         -------
@@ -2220,6 +2270,7 @@ def partial_fit(self, X, y=None, sample_weight=None):
                 init=init,
                 random_state=self._random_state,
                 init_size=self._init_size,
+                sample_weight=sample_weight,
             )
 
             # Initialize counts

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_bisect_k_means.py b/sklearn/cluster/tests/test_bisect_k_means.py
--- a/sklearn/cluster/tests/test_bisect_k_means.py
+++ b/sklearn/cluster/tests/test_bisect_k_means.py
@@ -4,34 +4,33 @@
 
 from sklearn.utils._testing import assert_array_equal, assert_allclose
 from sklearn.cluster import BisectingKMeans
+from sklearn.metrics import v_measure_score
 
 
 @pytest.mark.parametrize("bisecting_strategy", ["biggest_inertia", "largest_cluster"])
-def test_three_clusters(bisecting_strategy):
+@pytest.mark.parametrize("init", ["k-means++", "random"])
+def test_three_clusters(bisecting_strategy, init):
     """Tries to perform bisect k-means for three clusters to check
     if splitting data is performed correctly.
     """
-
-    # X = np.array([[1, 2], [1, 4], [1, 0],
-    #               [10, 2], [10, 4], [10, 0],
-    #               [10, 6], [10, 8], [10, 10]])
-
-    # X[0][1] swapped with X[1][1] intentionally for checking labeling
     X = np.array(
-        [[1, 2], [10, 4], [1, 0], [10, 2], [1, 4], [10, 0], [10, 6], [10, 8], [10, 10]]
+        [[1, 1], [10, 1], [3, 1], [10, 0], [2, 1], [10, 2], [10, 8], [10, 9], [10, 10]]
     )
     bisect_means = BisectingKMeans(
-        n_clusters=3, random_state=0, bisecting_strategy=bisecting_strategy
+        n_clusters=3,
+        random_state=0,
+        bisecting_strategy=bisecting_strategy,
+        init=init,
     )
     bisect_means.fit(X)
 
-    expected_centers = [[10, 2], [10, 8], [1, 2]]
-    expected_predict = [2, 0]
-    expected_labels = [2, 0, 2, 0, 2, 0, 1, 1, 1]
+    expected_centers = [[2, 1], [10, 1], [10, 9]]
+    expected_labels = [0, 1, 0, 1, 0, 1, 2, 2, 2]
 
-    assert_allclose(expected_centers, bisect_means.cluster_centers_)
-    assert_array_equal(expected_predict, bisect_means.predict([[0, 0], [12, 3]]))
-    assert_array_equal(expected_labels, bisect_means.labels_)
+    assert_allclose(
+        sorted(expected_centers), sorted(bisect_means.cluster_centers_.tolist())
+    )
+    assert_allclose(v_measure_score(expected_labels, bisect_means.labels_), 1.0)
 
 
 def test_sparse():
diff --git a/sklearn/cluster/tests/test_k_means.py b/sklearn/cluster/tests/test_k_means.py
--- a/sklearn/cluster/tests/test_k_means.py
+++ b/sklearn/cluster/tests/test_k_means.py
@@ -17,6 +17,7 @@
 from sklearn.utils.extmath import row_norms
 from sklearn.metrics import pairwise_distances
 from sklearn.metrics import pairwise_distances_argmin
+from sklearn.metrics.pairwise import euclidean_distances
 from sklearn.metrics.cluster import v_measure_score
 from sklearn.cluster import KMeans, k_means, kmeans_plusplus
 from sklearn.cluster import MiniBatchKMeans
@@ -1276,3 +1277,67 @@ def test_predict_does_not_change_cluster_centers(is_sparse):
 
     y_pred2 = kmeans.predict(X)
     assert_array_equal(y_pred1, y_pred2)
+
+
+@pytest.mark.parametrize("init", ["k-means++", "random"])
+def test_sample_weight_init(init, global_random_seed):
+    """Check that sample weight is used during init.
+
+    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so
+    it's enough to check for KMeans.
+    """
+    rng = np.random.RandomState(global_random_seed)
+    X, _ = make_blobs(
+        n_samples=200, n_features=10, centers=10, random_state=global_random_seed
+    )
+    x_squared_norms = row_norms(X, squared=True)
+
+    kmeans = KMeans()
+    clusters_weighted = kmeans._init_centroids(
+        X=X,
+        x_squared_norms=x_squared_norms,
+        init=init,
+        sample_weight=rng.uniform(size=X.shape[0]),
+        n_centroids=5,
+        random_state=np.random.RandomState(global_random_seed),
+    )
+    clusters = kmeans._init_centroids(
+        X=X,
+        x_squared_norms=x_squared_norms,
+        init=init,
+        sample_weight=np.ones(X.shape[0]),
+        n_centroids=5,
+        random_state=np.random.RandomState(global_random_seed),
+    )
+    with pytest.raises(AssertionError):
+        assert_allclose(clusters_weighted, clusters)
+
+
+@pytest.mark.parametrize("init", ["k-means++", "random"])
+def test_sample_weight_zero(init, global_random_seed):
+    """Check that if sample weight is 0, this sample won't be chosen.
+
+    `_init_centroids` is shared across all classes inheriting from _BaseKMeans so
+    it's enough to check for KMeans.
+    """
+    rng = np.random.RandomState(global_random_seed)
+    X, _ = make_blobs(
+        n_samples=100, n_features=5, centers=5, random_state=global_random_seed
+    )
+    sample_weight = rng.uniform(size=X.shape[0])
+    sample_weight[::2] = 0
+    x_squared_norms = row_norms(X, squared=True)
+
+    kmeans = KMeans()
+    clusters_weighted = kmeans._init_centroids(
+        X=X,
+        x_squared_norms=x_squared_norms,
+        init=init,
+        sample_weight=sample_weight,
+        n_centroids=10,
+        random_state=np.random.RandomState(global_random_seed),
+    )
+    # No center should be one of the 0 sample weight point
+    # (i.e. be at a distance=0 from it)
+    d = euclidean_distances(X[::2], clusters_weighted)
+    assert not np.any(np.isclose(d, 0))
diff --git a/sklearn/manifold/tests/test_spectral_embedding.py b/sklearn/manifold/tests/test_spectral_embedding.py
--- a/sklearn/manifold/tests/test_spectral_embedding.py
+++ b/sklearn/manifold/tests/test_spectral_embedding.py
@@ -336,7 +336,7 @@ def test_pipeline_spectral_clustering(seed=36):
         random_state=random_state,
     )
     for se in [se_rbf, se_knn]:
-        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
+        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
         km.fit(se.fit_transform(S))
         assert_array_almost_equal(
             normalized_mutual_info_score(km.labels_, true_labels), 1.0, 2

```


## Code snippets

### 1 - sklearn/cluster/_kmeans.py:

Start line: 419, End line: 433

```python
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "copy_x": [bool],
        "algorithm": [
            StrOptions({"lloyd", "elkan", "auto", "full"}, deprecated={"auto", "full"})
        ],
        "return_n_iter": [bool],
    }
)
def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="warn",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    est = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_
```
### 2 - sklearn/cluster/_kmeans.py:

Start line: 1610, End line: 2256

```python
def _mini_batch_step(
    X,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    # ... other code
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > 0.5 * X.shape[0]:
            indices_dont_reassign = np.argsort(weight_sums)[int(0.5 * X.shape[0]) :]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(
                X.shape[0], replace=False, size=n_reassigns
            )
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} cluster centers.")

            if sp.issparse(X):
                assign_rows_csr(
                    X,
                    new_centers.astype(np.intp, copy=False),
                    np.where(to_reassign)[0].astype(np.intp, copy=False),
                    centers_new,
                )
            else:
                centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia


class MiniBatchKMeans(_BaseKMeans):
```
### 3 - sklearn/cluster/_kmeans.py:

Start line: 1171, End line: 1334

```python
class KMeans(_BaseKMeans):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : 'auto' or int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'`, 1 if using `init='k-means++'`.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` will change from 10 to `'auto'` in version 1.4.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative online implementation that does incremental
        updates of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), where n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features.
    Refer to :doi:`"How slow is the k-means method?" D. Arthur and S. Vassilvitskii -
    SoCG2006.<10.1145/1137856.1137880>` for more details.

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """
```
### 4 - sklearn/cluster/_kmeans.py:

Start line: 1370, End line: 1396

```python
class KMeans(_BaseKMeans):

    def _check_params_vs_input(self, X):
        super()._check_params_vs_input(X, default_n_init=10)

        self._algorithm = self.algorithm
        if self._algorithm in ("auto", "full"):
            warnings.warn(
                f"algorithm='{self._algorithm}' is deprecated, it will be "
                "removed in 1.3. Using 'lloyd' instead.",
                FutureWarning,
            )
            self._algorithm = "lloyd"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                "algorithm='elkan' doesn't make sense for a single "
                "cluster. Using 'lloyd' instead.",
                RuntimeWarning,
            )
            self._algorithm = "lloyd"

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "KMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )
```
### 5 - sklearn/cluster/_kmeans.py:

Start line: 530, End line: 598

```python
def _kmeans_single_elkan(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    # ... other code

    for i in range(max_iter):
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
        )

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0
        )[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(
            X,
            sample_weight,
            centers,
            centers,
            weight_in_clusters,
            center_half_distances,
            distance_next_center,
            upper_bounds,
            lower_bounds,
            labels,
            center_shift,
            n_threads,
            update_centers=False,
        )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1
```
### 6 - sklearn/cluster/_kmeans.py:

Start line: 1, End line: 55

```python
"""K-means clustering."""

from abc import ABC, abstractmethod
from numbers import Integral, Real
import warnings

import numpy as np
import scipy.sparse as sp

from ..base import (
    BaseEstimator,
    ClusterMixin,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
)
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import _euclidean_distances
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_limits
from ..utils.fixes import threadpool_info
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils import check_array
from ..utils import check_random_state
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils.validation import _is_arraylike_not_scalar
from ..utils._param_validation import Hidden
from ..utils._param_validation import Interval
from ..utils._param_validation import StrOptions
from ..utils._param_validation import validate_params
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..exceptions import ConvergenceWarning
from ._k_means_common import CHUNK_SIZE
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse
from ._k_means_common import _is_same_clustering
from ._k_means_minibatch import _minibatch_update_dense
from ._k_means_minibatch import _minibatch_update_sparse
from ._k_means_lloyd import lloyd_iter_chunked_dense
from ._k_means_lloyd import lloyd_iter_chunked_sparse
from ._k_means_elkan import init_bounds_dense
from ._k_means_elkan import init_bounds_sparse
from ._k_means_elkan import elkan_iter_chunked_dense
from ._k_means_elkan import elkan_iter_chunked_sparse
```
### 7 - sklearn/cluster/_kmeans.py:

Start line: 853, End line: 888

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def _check_params_vs_input(self, X, default_n_init=None):
        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = _tolerance(X, self.tol)

        # n-init
        # TODO(1.4): Remove
        self._n_init = self.n_init
        if self._n_init == "warn":
            warnings.warn(
                "The default value of `n_init` will change from "
                f"{default_n_init} to 'auto' in 1.4. Set the value of `n_init`"
                " explicitly to suppress the warning",
                FutureWarning,
            )
            self._n_init = default_n_init
        if self._n_init == "auto":
            if self.init == "k-means++":
                self._n_init = 1
            else:
                self._n_init = default_n_init

        if _is_arraylike_not_scalar(self.init) and self._n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1
```
### 8 - examples/cluster/plot_kmeans_assumptions.py:

Start line: 1, End line: 85

```python
"""
====================================
Demonstration of k-means assumptions
====================================

This example is meant to illustrate situations where k-means produces
unintuitive and possibly undesirable clusters.

"""

import numpy as np
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10

# %%
# We can visualize the resulting data:

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

axs[0, 0].scatter(X[:, 0], X[:, 1], c=y)
axs[0, 0].set_title("Mixture of Gaussian Blobs")

axs[0, 1].scatter(X_aniso[:, 0], X_aniso[:, 1], c=y)
axs[0, 1].set_title("Anisotropically Distributed Blobs")

axs[1, 0].scatter(X_varied[:, 0], X_varied[:, 1], c=y_varied)
axs[1, 0].set_title("Unequal Variance")

axs[1, 1].scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_filtered)
axs[1, 1].set_title("Unevenly Sized Blobs")

plt.suptitle("Ground truth clusters").set_y(0.95)
plt.show()

# %%
# Fit models and plot results
# ---------------------------
#
# The previously generated data is now used to show how
# :class:`~sklearn.cluster.KMeans` behaves in the following scenarios:
#
# - Non-optimal number of clusters: in a real setting there is no uniquely
#   defined **true** number of clusters. An appropriate number of clusters has
#   to be decided from data-based criteria and knowledge of the intended goal.
# - Anisotropically distributed blobs: k-means consists of minimizing sample's
#   euclidean distances to the centroid of the cluster they are assigned to. As
#   a consequence, k-means is more appropriate for clusters that are isotropic
#   and normally distributed (i.e. spherical gaussians).
# - Unequal variance: k-means is equivalent to taking the maximum likelihood
#   estimator for a "mixture" of k gaussian distributions with the same
#   variances but with possibly different means.
# - Unevenly sized blobs: there is no theoretical result about k-means that
#   states that it requires similar cluster sizes to perform well, yet
#   minimizing euclidean distances does mean that the more sparse and
#   high-dimensional the problem is, the higher is the need to run the algorithm
#   with different centroid seeds to ensure a global minimal inertia.

from sklearn.cluster import KMeans
```
### 9 - examples/cluster/plot_kmeans_stability_low_dim_dense.py:

Start line: 69, End line: 135

```python
# Part 1: Quantitative evaluation of various init methods


plt.figure()
plots = []
legends = []

cases = [
    (KMeans, "k-means++", {}, "^-"),
    (KMeans, "random", {}, "o-"),
    (MiniBatchKMeans, "k-means++", {"max_no_improvement": 3}, "x-"),
    (MiniBatchKMeans, "random", {"max_no_improvement": 3, "init_size": 500}, "d-"),
]

for factory, init, params, format in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))

    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(
                n_clusters=n_clusters,
                init=init,
                random_state=run_id,
                n_init=n_init,
                **params,
            ).fit(X)
            inertia[i, run_id] = km.inertia_
    p = plt.errorbar(
        n_init_range, inertia.mean(axis=1), inertia.std(axis=1), fmt=format
    )
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel("n_init")
plt.ylabel("inertia")
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

# Part 2: Qualitative visual inspection of the convergence

X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
km = MiniBatchKMeans(
    n_clusters=n_clusters, init="random", n_init=1, random_state=random_state
).fit(X)

plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.nipy_spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0], X[my_members, 1], ".", c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=color,
        markeredgecolor="k",
        markersize=6,
    )
    plt.title(
        "Example cluster allocation with a single random init\nwith MiniBatchKMeans"
    )

plt.show()
```
### 10 - sklearn/cluster/_kmeans.py:

Start line: 1507, End line: 1521

```python
class KMeans(_BaseKMeans):

    def fit(self, X, y=None, sample_weight=None):
        # ... other code
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self._n_features_out = self.cluster_centers_.shape[0]
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self
```
### 12 - sklearn/cluster/_kmeans.py:

Start line: 678, End line: 732

```python
def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    # ... other code
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(
                X,
                sample_weight,
                centers,
                centers_new,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
            )

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels, n_threads)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(
                            f"Converged at iteration {i}: center shift "
                            f"{center_shift_tot} within tolerance {tol}."
                        )
                    break

            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(
                X,
                sample_weight,
                centers,
                centers,
                weight_in_clusters,
                labels,
                center_shift,
                n_threads,
                update_centers=False,
            )

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1
```
### 14 - sklearn/cluster/_kmeans.py:

Start line: 1336, End line: 1368

```python
class KMeans(_BaseKMeans):

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "copy_x": ["boolean"],
        "algorithm": [
            StrOptions({"lloyd", "elkan", "auto", "full"}, deprecated={"auto", "full"})
        ],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="warn",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm
```
### 15 - sklearn/cluster/_kmeans.py:

Start line: 941, End line: 1005

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def _init_centroids(
        self, X, x_squared_norms, init, random_state, init_size=None, n_centroids=None
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters)

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers
```
### 19 - sklearn/cluster/_kmeans.py:

Start line: 147, End line: 160

```python
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    }
)
def kmeans_plusplus(
    X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None
):
    # ... other code

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}."
        )

    random_state = check_random_state(random_state)

    # Call private k-means++
    centers, indices = _kmeans_plusplus(
        X, n_clusters, x_squared_norms, random_state, n_local_trials
    )

    return centers, indices
```
### 20 - sklearn/cluster/_kmeans.py:

Start line: 1646, End line: 1836

```python
class MiniBatchKMeans(_BaseKMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=1024
        Size of the mini batches.
        For faster computations, you can set the ``batch_size`` greater than
        256 * number of cores to enable parallelism on all cores.

        .. versionchanged:: 1.0
           `batch_size` default changed from 100 to 1024.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : 'auto' or int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the best of
        the `n_init` initializations as measured by inertia. Several runs are
        recommended for sparse high-dimensional problems (see
        :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs depends on the value of init:
        3 if using `init='random'`, 1 if using `init='k-means++'`.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` will change from 3 to `'auto'` in version 1.4.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a center to
        be reassigned. A higher value means that low count centers are more
        easily reassigned, which means that the model will take longer to
        converge, but should converge in a better clustering. However, too high
        a value may cause convergence issues, especially with a small batch
        size.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition if compute_labels is set to True. If compute_labels is set to
        False, it's an approximation of the inertia based on an exponentially
        weighted average of the batch inertiae.
        The inertia is defined as the sum of square distances of samples to
        their cluster center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations over the full dataset.

    n_steps_ : int
        Number of minibatches processed.

        .. versionadded:: 1.0

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    KMeans : The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    When there are too few points in the dataset, some centers may be
    duplicated, which means that a proper clustering in terms of the number
    of requesting clusters and the number of returned clusters will not
    always match. One solution is to set `reassignment_ratio=0`, which
    prevents reassignments of clusters that are too small.

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          n_init="auto")
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[2. , 1. ],
           [3.5, 4.5]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10,
    ...                          n_init="auto").fit(X)
    >>> kmeans.cluster_centers_
    array([[3.97727273, 2.43181818],
           [1.125     , 1.6       ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([1, 0], dtype=int32)
    """
```
### 22 - sklearn/cluster/_kmeans.py:

Start line: 1880, End line: 1907

```python
class MiniBatchKMeans(_BaseKMeans):

    def _check_params_vs_input(self, X):
        super()._check_params_vs_input(X, default_n_init=3)

        self._batch_size = min(self.batch_size, X.shape[0])

        # init_size
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self._batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                f"init_size={self._init_size} should be larger than "
                f"n_clusters={self.n_clusters}. Setting it to "
                "min(3*n_clusters, n_samples)",
                RuntimeWarning,
                stacklevel=2,
            )
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                "reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead."
            )
```
### 23 - sklearn/cluster/_kmeans.py:

Start line: 2105, End line: 2154

```python
class MiniBatchKMeans(_BaseKMeans):

    def fit(self, X, y=None, sample_weight=None):
        # ... other code

        with threadpool_limits(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.randint(0, n_samples, self._batch_size)

                # Perform the actual update step on the minibatch data
                batch_inertia = _mini_batch_step(
                    X=X[minibatch_indices],
                    sample_weight=sample_weight[minibatch_indices],
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                    i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]

        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )
        else:
            self.inertia_ = self._ewa_inertia * n_samples

        return self
```
### 24 - sklearn/cluster/_kmeans.py:

Start line: 272, End line: 418

```python
@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "sample_weight": ["array-like", None],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "copy_x": [bool],
        "algorithm": [
            StrOptions({"lloyd", "elkan", "auto", "full"}, deprecated={"auto", "full"})
        ],
        "return_n_iter": [bool],
    }
)
def k_means(
    X,
    n_clusters,
    *,
    sample_weight=None,
    init="k-means++",
    n_init="warn",
    max_iter=300,
    verbose=False,
    tol=1e-4,
    random_state=None,
    copy_x=True,
    algorithm="lloyd",
    return_n_iter=False,
):
    """Perform K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in `X`. If `None`, all observations
        are assigned equal weight.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        - `'k-means++'` : selects initial cluster centers for k-mean
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.
        - `'random'`: choose `n_clusters` observations (rows) at random from data
          for the initial centroids.
        - If an array is passed, it should be of shape `(n_clusters, n_features)`
          and gives the initial centers.
        - If a callable is passed, it should take arguments `X`, `n_clusters` and a
          random state and return an initialization.

    n_init : 'auto' or int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

        When `n_init='auto'`, the number of runs depends on the value of init:
        10 if using `init='random'`, 1 if using `init='k-means++'`.

        .. versionadded:: 1.2
           Added 'auto' option for `n_init`.

        .. versionchanged:: 1.4
           Default value for `n_init` will change from 10 to `'auto'` in version 1.4.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If `copy_x` is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        `copy_x` is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if `copy_x` is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

        .. versionchanged:: 0.18
            Added Elkan algorithm

        .. versionchanged:: 1.1
            Renamed "full" to "lloyd", and deprecated "auto" and "full".
            Changed "auto" to use "lloyd" instead of "elkan".

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        The `label[i]` is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """
    # ... other code
```
### 25 - sklearn/cluster/_kmeans.py:

Start line: 917, End line: 939

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X
```
### 27 - sklearn/cluster/_kmeans.py:

Start line: 1398, End line: 1506

```python
class KMeans(_BaseKMeans):

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        init_is_array_like = _is_arraylike_not_scalar(init)
        if init_is_array_like:
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if init_is_array_like:
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "elkan":
            kmeans_single = _kmeans_single_elkan
        else:
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        # ... other code
```
### 28 - sklearn/cluster/_bisect_k_means.py:

Start line: 207, End line: 251

```python
class BisectingKMeans(_BaseKMeans):

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "init": [StrOptions({"k-means++", "random"}), callable],
        "copy_x": ["boolean"],
        "algorithm": [StrOptions({"lloyd", "elkan"})],
        "bisecting_strategy": [StrOptions({"biggest_inertia", "largest_cluster"})],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="random",
        n_init=1,
        random_state=None,
        max_iter=300,
        verbose=0,
        tol=1e-4,
        copy_x=True,
        algorithm="lloyd",
        bisecting_strategy="biggest_inertia",
    ):

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        self.copy_x = copy_x
        self.algorithm = algorithm
        self.bisecting_strategy = bisecting_strategy

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "BisectingKMeans is known to have a memory leak on Windows "
            "with MKL, when there are less chunks than available "
            "threads. You can avoid it by setting the environment"
            f" variable OMP_NUM_THREADS={n_active_threads}."
        )
```
### 29 - sklearn/cluster/_kmeans.py:

Start line: 58, End line: 145

```python
###############################################################################
# Initialization heuristic


@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "x_squared_norms": ["array-like", None],
        "random_state": ["random_state"],
        "n_local_trials": [Interval(Integral, 1, None, closed="left"), None],
    }
)
def kmeans_plusplus(
    X, n_clusters, *, x_squared_norms=None, random_state=None, n_local_trials=None
):
    """Init n_clusters seeds according to k-means++.

    .. versionadded:: 0.24

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.

    n_clusters : int
        The number of centroids to initialize.

    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.

    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)) which is the recommended setting.
        Setting to 1 disables the greedy cluster selection and recovers the
        vanilla k-means++ algorithm which was empirically shown to work less
        well than its greedy variant.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Examples
    --------

    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  4],
           [ 1,  0]])
    >>> indices
    array([4, 2])
    """
    # Check data
    check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32])

    if X.shape[0] < n_clusters:
        raise ValueError(
            f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}."
        )

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms, dtype=X.dtype, ensure_2d=False)
    # ... other code
```
### 30 - sklearn/cluster/_kmeans.py:

Start line: 163, End line: 223

```python
def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    # ... other code
```
### 31 - sklearn/cluster/_kmeans.py:

Start line: 735, End line: 802

```python
def _labels_inertia(X, sample_weight, centers, n_threads=1, return_inertia=True):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    return_inertia : bool, default=True
        Whether to compute and return the inertia.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        Inertia is only returned if return_inertia is True.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    center_shift = np.zeros(n_clusters, dtype=centers.dtype)

    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    _labels(
        X,
        sample_weight,
        centers,
        centers_new=None,
        weight_in_clusters=None,
        labels=labels,
        center_shift=center_shift,
        n_threads=n_threads,
        update_centers=False,
    )

    if return_inertia:
        inertia = _inertia(X, sample_weight, centers, labels, n_threads)
        return labels, inertia

    return labels
```
### 33 - sklearn/cluster/_kmeans.py:

Start line: 1524, End line: 1609

```python
def _mini_batch_step(
    X,
    sample_weight,
    centers,
    centers_new,
    weight_sums,
    random_state,
    random_reassign=False,
    reassignment_ratio=0.01,
    verbose=False,
    n_threads=1,
):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. If sparse, must be in CSR format.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
        The inertia is computed after finding the labels and before updating
        the centers.
    """
    # Perform label assignment to nearest centers
    # For better efficiency, it's better to run _mini_batch_step in a
    # threadpool_limit context than using _labels_inertia_threadpool_limit here
    labels, inertia = _labels_inertia(X, sample_weight, centers, n_threads=n_threads)

    # Update centers according to the labels
    if sp.issparse(X):
        _minibatch_update_sparse(
            X, sample_weight, centers, centers_new, weight_sums, labels, n_threads
        )
    else:
        _minibatch_update_dense(
            X,
            sample_weight,
            centers,
            centers_new,
            weight_sums,
            labels,
            n_threads,
        )

    # Reassign clusters that have very low weight
    # ... other code
```
### 34 - sklearn/cluster/_bisect_k_means.py:

Start line: 253, End line: 283

```python
class BisectingKMeans(_BaseKMeans):

    def _inertia_per_cluster(self, X, centers, labels, sample_weight):
        """Calculate the sum of squared errors (inertia) per cluster.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The input samples.

        centers : ndarray of shape (n_clusters, n_features)
            The cluster centers.

        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        Returns
        -------
        inertia_per_cluster : ndarray of shape (n_clusters,)
            Sum of squared errors (inertia) for each cluster.
        """
        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense

        inertia_per_cluster = np.empty(centers.shape[1])
        for label in range(centers.shape[0]):
            inertia_per_cluster[label] = _inertia(
                X, sample_weight, centers, labels, self._n_threads, single_label=label
            )

        return inertia_per_cluster
```
### 37 - sklearn/cluster/_kmeans.py:

Start line: 2001, End line: 2103

```python
class MiniBatchKMeans(_BaseKMeans):

    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]

        # perform several inits with random subsets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
            )

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid,
                sample_weight_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size
        # ... other code
```
### 38 - sklearn/cluster/_kmeans.py:

Start line: 1838, End line: 1878

```python
class MiniBatchKMeans(_BaseKMeans):

    _parameter_constraints: dict = {
        **_BaseKMeans._parameter_constraints,
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "compute_labels": ["boolean"],
        "max_no_improvement": [Interval(Integral, 0, None, closed="left"), None],
        "init_size": [Interval(Integral, 1, None, closed="left"), None],
        "reassignment_ratio": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        max_iter=100,
        batch_size=1024,
        verbose=0,
        compute_labels=True,
        random_state=None,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init="warn",
        reassignment_ratio=0.01,
    ):

        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
        )

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
```
### 39 - sklearn/cluster/_bisect_k_means.py:

Start line: 76, End line: 205

```python
class BisectingKMeans(_BaseKMeans):
    """Bisecting K-Means clustering.

    Read more in the :ref:`User Guide <bisect_k_means>`.

    .. versionadded:: 1.1

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'} or callable, default='random'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=1
        Number of time the inner k-means algorithm will be run with different
        centroid seeds in each bisection.
        That will result producing for each bisection best output of n_init
        consecutive runs in terms of inertia.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization
        in inner K-Means. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    max_iter : int, default=300
        Maximum number of iterations of the inner k-means algorithm at each
        bisection.

    verbose : int, default=0
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations  to declare
        convergence. Used in inner k-means algorithm at each bisection to pick
        best possible clusters.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan"}, default="lloyd"
        Inner K-means algorithm used in bisection.
        The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

    bisecting_strategy : {"biggest_inertia", "largest_cluster"},\
            default="biggest_inertia"
        Defines how bisection should be performed:

         - "biggest_inertia" means that BisectingKMeans will always check
            all calculated cluster for cluster with biggest SSE
            (Sum of squared errors) and bisect it. This approach concentrates on
            precision, but may be costly in terms of execution time (especially for
            larger amount of data points).

         - "largest_cluster" - BisectingKMeans will always split cluster with
            largest amount of points assigned to it from all clusters
            previously calculated. That should work faster than picking by SSE
            ('biggest_inertia') and may produce similar results in most cases.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    KMeans : Original implementation of K-Means algorithm.

    Notes
    -----
    It might be inefficient when n_cluster is less than 3, due to unnecessary
    calculations for that case.

    Examples
    --------
    >>> from sklearn.cluster import BisectingKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0],
    ...               [10, 6], [10, 8], [10, 10]])
    >>> bisect_means = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
    >>> bisect_means.labels_
    array([2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=int32)
    >>> bisect_means.predict([[0, 0], [12, 3]])
    array([2, 0], dtype=int32)
    >>> bisect_means.cluster_centers_
    array([[10.,  2.],
           [10.,  8.],
           [ 1., 2.]])
    """
```
### 47 - sklearn/cluster/_kmeans.py:

Start line: 1985, End line: 1999

```python
class MiniBatchKMeans(_BaseKMeans):

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.

        If there are empty clusters we always want to reassign.
        """
        self._n_since_last_reassign += self._batch_size
        if (self._counts == 0).any() or self._n_since_last_reassign >= (
            10 * self.n_clusters
        ):
            self._n_since_last_reassign = 0
            return True
        return False
```
### 52 - sklearn/cluster/_bisect_k_means.py:

Start line: 1, End line: 20

```python
"""Bisecting K-means clustering."""

import warnings

import numpy as np
import scipy.sparse as sp

from ._kmeans import _BaseKMeans
from ._kmeans import _kmeans_single_elkan
from ._kmeans import _kmeans_single_lloyd
from ._kmeans import _labels_inertia_threadpool_limit
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse
from ..utils.extmath import row_norms
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_sample_weight
from ..utils.validation import check_random_state
from ..utils._param_validation import StrOptions
```
### 53 - sklearn/cluster/_bisect_k_means.py:

Start line: 346, End line: 437

```python
class BisectingKMeans(_BaseKMeans):

    def fit(self, X, y=None, sample_weight=None):
        """Compute bisecting k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

            Training instances to cluster.

            .. note:: The data will be converted to C ordering,
                which will cause a memory copy
                if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params_vs_input(X)

        self._random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        if self.algorithm == "lloyd" or self.n_clusters == 1:
            self._kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            self._kmeans_single = _kmeans_single_elkan

        # Subtract of mean of X for more accurate distance computations
        if not sp.issparse(X):
            self._X_mean = X.mean(axis=0)
            X -= self._X_mean

        # Initialize the hierarchical clusters tree
        self._bisecting_tree = _BisectingTree(
            indices=np.arange(X.shape[0]),
            center=X.mean(axis=0),
            score=0,
        )

        x_squared_norms = row_norms(X, squared=True)

        for _ in range(self.n_clusters - 1):
            # Chose cluster to bisect
            cluster_to_bisect = self._bisecting_tree.get_cluster_to_bisect()

            # Split this cluster into 2 subclusters
            self._bisect(X, x_squared_norms, sample_weight, cluster_to_bisect)

        # Aggregate final labels and centers from the bisecting tree
        self.labels_ = np.full(X.shape[0], -1, dtype=np.int32)
        self.cluster_centers_ = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        for i, cluster_node in enumerate(self._bisecting_tree.iter_leaves()):
            self.labels_[cluster_node.indices] = i
            self.cluster_centers_[i] = cluster_node.center
            cluster_node.label = i  # label final clusters for future prediction
            cluster_node.indices = None  # release memory

        # Restore original data
        if not sp.issparse(X):
            X += self._X_mean
            self.cluster_centers_ += self._X_mean

        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
        self.inertia_ = _inertia(
            X, sample_weight, self.cluster_centers_, self.labels_, self._n_threads
        )

        self._n_features_out = self.cluster_centers_.shape[0]

        return self
```
### 54 - sklearn/cluster/_kmeans.py:

Start line: 224, End line: 269

```python
def _kmeans_plusplus(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    # ... other code
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


###############################################################################
# K-means batch estimation by EM (expectation maximization)


def _tolerance(X, tol):
    """Return a tolerance which is dependent on the dataset."""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol
```
### 59 - sklearn/cluster/_kmeans.py:

Start line: 1007, End line: 1030

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_
```
### 60 - sklearn/cluster/_kmeans.py:

Start line: 436, End line: 528

```python
def _kmeans_single_elkan(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means elkan, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(
        np.asarray(center_half_distances), kth=1, axis=0
    )[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(
        X,
        centers,
        center_half_distances,
        labels,
        upper_bounds,
        lower_bounds,
        n_threads=n_threads,
    )

    strict_convergence = False
    # ... other code
```
### 62 - sklearn/cluster/_kmeans.py:

Start line: 805, End line: 851

```python
def _labels_inertia_threadpool_limit(
    X, sample_weight, centers, n_threads=1, return_inertia=True
):
    """Same as _labels_inertia but in a threadpool_limits context."""
    with threadpool_limits(limits=1, user_api="blas"):
        result = _labels_inertia(X, sample_weight, centers, n_threads, return_inertia)

    return result


class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):
    """Base class for KMeans and MiniBatchKMeans"""

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters,
        *,
        init,
        n_init,
        max_iter,
        tol,
        verbose,
        random_state,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
```
### 66 - sklearn/cluster/_kmeans.py:

Start line: 890, End line: 915

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        """Issue an estimator specific warning when vcomp and mkl are both present

        This method is called by `_check_mkl_vcomp`.
        """

    def _check_mkl_vcomp(self, X, n_samples):
        """Check when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if n_active_threads < self._n_threads:
            modules = threadpool_info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)
```
### 71 - sklearn/cluster/_kmeans.py:

Start line: 1032, End line: 1078

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def predict(self, X, sample_weight="deprecated"):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. deprecated:: 1.3
               The parameter `sample_weight` is deprecated in version 1.3
               and will be removed in 1.5.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        if not (isinstance(sample_weight, str) and sample_weight == "deprecated"):
            warnings.warn(
                "'sample_weight' was deprecated in version 1.3 and "
                "will be removed in 1.5.",
                FutureWarning,
            )
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        else:
            sample_weight = _check_sample_weight(None, X, dtype=X.dtype)

        labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )

        return labels
```
### 72 - sklearn/cluster/_bicluster.py:

Start line: 181, End line: 199

```python
class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):

    def _k_means(self, data, n_clusters):
        if self.mini_batch:
            model = MiniBatchKMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        else:
            model = KMeans(
                n_clusters,
                init=self.init,
                n_init=self.n_init,
                random_state=self.random_state,
            )
        model.fit(data)
        centroid = model.cluster_centers_
        labels = model.labels_
        return centroid, labels
```
### 76 - sklearn/cluster/_kmeans.py:

Start line: 601, End line: 677

```python
def _kmeans_single_lloyd(
    X,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-4,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubscription.
    # ... other code
```
### 82 - sklearn/cluster/_kmeans.py:

Start line: 1909, End line: 1918

```python
class MiniBatchKMeans(_BaseKMeans):

    def _warn_mkl_vcomp(self, n_active_threads):
        """Warn when vcomp and mkl are both present"""
        warnings.warn(
            "MiniBatchKMeans is known to have a memory leak on "
            "Windows with MKL, when there are less chunks than "
            "available threads. You can prevent it by setting "
            f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
            "setting the environment variable "
            f"OMP_NUM_THREADS={n_active_threads}"
        )
```
### 89 - sklearn/cluster/_kmeans.py:

Start line: 2156, End line: 2257

```python
class MiniBatchKMeans(_BaseKMeans):

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self : object
            Return updated estimator.
        """
        has_centers = hasattr(self, "cluster_centers_")

        if not has_centers:
            self._validate_params()

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )

        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, "n_steps_", 0)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if not has_centers:
            # this instance has not been fitted yet (fit or partial_fit)
            self._check_params_vs_input(X)
            self._n_threads = _openmp_effective_n_threads()

            # Validate init array
            init = self.init
            if _is_arraylike_not_scalar(init):
                init = check_array(init, dtype=X.dtype, copy=True, order="C")
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size,
            )

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            # Initialize number of samples seen since last reassignment
            self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            _mini_batch_step(
                X,
                sample_weight=sample_weight,
                centers=self.cluster_centers_,
                centers_new=self.cluster_centers_,
                weight_sums=self._counts,
                random_state=self._random_state,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
                n_threads=self._n_threads,
            )

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )

        self.n_steps_ += 1
        self._n_features_out = self.cluster_centers_.shape[0]

        return self
```
### 91 - sklearn/cluster/_kmeans.py:

Start line: 1080, End line: 1102

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)
```
### 97 - sklearn/cluster/_bisect_k_means.py:

Start line: 285, End line: 344

```python
class BisectingKMeans(_BaseKMeans):

    def _bisect(self, X, x_squared_norms, sample_weight, cluster_to_bisect):
        """Split a cluster into 2 subsclusters.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        cluster_to_bisect : _BisectingTree node object
            The cluster node to split.
        """
        X = X[cluster_to_bisect.indices]
        x_squared_norms = x_squared_norms[cluster_to_bisect.indices]
        sample_weight = sample_weight[cluster_to_bisect.indices]

        best_inertia = None

        # Split samples in X into 2 clusters.
        # Repeating `n_init` times to obtain best clusters
        for _ in range(self.n_init):
            centers_init = self._init_centroids(
                X, x_squared_norms, self.init, self._random_state, n_centroids=2
            )

            labels, inertia, centers, _ = self._kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self.tol,
                n_threads=self._n_threads,
            )

            # allow small tolerance on the inertia to accommodate for
            # non-deterministic rounding errors due to parallel computation
            if best_inertia is None or inertia < best_inertia * (1 - 1e-6):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia

        if self.verbose:
            print(f"New centroids from bisection: {best_centers}")

        if self.bisecting_strategy == "biggest_inertia":
            scores = self._inertia_per_cluster(
                X, best_centers, best_labels, sample_weight
            )
        else:  # bisecting_strategy == "largest_cluster"
            # Using minlength to make sure that we have the counts for both labels even
            # if all samples are labelled 0.
            scores = np.bincount(best_labels, minlength=2)

        cluster_to_bisect.split(best_labels, best_centers, scores)
```
### 117 - sklearn/cluster/_kmeans.py:

Start line: 1104, End line: 1128

```python
class _BaseKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC
):

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return euclidean_distances(X, self.cluster_centers_)
```
### 119 - sklearn/cluster/_bisect_k_means.py:

Start line: 471, End line: 524

```python
class BisectingKMeans(_BaseKMeans):

    def _predict_recursive(self, X, sample_weight, cluster_node):
        """Predict recursively by going down the hierarchical tree.

        Parameters
        ----------
        X : {ndarray, csr_matrix} of shape (n_samples, n_features)
            The data points, currently assigned to `cluster_node`, to predict between
            the subclusters of this node.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X.

        cluster_node : _BisectingTree node object
            The cluster node of the hierarchical tree.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if cluster_node.left is None:
            # This cluster has no subcluster. Labels are just the label of the cluster.
            return np.full(X.shape[0], cluster_node.label, dtype=np.int32)

        # Determine if data points belong to the left or right subcluster
        centers = np.vstack((cluster_node.left.center, cluster_node.right.center))
        if hasattr(self, "_X_mean"):
            centers += self._X_mean

        cluster_labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            centers,
            self._n_threads,
            return_inertia=False,
        )
        mask = cluster_labels == 0

        # Compute the labels for each subset of the data points.
        labels = np.full(X.shape[0], -1, dtype=np.int32)

        labels[mask] = self._predict_recursive(
            X[mask], sample_weight[mask], cluster_node.left
        )

        labels[~mask] = self._predict_recursive(
            X[~mask], sample_weight[~mask], cluster_node.right
        )

        return labels

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}
```
### 124 - sklearn/cluster/_kmeans.py:

Start line: 1920, End line: 1983

```python
class MiniBatchKMeans(_BaseKMeans):

    def _mini_batch_convergence(
        self, step, n_steps, n_samples, centers_squared_diff, batch_inertia
    ):
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self._batch_size

        # count steps starting from 1 for user friendly verbose mode.
        step = step + 1

        # Ignore first iteration because it's inertia from initialization.
        if step == 1:
            if self.verbose:
                print(
                    f"Minibatch step {step}/{n_steps}: mean batch "
                    f"inertia: {batch_inertia}"
                )
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        if self._ewa_inertia is None:
            self._ewa_inertia = batch_inertia
        else:
            alpha = self._batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(
                f"Minibatch step {step}/{n_steps}: mean batch inertia: "
                f"{batch_inertia}, ewa inertia: {self._ewa_inertia}"
            )

        # Early stopping based on absolute tolerance on squared change of
        # centers position
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at step {step}/{n_steps}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
            self._no_improvement = 0
            self._ewa_inertia_min = self._ewa_inertia
        else:
            self._no_improvement += 1

        if (
            self.max_no_improvement is not None
            and self._no_improvement >= self.max_no_improvement
        ):
            if self.verbose:
                print(
                    "Converged (lack of improvement in inertia) at step "
                    f"{step}/{n_steps}"
                )
            return True

        return False
```

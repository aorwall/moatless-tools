# scikit-learn__scikit-learn-12421

| **scikit-learn/scikit-learn** | `013d295a13721ffade7ac321437c6d4458a64c7d` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4253 |
| **Any found context length** | 1703 |
| **Avg pos** | 43.0 |
| **Min pos** | 1 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/cluster/optics_.py b/sklearn/cluster/optics_.py
--- a/sklearn/cluster/optics_.py
+++ b/sklearn/cluster/optics_.py
@@ -39,9 +39,8 @@ def optics(X, min_samples=5, max_eps=np.inf, metric='minkowski',
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order. It also does not employ a heap to manage the expansion
-    candiates, but rather uses numpy masked arrays. This can be potentially
-    slower with some parameters (at the benefit from using fast numpy code).
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -199,7 +198,8 @@ class OPTICS(BaseEstimator, ClusterMixin):
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order.
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -430,7 +430,11 @@ def fit(self, X, y=None):
                                 n_jobs=self.n_jobs)
 
         nbrs.fit(X)
+        # Here we first do a kNN query for each point, this differs from
+        # the original OPTICS that only used epsilon range queries.
         self.core_distances_ = self._compute_core_distances_(X, nbrs)
+        # OPTICS puts an upper limit on these, use inf for undefined.
+        self.core_distances_[self.core_distances_ > self.max_eps] = np.inf
         self.ordering_ = self._calculate_optics_order(X, nbrs)
 
         indices_, self.labels_ = _extract_optics(self.ordering_,
@@ -445,7 +449,6 @@ def fit(self, X, y=None):
         return self
 
     # OPTICS helper functions
-
     def _compute_core_distances_(self, X, neighbors, working_memory=None):
         """Compute the k-th nearest neighbor of each sample
 
@@ -485,37 +488,38 @@ def _compute_core_distances_(self, X, neighbors, working_memory=None):
     def _calculate_optics_order(self, X, nbrs):
         # Main OPTICS loop. Not parallelizable. The order that entries are
         # written to the 'ordering_' list is important!
+        # Note that this implementation is O(n^2) theoretically, but
+        # supposedly with very low constant factors.
         processed = np.zeros(X.shape[0], dtype=bool)
         ordering = np.zeros(X.shape[0], dtype=int)
-        ordering_idx = 0
-        for point in range(X.shape[0]):
-            if processed[point]:
-                continue
-            if self.core_distances_[point] <= self.max_eps:
-                while not processed[point]:
-                    processed[point] = True
-                    ordering[ordering_idx] = point
-                    ordering_idx += 1
-                    point = self._set_reach_dist(point, processed, X, nbrs)
-            else:  # For very noisy points
-                ordering[ordering_idx] = point
-                ordering_idx += 1
-                processed[point] = True
+        for ordering_idx in range(X.shape[0]):
+            # Choose next based on smallest reachability distance
+            # (And prefer smaller ids on ties, possibly np.inf!)
+            index = np.where(processed == 0)[0]
+            point = index[np.argmin(self.reachability_[index])]
+
+            processed[point] = True
+            ordering[ordering_idx] = point
+            if self.core_distances_[point] != np.inf:
+                self._set_reach_dist(point, processed, X, nbrs)
         return ordering
 
     def _set_reach_dist(self, point_index, processed, X, nbrs):
         P = X[point_index:point_index + 1]
+        # Assume that radius_neighbors is faster without distances
+        # and we don't need all distances, nevertheless, this means
+        # we may be doing some work twice.
         indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                         return_distance=False)[0]
 
         # Getting indices of neighbors that have not been processed
         unproc = np.compress((~np.take(processed, indices)).ravel(),
                              indices, axis=0)
-        # Keep n_jobs = 1 in the following lines...please
+        # Neighbors of current point are already processed.
         if not unproc.size:
-            # Everything is already processed. Return to main loop
-            return point_index
+            return
 
+        # Only compute distances to unprocessed neighbors:
         if self.metric == 'precomputed':
             dists = X[point_index, unproc]
         else:
@@ -527,12 +531,6 @@ def _set_reach_dist(self, point_index, processed, X, nbrs):
         self.reachability_[unproc[improved]] = rdists[improved]
         self.predecessor_[unproc[improved]] = point_index
 
-        # Choose next based on smallest reachability distance
-        # (And prefer smaller ids on ties).
-        # All unprocessed points qualify, not just new neighbors ("unproc")
-        return (np.ma.array(self.reachability_, mask=processed)
-                .argmin(fill_value=np.inf))
-
     def extract_dbscan(self, eps):
         """Performs DBSCAN extraction for an arbitrary epsilon.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/cluster/optics_.py | 42 | 44 | 8 | 1 | 5659
| sklearn/cluster/optics_.py | 202 | 202 | 4 | 1 | 4253
| sklearn/cluster/optics_.py | 433 | 433 | 4 | 1 | 4253
| sklearn/cluster/optics_.py | 448 | 448 | 7 | 1 | 5491
| sklearn/cluster/optics_.py | 488 | 515 | 4 | 1 | 4253
| sklearn/cluster/optics_.py | 530 | 535 | 4 | 1 | 4253


## Problem Statement

```
OPTICS: self.core_distances_ inconsistent with documentation&R implementation
In the doc, we state that ``Points which will never be core have a distance of inf.``, but it's not the case.
Result from scikit-learn:
\`\`\`
import numpy as np
from sklearn.cluster import OPTICS
X = np.array([-5, -2, -4.8, -1.8, -5.2, -2.2, 100, 200, 4, 2, 3.8, 1.8, 4.2, 2.2])
X = X.reshape(-1, 2)
clust = OPTICS(min_samples=3, max_bound=1)
clust.fit(X)
clust.core_distances_
\`\`\`
\`\`\`
array([  0.28284271,   0.56568542,   0.56568542, 220.04544985, 
         0.28284271,   0.56568542,   0.56568542])
\`\`\`
Result from R:
\`\`\`
x <- matrix(c(-5, -2, -4.8, -1.8, -5.2, -2.2, 100, 200,
              4, 2, 3.8, 1.8, 4.2, 2.2), ncol=2, byrow=TRUE)
result <- optics(x, eps=1, minPts=3)
result$coredist
\`\`\`
\`\`\`
[1] 0.2828427 0.5656854 0.5656854       Inf 0.2828427 0.5656854 0.5656854
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/cluster/optics_.py** | 191 | 354| 1703 | 1703 | 8247 | 
| 2 | **1 sklearn/cluster/optics_.py** | 506 | 534| 324 | 2027 | 8247 | 
| **-> 3 <-** | **1 sklearn/cluster/optics_.py** | 378 | 445| 536 | 2563 | 8247 | 
| **-> 4 <-** | **1 sklearn/cluster/optics_.py** | 31 | 573| 1690 | 4253 | 8247 | 
| 5 | 2 examples/cluster/plot_optics.py | 1 | 80| 738 | 4991 | 9347 | 
| 6 | **2 sklearn/cluster/optics_.py** | 485 | 504| 204 | 5195 | 9347 | 
| **-> 7 <-** | **2 sklearn/cluster/optics_.py** | 447 | 483| 296 | 5491 | 9347 | 
| **-> 8 <-** | **2 sklearn/cluster/optics_.py** | 1 | 187| 168 | 5659 | 9347 | 
| 9 | **2 sklearn/cluster/optics_.py** | 611 | 693| 808 | 6467 | 9347 | 
| 10 | **2 sklearn/cluster/optics_.py** | 356 | 376| 215 | 6682 | 9347 | 
| 11 | **2 sklearn/cluster/optics_.py** | 576 | 608| 277 | 6959 | 9347 | 
| 12 | **2 sklearn/cluster/optics_.py** | 536 | 573| 358 | 7317 | 9347 | 
| 13 | 2 examples/cluster/plot_optics.py | 81 | 102| 325 | 7642 | 9347 | 
| 14 | **2 sklearn/cluster/optics_.py** | 730 | 745| 157 | 7799 | 9347 | 
| 15 | 3 examples/covariance/plot_mahalanobis_distances.py | 79 | 134| 726 | 8525 | 11010 | 
| 16 | 4 sklearn/manifold/mds.py | 92 | 132| 424 | 8949 | 14871 | 
| 17 | 5 sklearn/metrics/pairwise.py | 1402 | 1428| 192 | 9141 | 27887 | 
| 18 | 6 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 9652 | 28420 | 
| 19 | **6 sklearn/cluster/optics_.py** | 748 | 763| 193 | 9845 | 28420 | 
| 20 | 6 sklearn/manifold/mds.py | 234 | 276| 436 | 10281 | 28420 | 
| 21 | 7 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 10644 | 29538 | 
| 22 | 7 examples/covariance/plot_mahalanobis_distances.py | 135 | 145| 147 | 10791 | 29538 | 
| 23 | **7 sklearn/cluster/optics_.py** | 845 | 886| 392 | 11183 | 29538 | 
| 24 | 7 sklearn/metrics/pairwise.py | 262 | 363| 855 | 12038 | 29538 | 
| 25 | 7 examples/covariance/plot_mahalanobis_distances.py | 1 | 78| 789 | 12827 | 29538 | 
| 26 | 8 sklearn/neighbors/base.py | 680 | 759| 685 | 13512 | 37110 | 
| 27 | 8 sklearn/metrics/pairwise.py | 366 | 443| 637 | 14149 | 37110 | 
| 28 | 8 sklearn/metrics/pairwise.py | 1369 | 1399| 322 | 14471 | 37110 | 
| 29 | 9 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 15256 | 38782 | 
| 30 | 10 examples/plot_johnson_lindenstrauss_bound.py | 162 | 200| 376 | 15632 | 40562 | 
| 31 | 11 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 16299 | 41229 | 
| 32 | 12 examples/cluster/plot_cluster_comparison.py | 1 | 86| 698 | 16997 | 42814 | 
| 33 | 13 sklearn/metrics/cluster/unsupervised.py | 228 | 236| 140 | 17137 | 45940 | 
| 34 | 14 sklearn/utils/estimator_checks.py | 1269 | 1327| 600 | 17737 | 66933 | 
| 35 | 15 examples/manifold/plot_mds.py | 1 | 89| 743 | 18480 | 67699 | 
| 36 | 16 examples/covariance/plot_lw_vs_oas.py | 1 | 84| 786 | 19266 | 68485 | 
| 37 | 17 sklearn/cluster/dbscan_.py | 132 | 182| 523 | 19789 | 71805 | 
| 38 | 17 sklearn/metrics/pairwise.py | 164 | 259| 849 | 20638 | 71805 | 
| 39 | 18 examples/cluster/plot_cluster_iris.py | 1 | 93| 751 | 21389 | 72564 | 
| 40 | 18 sklearn/metrics/pairwise.py | 1014 | 1025| 110 | 21499 | 72564 | 
| 41 | 18 examples/cluster/plot_cluster_comparison.py | 88 | 189| 887 | 22386 | 72564 | 
| 42 | 18 sklearn/metrics/pairwise.py | 601 | 632| 219 | 22605 | 72564 | 
| 43 | 19 sklearn/cluster/k_means_.py | 583 | 628| 365 | 22970 | 87641 | 
| 44 | 20 sklearn/linear_model/omp.py | 346 | 404| 625 | 23595 | 95716 | 
| 45 | 21 examples/applications/plot_outlier_detection_housing.py | 1 | 78| 739 | 24334 | 97117 | 
| 46 | 21 sklearn/cluster/dbscan_.py | 185 | 302| 1117 | 25451 | 97117 | 
| 47 | 21 sklearn/neighbors/base.py | 602 | 678| 714 | 26165 | 97117 | 
| 48 | 21 sklearn/metrics/pairwise.py | 446 | 516| 629 | 26794 | 97117 | 
| 49 | 22 examples/cluster/plot_dbscan.py | 1 | 76| 625 | 27419 | 97742 | 
| 50 | **22 sklearn/cluster/optics_.py** | 696 | 727| 246 | 27665 | 97742 | 
| 51 | 23 examples/linear_model/plot_ols_ridge_variance.py | 1 | 68| 484 | 28149 | 98234 | 
| 52 | 24 examples/plot_anomaly_comparison.py | 79 | 151| 763 | 28912 | 99739 | 
| 53 | 25 benchmarks/bench_plot_omp_lars.py | 1 | 101| 858 | 29770 | 100905 | 
| 54 | 25 sklearn/metrics/pairwise.py | 1 | 32| 146 | 29916 | 100905 | 
| 55 | 25 sklearn/linear_model/omp.py | 92 | 136| 508 | 30424 | 100905 | 
| 56 | 25 sklearn/linear_model/omp.py | 1 | 23| 121 | 30545 | 100905 | 
| 57 | 25 examples/plot_johnson_lindenstrauss_bound.py | 94 | 160| 646 | 31191 | 100905 | 
| 58 | 25 sklearn/linear_model/omp.py | 265 | 345| 755 | 31946 | 100905 | 
| 59 | 26 sklearn/cluster/birch.py | 5 | 19| 113 | 32059 | 106208 | 
| 60 | 27 sklearn/covariance/robust_covariance.py | 387 | 413| 365 | 32424 | 113294 | 
| 61 | 28 examples/neighbors/plot_kde_1d.py | 69 | 145| 741 | 33165 | 114799 | 
| 62 | 29 sklearn/covariance/graph_lasso_.py | 202 | 277| 858 | 34023 | 123151 | 
| 63 | 29 sklearn/linear_model/omp.py | 542 | 618| 716 | 34739 | 123151 | 
| 64 | 30 examples/neighbors/plot_lof_novelty_detection.py | 68 | 84| 177 | 34916 | 124081 | 
| 65 | 31 examples/applications/wikipedia_principal_eigenvector.py | 189 | 232| 376 | 35292 | 126006 | 
| 66 | 31 examples/cluster/plot_agglomerative_clustering_metrics.py | 1 | 91| 733 | 36025 | 126006 | 
| 67 | 32 sklearn/cluster/bicluster.py | 283 | 311| 285 | 36310 | 130938 | 
| 68 | 32 examples/applications/plot_outlier_detection_housing.py | 79 | 134| 633 | 36943 | 130938 | 
| 69 | 32 sklearn/metrics/pairwise.py | 1073 | 1102| 273 | 37216 | 130938 | 
| 70 | 33 benchmarks/bench_plot_randomized_svd.py | 297 | 340| 517 | 37733 | 135318 | 
| 71 | 34 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 38292 | 136453 | 
| 72 | 35 sklearn/neighbors/lof.py | 478 | 505| 274 | 38566 | 140967 | 
| 73 | 35 benchmarks/bench_plot_randomized_svd.py | 199 | 223| 284 | 38850 | 140967 | 
| 74 | 35 sklearn/covariance/robust_covariance.py | 414 | 510| 1054 | 39904 | 140967 | 
| 75 | 36 examples/cluster/plot_affinity_propagation.py | 1 | 63| 526 | 40430 | 141493 | 
| 76 | 36 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 41006 | 141493 | 
| 77 | 36 sklearn/manifold/mds.py | 1 | 91| 727 | 41733 | 141493 | 
| 78 | 37 examples/cluster/plot_kmeans_digits.py | 1 | 56| 319 | 42052 | 142537 | 
| 79 | 38 sklearn/neighbors/classification.py | 325 | 336| 144 | 42196 | 145936 | 
| 80 | 39 sklearn/covariance/elliptic_envelope.py | 153 | 187| 247 | 42443 | 147602 | 
| 81 | 40 benchmarks/bench_plot_parallel_pairwise.py | 3 | 47| 299 | 42742 | 147925 | 
| 82 | 41 sklearn/linear_model/coordinate_descent.py | 1035 | 1048| 226 | 42968 | 169028 | 
| 83 | 42 examples/covariance/plot_covariance_estimation.py | 1 | 86| 761 | 43729 | 170173 | 
| 84 | 43 sklearn/manifold/t_sne.py | 397 | 905| 791 | 44520 | 178594 | 
| 85 | 44 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 45026 | 179653 | 
| 86 | 45 examples/text/plot_document_clustering.py | 191 | 230| 379 | 45405 | 181539 | 
| 87 | 46 examples/linear_model/plot_omp.py | 1 | 83| 560 | 45965 | 182099 | 
| 88 | 46 sklearn/metrics/pairwise.py | 1105 | 1110| 131 | 46096 | 182099 | 
| 89 | 47 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 46855 | 183582 | 
| 90 | 48 sklearn/cluster/spectral.py | 446 | 499| 462 | 47317 | 188091 | 
| 91 | 49 examples/linear_model/plot_ridge_path.py | 1 | 67| 434 | 47751 | 188554 | 
| 92 | 50 sklearn/decomposition/online_lda.py | 96 | 133| 445 | 48196 | 195119 | 
| 93 | 50 sklearn/manifold/mds.py | 382 | 400| 155 | 48351 | 195119 | 
| 94 | 50 sklearn/neighbors/lof.py | 161 | 191| 254 | 48605 | 195119 | 
| 95 | 51 examples/linear_model/plot_huber_vs_ridge.py | 1 | 66| 592 | 49197 | 195733 | 
| 96 | 52 examples/cluster/plot_digits_linkage.py | 57 | 74| 189 | 49386 | 196462 | 
| 97 | 52 sklearn/neighbors/lof.py | 150 | 159| 134 | 49520 | 196462 | 
| 98 | 52 examples/covariance/plot_robust_vs_empirical_covariance.py | 1 | 76| 747 | 50267 | 196462 | 
| 99 | 53 benchmarks/bench_plot_fastkmeans.py | 96 | 141| 499 | 50766 | 197688 | 
| 100 | 53 examples/neighbors/plot_lof_novelty_detection.py | 1 | 67| 753 | 51519 | 197688 | 
| 101 | 54 sklearn/neighbors/nearest_centroid.py | 147 | 169| 290 | 51809 | 199376 | 


### Hint

```
Does this have an impact on the clustering? I assume it doesn't. But yes, I
suppose we can mask those out as inf.

> Does this have an impact on the clustering?

AFAIK, no. So maybe it's not an urgent one. (I'll try to debug other urgent issues these days). My point here is that we should ensure the correctness of public attributes (at least consistent with our doc).
I agree

```

## Patch

```diff
diff --git a/sklearn/cluster/optics_.py b/sklearn/cluster/optics_.py
--- a/sklearn/cluster/optics_.py
+++ b/sklearn/cluster/optics_.py
@@ -39,9 +39,8 @@ def optics(X, min_samples=5, max_eps=np.inf, metric='minkowski',
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order. It also does not employ a heap to manage the expansion
-    candiates, but rather uses numpy masked arrays. This can be potentially
-    slower with some parameters (at the benefit from using fast numpy code).
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -199,7 +198,8 @@ class OPTICS(BaseEstimator, ClusterMixin):
     This implementation deviates from the original OPTICS by first performing
     k-nearest-neighborhood searches on all points to identify core sizes, then
     computing only the distances to unprocessed points when constructing the
-    cluster order.
+    cluster order. Note that we do not employ a heap to manage the expansion
+    candidates, so the time complexity will be O(n^2).
 
     Read more in the :ref:`User Guide <optics>`.
 
@@ -430,7 +430,11 @@ def fit(self, X, y=None):
                                 n_jobs=self.n_jobs)
 
         nbrs.fit(X)
+        # Here we first do a kNN query for each point, this differs from
+        # the original OPTICS that only used epsilon range queries.
         self.core_distances_ = self._compute_core_distances_(X, nbrs)
+        # OPTICS puts an upper limit on these, use inf for undefined.
+        self.core_distances_[self.core_distances_ > self.max_eps] = np.inf
         self.ordering_ = self._calculate_optics_order(X, nbrs)
 
         indices_, self.labels_ = _extract_optics(self.ordering_,
@@ -445,7 +449,6 @@ def fit(self, X, y=None):
         return self
 
     # OPTICS helper functions
-
     def _compute_core_distances_(self, X, neighbors, working_memory=None):
         """Compute the k-th nearest neighbor of each sample
 
@@ -485,37 +488,38 @@ def _compute_core_distances_(self, X, neighbors, working_memory=None):
     def _calculate_optics_order(self, X, nbrs):
         # Main OPTICS loop. Not parallelizable. The order that entries are
         # written to the 'ordering_' list is important!
+        # Note that this implementation is O(n^2) theoretically, but
+        # supposedly with very low constant factors.
         processed = np.zeros(X.shape[0], dtype=bool)
         ordering = np.zeros(X.shape[0], dtype=int)
-        ordering_idx = 0
-        for point in range(X.shape[0]):
-            if processed[point]:
-                continue
-            if self.core_distances_[point] <= self.max_eps:
-                while not processed[point]:
-                    processed[point] = True
-                    ordering[ordering_idx] = point
-                    ordering_idx += 1
-                    point = self._set_reach_dist(point, processed, X, nbrs)
-            else:  # For very noisy points
-                ordering[ordering_idx] = point
-                ordering_idx += 1
-                processed[point] = True
+        for ordering_idx in range(X.shape[0]):
+            # Choose next based on smallest reachability distance
+            # (And prefer smaller ids on ties, possibly np.inf!)
+            index = np.where(processed == 0)[0]
+            point = index[np.argmin(self.reachability_[index])]
+
+            processed[point] = True
+            ordering[ordering_idx] = point
+            if self.core_distances_[point] != np.inf:
+                self._set_reach_dist(point, processed, X, nbrs)
         return ordering
 
     def _set_reach_dist(self, point_index, processed, X, nbrs):
         P = X[point_index:point_index + 1]
+        # Assume that radius_neighbors is faster without distances
+        # and we don't need all distances, nevertheless, this means
+        # we may be doing some work twice.
         indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                         return_distance=False)[0]
 
         # Getting indices of neighbors that have not been processed
         unproc = np.compress((~np.take(processed, indices)).ravel(),
                              indices, axis=0)
-        # Keep n_jobs = 1 in the following lines...please
+        # Neighbors of current point are already processed.
         if not unproc.size:
-            # Everything is already processed. Return to main loop
-            return point_index
+            return
 
+        # Only compute distances to unprocessed neighbors:
         if self.metric == 'precomputed':
             dists = X[point_index, unproc]
         else:
@@ -527,12 +531,6 @@ def _set_reach_dist(self, point_index, processed, X, nbrs):
         self.reachability_[unproc[improved]] = rdists[improved]
         self.predecessor_[unproc[improved]] = point_index
 
-        # Choose next based on smallest reachability distance
-        # (And prefer smaller ids on ties).
-        # All unprocessed points qualify, not just new neighbors ("unproc")
-        return (np.ma.array(self.reachability_, mask=processed)
-                .argmin(fill_value=np.inf))
-
     def extract_dbscan(self, eps):
         """Performs DBSCAN extraction for an arbitrary epsilon.
 

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_optics.py b/sklearn/cluster/tests/test_optics.py
--- a/sklearn/cluster/tests/test_optics.py
+++ b/sklearn/cluster/tests/test_optics.py
@@ -22,7 +22,7 @@
 
 
 rng = np.random.RandomState(0)
-n_points_per_cluster = 50
+n_points_per_cluster = 10
 C1 = [-5, -2] + .8 * rng.randn(n_points_per_cluster, 2)
 C2 = [4, -1] + .1 * rng.randn(n_points_per_cluster, 2)
 C3 = [1, -2] + .2 * rng.randn(n_points_per_cluster, 2)
@@ -155,16 +155,10 @@ def test_dbscan_optics_parity(eps, min_samples):
     assert percent_mismatch <= 0.05
 
 
-def test_auto_extract_hier():
-    # Tests auto extraction gets correct # of clusters with varying density
-    clust = OPTICS(min_samples=9).fit(X)
-    assert_equal(len(set(clust.labels_)), 6)
-
-
 # try arbitrary minimum sizes
 @pytest.mark.parametrize('min_cluster_size', range(2, X.shape[0] // 10, 23))
 def test_min_cluster_size(min_cluster_size):
-    redX = X[::10]  # reduce for speed
+    redX = X[::2]  # reduce for speed
     clust = OPTICS(min_samples=9, min_cluster_size=min_cluster_size).fit(redX)
     cluster_sizes = np.bincount(clust.labels_[clust.labels_ != -1])
     if cluster_sizes.size:
@@ -215,171 +209,100 @@ def test_cluster_sigmin_pruning(reach, n_child, members):
     assert_array_equal(members, root.children[0].points)
 
 
+def test_processing_order():
+    # Ensure that we consider all unprocessed points,
+    # not only direct neighbors. when picking the next point.
+    Y = [[0], [10], [-10], [25]]
+    clust = OPTICS(min_samples=3, max_eps=15).fit(Y)
+    assert_array_equal(clust.reachability_, [np.inf, 10, 10, 15])
+    assert_array_equal(clust.core_distances_, [10, 15, np.inf, np.inf])
+    assert_array_equal(clust.ordering_, [0, 1, 2, 3])
+
+
 def test_compare_to_ELKI():
     # Expected values, computed with (future) ELKI 0.7.5 using:
     # java -jar elki.jar cli -dbc.in csv -dbc.filter FixedDBIDsFilter
     #   -algorithm clustering.optics.OPTICSHeap -optics.minpts 5
     # where the FixedDBIDsFilter gives 0-indexed ids.
-    r = [np.inf, 0.7865694338710508, 0.4373157299595305, 0.4121908069391695,
-         0.302907091394212, 0.20815674060999778, 0.20815674060999778,
-         0.15190193459676368, 0.15190193459676368, 0.28229645104833345,
-         0.302907091394212, 0.30507239477026865, 0.30820580778767087,
-         0.3289019667317037, 0.3458462228589966, 0.3458462228589966,
-         0.2931114364132193, 0.2931114364132193, 0.2562790168458507,
-         0.23654635530592025, 0.37903448688824876, 0.3920764620583683,
-         0.4121908069391695, 0.4364542226186831, 0.45523658462146793,
-         0.458757846268185, 0.458757846268185, 0.4752907412198826,
-         0.42350366820623375, 0.42350366820623375, 0.42350366820623375,
-         0.47758738570352993, 0.47758738570352993, 0.4776963110272057,
-         0.5272079288923731, 0.5591861752070968, 0.5592057084987357,
-         0.5609913790596295, 0.5909117211348757, 0.5940470220777727,
-         0.5940470220777727, 0.6861627576116127, 0.687795873252133,
-         0.7538541412862811, 0.7865694338710508, 0.8038180561910464,
-         0.8038180561910464, 0.8242451615289921, 0.8548361202185057,
-         0.8790098789921685, 2.9281214555815764, 1.3256656984284734,
-         0.19590944671099267, 0.1339924636672767, 0.1137384200258616,
-         0.061455005237474075, 0.061455005237474075, 0.061455005237474075,
-         0.045627777293497276, 0.045627777293497276, 0.045627777293497276,
-         0.04900902556283447, 0.061455005237474075, 0.06225461602815799,
-         0.06835750467748272, 0.07882900172724974, 0.07882900172724974,
-         0.07650735397943846, 0.07650735397943846, 0.07650735397943846,
-         0.07650735397943846, 0.07650735397943846, 0.07113275489288699,
-         0.07890196345324527, 0.07052683707634783, 0.07052683707634783,
-         0.07052683707634783, 0.08284027053523288, 0.08725436842020087,
-         0.08725436842020087, 0.09010229261951723, 0.09128578974358925,
-         0.09154172670176584, 0.0968576383038391, 0.12007572768323092,
-         0.12024155806196564, 0.12141990481584404, 0.1339924636672767,
-         0.13694322786307633, 0.14275793459246572, 0.15093125027309579,
-         0.17927454395170142, 0.18151803569400365, 0.1906028449191095,
-         0.1906028449191095, 0.19604486784973194, 0.2096539172540186,
-         0.2096539172540186, 0.21614333983312325, 0.22036454909290296,
-         0.23610322103910933, 0.26028003932256766, 0.2607126030060721,
-         0.2891824876072483, 0.3258089271514364, 0.35968687619960743,
-         0.4512973330510512, 0.4746141313843085, 0.5958585488429471,
-         0.6468718886525733, 0.6878453052524358, 0.6911582799500199,
-         0.7172169499815705, 0.7209874999572031, 0.6326884657912096,
-         0.5755681293026617, 0.5755681293026617, 0.5755681293026617,
-         0.6015042225447333, 0.6756244556376542, 0.4722384908959966,
-         0.08775739179493615, 0.06665303472021758, 0.056308477780164796,
-         0.056308477780164796, 0.05507767260835565, 0.05368146914586802,
-         0.05163427719303039, 0.05163427719303039, 0.05163427719303039,
-         0.04918757627098621, 0.04918757627098621, 0.05368146914586802,
-         0.05473720349424546, 0.05473720349424546, 0.048442038421760626,
-         0.048442038421760626, 0.04598840269934622, 0.03984301937835033,
-         0.04598840269934622, 0.04598840269934622, 0.04303884892957088,
-         0.04303884892957088, 0.04303884892957088, 0.0431802780806032,
-         0.0520412490141781, 0.056308477780164796, 0.05080724020124642,
-         0.05080724020124642, 0.05080724020124642, 0.06385565101399236,
-         0.05840878369200427, 0.0474472391259039, 0.0474472391259039,
-         0.04232512684465669, 0.04232512684465669, 0.04232512684465669,
-         0.0474472391259039, 0.051802632822946656, 0.051802632822946656,
-         0.05316405104684577, 0.05316405104684577, 0.05840878369200427,
-         0.06385565101399236, 0.08025248922898705, 0.08775739179493615,
-         0.08993337040710143, 0.08993337040710143, 0.08993337040710143,
-         0.08993337040710143, 0.297457175321605, 0.29763608186278934,
-         0.3415255849656254, 0.34713336941105105, 0.44108940848708167,
-         0.35942962652965604, 0.35942962652965604, 0.33609522256535296,
-         0.5008111387107295, 0.5333587622018111, 0.6223243743872802,
-         0.6793840035409552, 0.7445032492109848, 0.7445032492109848,
-         0.6556432627279256, 0.6556432627279256, 0.6556432627279256,
-         0.8196566935960162, 0.8724089149982351, 0.9352758042365477,
-         0.9352758042365477, 1.0581847953137133, 1.0684332509194163,
-         1.0887817699873303, 1.2552604310322708, 1.3993856001769436,
-         1.4869615658197606, 1.6588098267326852, 1.679969559453028,
-         1.679969559453028, 1.6860509219163458, 1.6860509219163458,
-         1.1465697826627317, 0.992866533434785, 0.7691908270707519,
-         0.578131499171622, 0.578131499171622, 0.578131499171622,
-         0.5754243919945694, 0.8416199360035114, 0.8722493727270406,
-         0.9156549976203665, 0.9156549976203665, 0.7472322844356064,
-         0.715219324518981, 0.715219324518981, 0.715219324518981,
-         0.7472322844356064, 0.820988298336316, 0.908958489674247,
-         0.9234036745782839, 0.9519521817942455, 0.992866533434785,
-         0.992866533434785, 0.9995692674695029, 1.0727415198904493,
-         1.1395519941203158, 1.1395519941203158, 1.1741737271442092,
-         1.212860115632712, 0.8724097897372123, 0.8724097897372123,
-         0.8724097897372123, 1.2439272570611581, 1.2439272570611581,
-         1.3524538390109015, 1.3524538390109015, 1.2982303284415664,
-         1.3610655849680207, 1.3802783392089437, 1.3802783392089437,
-         1.4540636953090629, 1.5879329500533819, 1.5909193228826986,
-         1.72931779186001, 1.9619075944592093, 2.1994355761906257,
-         2.2508672067362165, 2.274436122235927, 2.417635732260135,
-         3.014235905390584, 0.30616929141177107, 0.16449675872754976,
-         0.09071681523805683, 0.09071681523805683, 0.09071681523805683,
-         0.08727060912039632, 0.09151721189581336, 0.12277953408786725,
-         0.14285575406641507, 0.16449675872754976, 0.16321992344119793,
-         0.1330971730344373, 0.11429891993167259, 0.11429891993167259,
-         0.11429891993167259, 0.11429891993167259, 0.11429891993167259,
-         0.0945498340011516, 0.11410457435712089, 0.1196414019798306,
-         0.12925682285016715, 0.12925682285016715, 0.12925682285016715,
-         0.12864887158869853, 0.12864887158869853, 0.12864887158869853,
-         0.13369634918690246, 0.14330826543275352, 0.14877705862323184,
-         0.15203263952428328, 0.15696350160889708, 0.1585326700393211,
-         0.1585326700393211, 0.16034306786654595, 0.16034306786654595,
-         0.15053328296567992, 0.16396729418886688, 0.16763548009617293,
-         0.1732029325454474, 0.21163390061029352, 0.21497664171864372,
-         0.22125889949299, 0.240251070192081, 0.240251070192081,
-         0.2413620965310808, 0.26319419022234064, 0.26319419022234064,
-         0.27989712380504483, 0.2909782800714374]
-    o = [0, 3, 6, 7, 15, 4, 27, 28, 49, 17, 35, 47, 46, 39, 13, 19,
-         22, 29, 30, 38, 34, 32, 43, 8, 25, 9, 37, 23, 33, 40, 44, 11, 36, 5,
-         45, 48, 41, 26, 24, 20, 31, 2, 16, 10, 18, 14, 42, 12, 1, 21, 234,
-         132, 112, 115, 107, 110, 120, 114, 100, 131, 137, 145, 130, 121, 134,
-         116, 149, 108, 111, 113, 142, 148, 119, 104, 126, 133, 138, 127, 101,
-         105, 103, 106, 125, 140, 123, 147, 144, 129, 141, 117, 143, 136, 128,
-         122, 124, 102, 109, 249, 146, 118, 135, 245, 139, 224, 241, 217, 202,
-         248, 233, 214, 236, 211, 206, 231, 212, 221, 229, 244, 208, 226, 83,
-         76, 53, 77, 88, 62, 66, 65, 89, 93, 79, 95, 74, 70, 82, 51, 73, 87,
-         67, 94, 56, 52, 63, 80, 75, 57, 96, 60, 69, 90, 86, 58, 68, 81, 64,
-         84, 85, 97, 59, 98, 61, 71, 78, 92, 50, 91, 55, 54, 72, 99, 210, 201,
-         216, 239, 203, 218, 219, 222, 240, 294, 243, 246, 204, 220, 200, 215,
-         230, 225, 205, 207, 237, 223, 235, 209, 228, 238, 227, 285, 232, 256,
-         281, 270, 260, 252, 272, 268, 292, 298, 269, 275, 257, 250, 284, 283,
-         286, 295, 297, 293, 289, 258, 299, 282, 262, 296, 287, 267, 255, 263,
-         288, 276, 251, 266, 274, 271, 277, 261, 279, 290, 253, 254, 291, 259,
-         280, 278, 273, 247, 265, 242, 264, 213, 199, 174, 154, 152, 180, 186,
-         195, 170, 181, 176, 187, 173, 157, 159, 158, 172, 182, 183, 151, 197,
-         177, 160, 156, 171, 175, 184, 193, 161, 179, 196, 185, 192, 165, 166,
-         164, 189, 155, 162, 188, 153, 178, 169, 194, 150, 163, 198, 190, 191,
-         168, 167]
-    p = [-1, 0, 3, 6, 7, 15, 15, 27, 27, 4, 7, 49, 47, 4, 39, 39,
-         19, 19, 29, 30, 30, 13, 6, 43, 34, 32, 32, 25, 23, 23, 23, 3, 11, 46,
-         46, 45, 9, 38, 33, 26, 26, 8, 20, 33, 0, 18, 18, 2, 18, 44, 0, 234,
-         132, 112, 115, 107, 107, 120, 114, 114, 114, 114, 107, 100, 100, 134,
-         134, 149, 149, 108, 108, 108, 148, 113, 104, 104, 104, 142, 127, 127,
-         126, 138, 126, 148, 127, 148, 127, 112, 147, 116, 117, 101, 145, 128,
-         128, 122, 136, 136, 249, 102, 102, 118, 143, 146, 245, 123, 139, 241,
-         241, 217, 248, 202, 248, 224, 231, 212, 212, 212, 229, 229, 226, 83,
-         76, 53, 53, 88, 62, 66, 66, 66, 93, 93, 79, 93, 70, 82, 82, 73, 87,
-         73, 94, 56, 56, 56, 63, 67, 53, 96, 96, 96, 69, 86, 58, 58, 81, 81,
-         81, 58, 64, 64, 59, 59, 86, 69, 78, 83, 84, 55, 55, 55, 72, 50, 201,
-         210, 216, 203, 203, 219, 54, 240, 239, 240, 236, 236, 220, 220, 220,
-         217, 139, 243, 243, 204, 211, 246, 215, 223, 294, 209, 227, 227, 209,
-         281, 270, 260, 252, 272, 272, 272, 298, 269, 298, 275, 275, 284, 283,
-         283, 283, 284, 286, 283, 298, 299, 260, 260, 250, 299, 258, 258, 296,
-         250, 276, 276, 276, 289, 289, 267, 267, 279, 261, 277, 277, 258, 266,
-         290, 209, 207, 290, 228, 278, 228, 290, 199, 174, 154, 154, 154, 186,
-         186, 180, 170, 174, 187, 173, 157, 159, 157, 157, 159, 183, 183, 172,
-         197, 160, 160, 171, 171, 171, 151, 173, 184, 151, 196, 185, 185, 179,
-         179, 189, 177, 165, 175, 162, 164, 181, 169, 169, 181, 178, 178, 178,
-         168]
+    r1 = [np.inf, 1.0574896366427478, 0.7587934993548423, 0.7290174038973836,
+          0.7290174038973836, 0.7290174038973836, 0.6861627576116127,
+          0.7587934993548423, 0.9280118450166668, 1.1748022534146194,
+          3.3355455741292257, 0.49618389254482587, 0.2552805046961355,
+          0.2552805046961355, 0.24944622248445714, 0.24944622248445714,
+          0.24944622248445714, 0.2552805046961355, 0.2552805046961355,
+          0.3086779122185853, 4.163024452756142, 1.623152630340929,
+          0.45315840475822655, 0.25468325192031926, 0.2254004358159971,
+          0.18765711877083036, 0.1821471333893275, 0.1821471333893275,
+          0.18765711877083036, 0.18765711877083036, 0.2240202988740153,
+          1.154337614548715, 1.342604473837069, 1.323308536402633,
+          0.8607514948648837, 0.27219111215810565, 0.13260875220533205,
+          0.13260875220533205, 0.09890587675958984, 0.09890587675958984,
+          0.13548790801634494, 0.1575483940837384, 0.17515137170530226,
+          0.17575920159442388, 0.27219111215810565, 0.6101447895405373,
+          1.3189208094864302, 1.323308536402633, 2.2509184159764577,
+          2.4517810628594527, 3.675977064404973, 3.8264795626020365,
+          2.9130735341510614, 2.9130735341510614, 2.9130735341510614,
+          2.9130735341510614, 2.8459300127258036, 2.8459300127258036,
+          2.8459300127258036, 3.0321982337972537]
+    o1 = [0, 3, 6, 4, 7, 8, 2, 9, 5, 1, 31, 30, 32, 34, 33, 38, 39, 35, 37, 36,
+          44, 21, 23, 24, 22, 25, 27, 29, 26, 28, 20, 40, 45, 46, 10, 15, 11,
+          13, 17, 19, 18, 12, 16, 14, 47, 49, 43, 48, 42, 41, 53, 57, 51, 52,
+          56, 59, 54, 55, 58, 50]
+    p1 = [-1, 0, 3, 6, 6, 6, 8, 3, 7, 5, 1, 31, 30, 30, 34, 34, 34, 32, 32, 37,
+          36, 44, 21, 23, 24, 22, 25, 25, 22, 22, 22, 21, 40, 45, 46, 10, 15,
+          15, 13, 13, 15, 11, 19, 15, 10, 47, 12, 45, 14, 43, 42, 53, 57, 57,
+          57, 57, 59, 59, 59, 58]
 
     # Tests against known extraction array
     # Does NOT work with metric='euclidean', because sklearn euclidean has
     # worse numeric precision. 'minkowski' is slower but more accurate.
-    clust = OPTICS(min_samples=5).fit(X)
+    clust1 = OPTICS(min_samples=5).fit(X)
 
-    assert_array_equal(clust.ordering_, np.array(o))
-    assert_array_equal(clust.predecessor_[clust.ordering_], np.array(p))
-    assert_allclose(clust.reachability_[clust.ordering_], np.array(r))
+    assert_array_equal(clust1.ordering_, np.array(o1))
+    assert_array_equal(clust1.predecessor_[clust1.ordering_], np.array(p1))
+    assert_allclose(clust1.reachability_[clust1.ordering_], np.array(r1))
     # ELKI currently does not print the core distances (which are not used much
     # in literature, but we can at least ensure to have this consistency:
-    for i in clust.ordering_[1:]:
-        assert (clust.reachability_[i] >=
-                clust.core_distances_[clust.predecessor_[i]])
+    for i in clust1.ordering_[1:]:
+        assert (clust1.reachability_[i] >=
+                clust1.core_distances_[clust1.predecessor_[i]])
+
+    # Expected values, computed with (future) ELKI 0.7.5 using
+    r2 = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf, np.inf, 0.27219111215810565, 0.13260875220533205,
+          0.13260875220533205, 0.09890587675958984, 0.09890587675958984,
+          0.13548790801634494, 0.1575483940837384, 0.17515137170530226,
+          0.17575920159442388, 0.27219111215810565, 0.4928068613197889,
+          np.inf, 0.2666183922512113, 0.18765711877083036, 0.1821471333893275,
+          0.1821471333893275, 0.1821471333893275, 0.18715928772277457,
+          0.18765711877083036, 0.18765711877083036, 0.25468325192031926,
+          np.inf, 0.2552805046961355, 0.2552805046961355, 0.24944622248445714,
+          0.24944622248445714, 0.24944622248445714, 0.2552805046961355,
+          0.2552805046961355, 0.3086779122185853, 0.34466409325984865,
+          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
+          np.inf, np.inf]
+    o2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 11, 13, 17, 19, 18, 12, 16, 14,
+          47, 46, 20, 22, 25, 23, 27, 29, 24, 26, 28, 21, 30, 32, 34, 33, 38,
+          39, 35, 37, 36, 31, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53,
+          54, 55, 56, 57, 58, 59]
+    p2 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 15, 15, 13, 13, 15,
+          11, 19, 15, 10, 47, -1, 20, 22, 25, 25, 25, 25, 22, 22, 23, -1, 30,
+          30, 34, 34, 34, 32, 32, 37, 38, -1, -1, -1, -1, -1, -1, -1, -1, -1,
+          -1, -1, -1, -1, -1, -1, -1, -1, -1]
+    clust2 = OPTICS(min_samples=5, max_eps=0.5).fit(X)
+
+    assert_array_equal(clust2.ordering_, np.array(o2))
+    assert_array_equal(clust2.predecessor_[clust2.ordering_], np.array(p2))
+    assert_allclose(clust2.reachability_[clust2.ordering_], np.array(r2))
+
+    index = np.where(clust1.core_distances_ <= 0.5)[0]
+    assert_allclose(clust1.core_distances_[index],
+                    clust2.core_distances_[index])
 
 
 def test_precomputed_dists():
-    redX = X[::10]
+    redX = X[::2]
     dists = pairwise_distances(redX, metric='euclidean')
     clust1 = OPTICS(min_samples=10, algorithm='brute',
                     metric='precomputed').fit(dists)
@@ -388,13 +311,3 @@ def test_precomputed_dists():
 
     assert_allclose(clust1.reachability_, clust2.reachability_)
     assert_array_equal(clust1.labels_, clust2.labels_)
-
-
-def test_processing_order():
-    """Early dev version of OPTICS would not consider all unprocessed points,
-    but only direct neighbors. This tests against this mistake."""
-    Y = [[0], [10], [-10], [25]]
-    clust = OPTICS(min_samples=3, max_eps=15).fit(Y)
-    assert_array_equal(clust.reachability_, [np.inf, 10, 10, 15])
-    assert_array_equal(clust.core_distances_, [10, 15, 20, 25])
-    assert_array_equal(clust.ordering_, [0, 1, 2, 3])

```


## Code snippets

### 1 - sklearn/cluster/optics_.py:

Start line: 191, End line: 354

```python
class OPTICS(BaseEstimator, ClusterMixin):
    """Estimate clustering structure from vector array

    OPTICS: Ordering Points To Identify the Clustering Structure
    Closely related to DBSCAN, finds core sample of high density and expands
    clusters from them. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large point datasets than
    the current sklearn implementation of DBSCAN.

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order.

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    min_samples : int (default=5)
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    max_eps : float, optional (default=np.inf)
        The maximum distance between two samples for them to be considered
        as in the same neighborhood. Default value of "np.inf" will identify
        clusters across all scales; reducing `max_eps` will result in
        shorter run times.

    metric : string or callable, optional (default='minkowski')
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        :class:`sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function.

    maxima_ratio : float, optional (default=.75)
        The maximum ratio we allow of average height of clusters on the
        right and left to the local maxima in question. The higher the
        ratio, the more generous the algorithm is to preserving local
        minima, and the more cuts the resulting tree will have.

    rejection_ratio : float, optional (default=.7)
        Adjusts the fitness of the clustering. When the maxima_ratio is
        exceeded, determine which of the clusters to the left and right to
        reject based on rejection_ratio. Higher values will result in points
        being more readily classified as noise; conversely, lower values will
        result in more points being clustered.

    similarity_threshold : float, optional (default=.4)
        Used to check if nodes can be moved up one level, that is, if the
        new cluster created is too "similar" to its parent, given the
        similarity threshold. Similarity can be determined by 1) the size
        of the new cluster relative to the size of the parent node or
        2) the average of the reachability values of the new cluster
        relative to the average of the reachability values of the parent
        node. A lower value for the similarity threshold means less levels
        in the tree.

    significant_min : float, optional (default=.003)
        Sets a lower threshold on how small a significant maxima can be.

    min_cluster_size : int > 1 or float between 0 and 1 (default=0.005)
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded
        to be at least 2).

    min_maxima_ratio : float, optional (default=.001)
        Used to determine neighborhood size for minimum cluster membership.
        Each local maxima should be a largest value in a neighborhood
        of the `size min_maxima_ratio * len(X)` from left and right.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree` (default)
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

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    core_sample_indices_ : array, shape (n_core_samples,)
        Indices of core samples.

    labels_ : array, shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    reachability_ : array, shape (n_samples,)
        Reachability distances per sample, indexed by object order. Use
        ``clust.reachability_[clust.ordering_]`` to access in cluster order.

    ordering_ : array, shape (n_samples,)
        The cluster ordered list of sample indices.

    core_distances_ : array, shape (n_samples,)
        Distance at which each sample becomes a core point, indexed by object
        order. Points which will never be core have a distance of inf. Use
        ``clust.core_distances_[clust.ordering_]`` to access in cluster order.

    predecessor_ : array, shape (n_samples,)
        Point that a sample was reached from, indexed by object order.
        Seed points have a predecessor of -1.

    See also
    --------

    DBSCAN
        A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander.
    "OPTICS: ordering points to identify the clustering structure." ACM SIGMOD
    Record 28, no. 2 (1999): 49-60.

    Schubert, Erich, Michael Gertz.
    "Improving the Cluster Structure Extracted from OPTICS Plots." Proc. of
    the Conference "Lernen, Wissen, Daten, Analysen" (LWDA) (2018): 318-329.
    """
```
### 2 - sklearn/cluster/optics_.py:

Start line: 506, End line: 534

```python
class OPTICS(BaseEstimator, ClusterMixin):

    def _set_reach_dist(self, point_index, processed, X, nbrs):
        P = X[point_index:point_index + 1]
        indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                        return_distance=False)[0]

        # Getting indices of neighbors that have not been processed
        unproc = np.compress((~np.take(processed, indices)).ravel(),
                             indices, axis=0)
        # Keep n_jobs = 1 in the following lines...please
        if not unproc.size:
            # Everything is already processed. Return to main loop
            return point_index

        if self.metric == 'precomputed':
            dists = X[point_index, unproc]
        else:
            dists = pairwise_distances(P, np.take(X, unproc, axis=0),
                                       self.metric, n_jobs=None).ravel()

        rdists = np.maximum(dists, self.core_distances_[point_index])
        improved = np.where(rdists < np.take(self.reachability_, unproc))
        self.reachability_[unproc[improved]] = rdists[improved]
        self.predecessor_[unproc[improved]] = point_index

        # Choose next based on smallest reachability distance
        # (And prefer smaller ids on ties).
        # All unprocessed points qualify, not just new neighbors ("unproc")
        return (np.ma.array(self.reachability_, mask=processed)
                .argmin(fill_value=np.inf))
```
### 3 - sklearn/cluster/optics_.py:

Start line: 378, End line: 445

```python
class OPTICS(BaseEstimator, ClusterMixin):

    def fit(self, X, y=None):
        """Perform OPTICS clustering

        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using `max_eps` distance specified at
        OPTICS object instantiation.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data.

        y : ignored

        Returns
        -------
        self : instance of OPTICS
            The instance.
        """
        X = check_array(X, dtype=np.float)

        n_samples = len(X)

        if self.min_samples > n_samples:
            raise ValueError("Number of training samples (n_samples=%d) must "
                             "be greater than min_samples (min_samples=%d) "
                             "used for clustering." %
                             (n_samples, self.min_samples))

        if self.min_cluster_size <= 0 or (self.min_cluster_size !=
                                          int(self.min_cluster_size)
                                          and self.min_cluster_size > 1):
            raise ValueError('min_cluster_size must be a positive integer or '
                             'a float between 0 and 1. Got %r' %
                             self.min_cluster_size)
        elif self.min_cluster_size > n_samples:
            raise ValueError('min_cluster_size must be no greater than the '
                             'number of samples (%d). Got %d' %
                             (n_samples, self.min_cluster_size))

        # Start all points as 'unprocessed' ##
        self.reachability_ = np.empty(n_samples)
        self.reachability_.fill(np.inf)
        self.predecessor_ = np.empty(n_samples, dtype=int)
        self.predecessor_.fill(-1)
        # Start all points as noise ##
        self.labels_ = np.full(n_samples, -1, dtype=int)

        nbrs = NearestNeighbors(n_neighbors=self.min_samples,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size, metric=self.metric,
                                metric_params=self.metric_params, p=self.p,
                                n_jobs=self.n_jobs)

        nbrs.fit(X)
        self.core_distances_ = self._compute_core_distances_(X, nbrs)
        self.ordering_ = self._calculate_optics_order(X, nbrs)

        indices_, self.labels_ = _extract_optics(self.ordering_,
                                                 self.reachability_,
                                                 self.maxima_ratio,
                                                 self.rejection_ratio,
                                                 self.similarity_threshold,
                                                 self.significant_min,
                                                 self.min_cluster_size,
                                                 self.min_maxima_ratio)
        self.core_sample_indices_ = indices_
        return self
```
### 4 - sklearn/cluster/optics_.py:

Start line: 31, End line: 573

```python
def optics(X, min_samples=5, max_eps=np.inf, metric='minkowski',
           p=2, metric_params=None, maxima_ratio=.75,
           rejection_ratio=.7, similarity_threshold=0.4,
           significant_min=.003, min_cluster_size=.005,
           min_maxima_ratio=0.001, algorithm='ball_tree',
           leaf_size=30, n_jobs=None):
    """Perform OPTICS clustering from vector array

    OPTICS: Ordering Points To Identify the Clustering Structure
    Closely related to DBSCAN, finds core sample of high density and expands
    clusters from them. Unlike DBSCAN, keeps cluster hierarchy for a variable
    neighborhood radius. Better suited for usage on large point datasets than
    the current sklearn implementation of DBSCAN.

    This implementation deviates from the original OPTICS by first performing
    k-nearest-neighborhood searches on all points to identify core sizes, then
    computing only the distances to unprocessed points when constructing the
    cluster order. It also does not employ a heap to manage the expansion
    candiates, but rather uses numpy masked arrays. This can be potentially
    slower with some parameters (at the benefit from using fast numpy code).

    Read more in the :ref:`User Guide <optics>`.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data.

    min_samples : int (default=5)
        The number of samples in a neighborhood for a point to be considered
        as a core point.

    max_eps : float, optional (default=np.inf)
        The maximum distance between two samples for them to be considered
        as in the same neighborhood. Default value of "np.inf" will identify
        clusters across all scales; reducing `max_eps` will result in
        shorter run times.

    metric : string or callable, optional (default='minkowski')
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        :class:`sklearn.metrics.pairwise_distances`. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function.

    maxima_ratio : float, optional (default=.75)
        The maximum ratio we allow of average height of clusters on the
        right and left to the local maxima in question. The higher the
        ratio, the more generous the algorithm is to preserving local
        minima, and the more cuts the resulting tree will have.

    rejection_ratio : float, optional (default=.7)
        Adjusts the fitness of the clustering. When the maxima_ratio is
        exceeded, determine which of the clusters to the left and right to
        reject based on rejection_ratio. Higher values will result in points
        being more readily classified as noise; conversely, lower values will
        result in more points being clustered.

    similarity_threshold : float, optional (default=.4)
        Used to check if nodes can be moved up one level, that is, if the
        new cluster created is too "similar" to its parent, given the
        similarity threshold. Similarity can be determined by 1) the size
        of the new cluster relative to the size of the parent node or
        2) the average of the reachability values of the new cluster
        relative to the average of the reachability values of the parent
        node. A lower value for the similarity threshold means less levels
        in the tree.

    significant_min : float, optional (default=.003)
        Sets a lower threshold on how small a significant maxima can be.

    min_cluster_size : int > 1 or float between 0 and 1 (default=0.005)
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded
        to be at least 2).

    min_maxima_ratio : float, optional (default=.001)
        Used to determine neighborhood size for minimum cluster membership.
        Each local maxima should be a largest value in a neighborhood
        of the `size min_maxima_ratio * len(X)` from left and right.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree` (default)
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

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Returns
    -------
    core_sample_indices_ : array, shape (n_core_samples,)
        The indices of the core samples.

    labels_ : array, shape (n_samples,)
        The estimated labels.

    See also
    --------
    OPTICS
        An estimator interface for this clustering algorithm.
    dbscan
        A similar clustering for a specified neighborhood radius (eps).
        Our implementation is optimized for runtime.

    References
    ----------
    Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander.
    "OPTICS: ordering points to identify the clustering structure." ACM SIGMOD
    Record 28, no. 2 (1999): 49-60.
    """

    clust = OPTICS(min_samples, max_eps, metric, p, metric_params,
                   maxima_ratio, rejection_ratio,
                   similarity_threshold, significant_min,
                   min_cluster_size, min_maxima_ratio,
                   algorithm, leaf_size, n_jobs)
    clust.fit(X)
    return clust.core_sample_indices_, clust.labels_


class OPTICS(BaseEstimator, ClusterMixin):
```
### 5 - examples/cluster/plot_optics.py:

Start line: 1, End line: 80

```python
"""
===================================
Demo of OPTICS clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.
This example uses data that is generated so that the clusters have
different densities.

The clustering is first used in its automatic settings, which is the
:class:`sklearn.cluster.OPTICS` algorithm, and then setting specific
thresholds on the reachability, which corresponds to DBSCAN.

We can see that the different clusters of OPTICS can be recovered with
different choices of thresholds in DBSCAN.

"""


from sklearn.cluster import OPTICS
import matplotlib.gridspec as gridspec


import numpy as np

import matplotlib.pyplot as plt

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

clust = OPTICS(min_samples=9, rejection_ratio=0.5)

# Run the fit
clust.fit(X)

_, labels_025 = clust.extract_dbscan(0.25)
_, labels_075 = clust.extract_dbscan(0.75)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
color = ['g.', 'r.', 'b.', 'y.', 'c.']
for k, c in zip(range(0, 5), color):
    Xk = space[labels == k]
    Rk = reachability[labels == k]
    ax1.plot(Xk, Rk, c, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 0.75, dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.25, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
color = ['g.', 'r.', 'b.', 'y.', 'c.']
for k, c in zip(range(0, 5), color):
    Xk = X[clust.labels_ == k]
    ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
```
### 6 - sklearn/cluster/optics_.py:

Start line: 485, End line: 504

```python
class OPTICS(BaseEstimator, ClusterMixin):

    def _calculate_optics_order(self, X, nbrs):
        # Main OPTICS loop. Not parallelizable. The order that entries are
        # written to the 'ordering_' list is important!
        processed = np.zeros(X.shape[0], dtype=bool)
        ordering = np.zeros(X.shape[0], dtype=int)
        ordering_idx = 0
        for point in range(X.shape[0]):
            if processed[point]:
                continue
            if self.core_distances_[point] <= self.max_eps:
                while not processed[point]:
                    processed[point] = True
                    ordering[ordering_idx] = point
                    ordering_idx += 1
                    point = self._set_reach_dist(point, processed, X, nbrs)
            else:  # For very noisy points
                ordering[ordering_idx] = point
                ordering_idx += 1
                processed[point] = True
        return ordering
```
### 7 - sklearn/cluster/optics_.py:

Start line: 447, End line: 483

```python
class OPTICS(BaseEstimator, ClusterMixin):

    # OPTICS helper functions

    def _compute_core_distances_(self, X, neighbors, working_memory=None):
        """Compute the k-th nearest neighbor of each sample

        Equivalent to neighbors.kneighbors(X, self.min_samples)[0][:, -1]
        but with more memory efficiency.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data.
        neighbors : NearestNeighbors instance
            The fitted nearest neighbors estimator.
        working_memory : int, optional
            The sought maximum memory for temporary distance matrix chunks.
            When None (default), the value of
            ``sklearn.get_config()['working_memory']`` is used.

        Returns
        -------
        core_distances : array, shape (n_samples,)
            Distance at which each sample becomes a core point.
            Points which will never be core have a distance of inf.
        """
        n_samples = len(X)
        core_distances = np.empty(n_samples)
        core_distances.fill(np.nan)

        chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self.min_samples,
                                        max_n_rows=n_samples,
                                        working_memory=working_memory)
        slices = gen_batches(n_samples, chunk_n_rows)
        for sl in slices:
            core_distances[sl] = neighbors.kneighbors(
                X[sl], self.min_samples)[0][:, -1]
        return core_distances
```
### 8 - sklearn/cluster/optics_.py:

Start line: 1, End line: 187

```python
# -*- coding: utf-8 -*-

from __future__ import division
import warnings
import numpy as np

from ..utils import check_array
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.validation import check_is_fitted
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator, ClusterMixin
from ..metrics import pairwise_distances


def optics(X, min_samples=5, max_eps=np.inf, metric='minkowski',
           p=2, metric_params=None, maxima_ratio=.75,
           rejection_ratio=.7, similarity_threshold=0.4,
           significant_min=.003, min_cluster_size=.005,
           min_maxima_ratio=0.001, algorithm='ball_tree',
           leaf_size=30, n_jobs=None):
    # ... other code
```
### 9 - sklearn/cluster/optics_.py:

Start line: 611, End line: 693

```python
def _extract_optics(ordering, reachability, maxima_ratio=.75,
                    rejection_ratio=.7, similarity_threshold=0.4,
                    significant_min=.003, min_cluster_size=.005,
                    min_maxima_ratio=0.001):
    """Performs automatic cluster extraction for variable density data.

    Parameters
    ----------
    ordering : array, shape (n_samples,)
        OPTICS ordered point indices (`ordering_`)

    reachability : array, shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`)

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

    min_cluster_size : int > 1 or float between 0 and 1
        Minimum number of samples in an OPTICS cluster, expressed as an
        absolute number or a fraction of the number of samples (rounded
        to be at least 2).

    min_maxima_ratio : float, optional
        Used to determine neighborhood size for minimum cluster membership.

    Returns
    -------
    core_sample_indices_ : array, shape (n_core_samples,)
        The indices of the core samples.

    labels_ : array, shape (n_samples,)
        The estimated labels.
    """

    # Extraction wrapper
    # according to Ankerst M. et.al. 1999 (p. 5), for a small enough
    # generative distance epsilong, there should be more than one INF.
    if np.all(np.isinf(reachability)):
        raise ValueError("All reachability values are inf. Set a larger"
                         " max_eps.")
    normalization_factor = np.max(reachability[reachability < np.inf])
    reachability = reachability / normalization_factor
    reachability_plot = reachability[ordering].tolist()
    root_node = _automatic_cluster(reachability_plot, ordering,
                                   maxima_ratio, rejection_ratio,
                                   similarity_threshold, significant_min,
                                   min_cluster_size, min_maxima_ratio)
    leaves = _get_leaves(root_node, [])
    # Start cluster id's at 0
    clustid = 0
    n_samples = len(reachability)
    is_core = np.zeros(n_samples, dtype=bool)
    labels = np.full(n_samples, -1, dtype=int)
    # Start all points as non-core noise
    for leaf in leaves:
        index = ordering[leaf.start:leaf.end]
        labels[index] = clustid
        is_core[index] = 1
        clustid += 1
    return np.arange(n_samples)[is_core], labels
```
### 10 - sklearn/cluster/optics_.py:

Start line: 356, End line: 376

```python
class OPTICS(BaseEstimator, ClusterMixin):

    def __init__(self, min_samples=5, max_eps=np.inf, metric='minkowski',
                 p=2, metric_params=None, maxima_ratio=.75,
                 rejection_ratio=.7, similarity_threshold=0.4,
                 significant_min=.003, min_cluster_size=.005,
                 min_maxima_ratio=0.001, algorithm='ball_tree',
                 leaf_size=30, n_jobs=None):

        self.max_eps = max_eps
        self.min_samples = min_samples
        self.maxima_ratio = maxima_ratio
        self.rejection_ratio = rejection_ratio
        self.similarity_threshold = similarity_threshold
        self.significant_min = significant_min
        self.min_cluster_size = min_cluster_size
        self.min_maxima_ratio = min_maxima_ratio
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
```
### 11 - sklearn/cluster/optics_.py:

Start line: 576, End line: 608

```python
def _extract_dbscan(ordering, core_distances, reachability, eps):
    """Performs DBSCAN extraction for an arbitrary epsilon (`eps`).

    Parameters
    ----------
    ordering : array, shape (n_samples,)
        OPTICS ordered point indices (`ordering_`)
    core_distances : array, shape (n_samples,)
        Distances at which points become core (`core_distances_`)
    reachability : array, shape (n_samples,)
        Reachability distances calculated by OPTICS (`reachability_`)
    eps : float or int
        DBSCAN `eps` parameter

    Returns
    -------
    core_sample_indices_ : array, shape (n_core_samples,)
        The indices of the core samples.

    labels_ : array, shape (n_samples,)
        The estimated labels.
    """

    n_samples = len(core_distances)
    is_core = np.zeros(n_samples, dtype=bool)
    labels = np.zeros(n_samples, dtype=int)

    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    is_core[near_core] = True
    return np.arange(n_samples)[is_core], labels
```
### 12 - sklearn/cluster/optics_.py:

Start line: 536, End line: 573

```python
class OPTICS(BaseEstimator, ClusterMixin):

    def extract_dbscan(self, eps):
        """Performs DBSCAN extraction for an arbitrary epsilon.

        Extraction runs in linear time. Note that if the `max_eps` OPTICS
        parameter was set to < inf for extracting reachability and ordering
        arrays, DBSCAN extractions will be unstable for `eps` values close to
        `max_eps`. Setting `eps` < (`max_eps` / 5.0) will guarantee
        extraction parity with DBSCAN.

        Parameters
        ----------
        eps : float or int, required
            DBSCAN `eps` parameter. Must be set to < `max_eps`. Equivalence
            with DBSCAN algorithm is achieved if `eps` is < (`max_eps` / 5)

        Returns
        -------
        core_sample_indices_ : array, shape (n_core_samples,)
            The indices of the core samples.

        labels_ : array, shape (n_samples,)
            The estimated labels.
        """
        check_is_fitted(self, 'reachability_')

        if eps > self.max_eps:
            raise ValueError('Specify an epsilon smaller than %s. Got %s.'
                             % (self.max_eps, eps))

        if eps * 5.0 > (self.max_eps * 1.05):
            warnings.warn(
                "Warning, max_eps (%s) is close to eps (%s): "
                "Output may be unstable." % (self.max_eps, eps),
                RuntimeWarning, stacklevel=2)
        # Stability warning is documented in _extract_dbscan method...

        return _extract_dbscan(self.ordering_, self.core_distances_,
                               self.reachability_, eps)
```
### 14 - sklearn/cluster/optics_.py:

Start line: 730, End line: 745

```python
class _TreeNode(object):
    # automatic cluster helper classes and functions
    def __init__(self, points, start, end, parent_node):
        self.points = points
        self.start = start
        self.end = end
        self.parent_node = parent_node
        self.children = []
        self.split_point = -1


def _is_local_maxima(index, reachability_plot, neighborhood_size):
    right_idx = slice(index + 1, index + neighborhood_size + 1)
    left_idx = slice(max(1, index - neighborhood_size - 1), index)
    return (np.all(reachability_plot[index] >= reachability_plot[left_idx]) and
            np.all(reachability_plot[index] >= reachability_plot[right_idx]))
```
### 19 - sklearn/cluster/optics_.py:

Start line: 748, End line: 763

```python
def _find_local_maxima(reachability_plot, neighborhood_size):
    local_maxima_points = {}
    # 1st and last points on Reachability Plot are not taken
    # as local maxima points
    for i in range(1, len(reachability_plot) - 1):
        # if the point is a local maxima on the reachability plot with
        # regard to neighborhood_size, insert it into priority queue and
        # maxima list
        if (reachability_plot[i] > reachability_plot[i - 1] and
            reachability_plot[i] >= reachability_plot[i + 1] and
            _is_local_maxima(i, np.array(reachability_plot),
                             neighborhood_size) == 1):
            local_maxima_points[i] = reachability_plot[i]

    return sorted(local_maxima_points,
                  key=local_maxima_points.__getitem__, reverse=True)
```
### 23 - sklearn/cluster/optics_.py:

Start line: 845, End line: 886

```python
def _cluster_tree(node, parent_node, local_maxima_points,
                  reachability_plot, reachability_ordering,
                  min_cluster_size, maxima_ratio, rejection_ratio,
                  similarity_threshold, significant_min):
    # ... other code
    if (len(node_2.points) < min_cluster_size and
            node_list.count((node_2, local_max_2)) > 0):
        # cluster 2 is too small
        node_list.remove((node_2, local_max_2))
    if not node_list:
        # parent_node will be a leaf
        node.split_point = -1
        return

    # Check if nodes can be moved up one level - the new cluster created
    # is too "similar" to its parent, given the similarity threshold.
    bypass_node = 0
    if parent_node is not None:
        if ((node.end - node.start) / (parent_node.end - parent_node.start) >
                similarity_threshold):

            parent_node.children.remove(node)
            bypass_node = 1

    for nl in node_list:
        if bypass_node == 1:
            parent_node.children.append(nl[0])
            _cluster_tree(nl[0], parent_node, nl[1],
                          reachability_plot, reachability_ordering,
                          min_cluster_size, maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)
        else:
            node.children.append(nl[0])
            _cluster_tree(nl[0], node, nl[1], reachability_plot,
                          reachability_ordering, min_cluster_size,
                          maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)


def _get_leaves(node, arr):
    if node is not None:
        if node.split_point == -1:
            arr.append(node)
        for n in node.children:
            _get_leaves(n, arr)
    return arr
```
### 50 - sklearn/cluster/optics_.py:

Start line: 696, End line: 727

```python
def _automatic_cluster(reachability_plot, ordering,
                       maxima_ratio, rejection_ratio,
                       similarity_threshold, significant_min,
                       min_cluster_size, min_maxima_ratio):
    """Converts reachability plot to cluster tree and returns root node.

    Parameters
    ----------

    reachability_plot : list, required
        Reachability distances ordered by OPTICS ordering index.

    """

    min_neighborhood_size = 2
    if min_cluster_size <= 1:
        min_cluster_size = max(2, min_cluster_size * len(ordering))
    neighborhood_size = int(min_maxima_ratio * len(ordering))

    # Again, should this check < min_samples, should the parameter be public?
    if neighborhood_size < min_neighborhood_size:
        neighborhood_size = min_neighborhood_size

    local_maxima_points = _find_local_maxima(reachability_plot,
                                             neighborhood_size)
    root_node = _TreeNode(ordering, 0, len(ordering), None)
    _cluster_tree(root_node, None, local_maxima_points,
                  reachability_plot, ordering, min_cluster_size,
                  maxima_ratio, rejection_ratio,
                  similarity_threshold, significant_min)

    return root_node
```

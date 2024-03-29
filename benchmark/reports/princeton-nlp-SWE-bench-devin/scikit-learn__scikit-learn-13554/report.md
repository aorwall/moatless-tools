# scikit-learn__scikit-learn-13554

| **scikit-learn/scikit-learn** | `c903d71c5b06aa4cf518de7e3676c207519e0295` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 849 |
| **Any found context length** | 849 |
| **Avg pos** | 5.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -193,6 +193,7 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     Y_norm_squared : array-like, shape (n_samples_2, ), optional
         Pre-computed dot-products of vectors in Y (e.g.,
         ``(Y**2).sum(axis=1)``)
+        May be ignored in some cases, see the note below.
 
     squared : boolean, optional
         Return squared Euclidean distances.
@@ -200,10 +201,16 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     X_norm_squared : array-like, shape = [n_samples_1], optional
         Pre-computed dot-products of vectors in X (e.g.,
         ``(X**2).sum(axis=1)``)
+        May be ignored in some cases, see the note below.
+
+    Notes
+    -----
+    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
+    unused if they are passed as ``float32``.
 
     Returns
     -------
-    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)
+    distances : array, shape (n_samples_1, n_samples_2)
 
     Examples
     --------
@@ -224,6 +231,9 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     """
     X, Y = check_pairwise_arrays(X, Y)
 
+    # If norms are passed as float32, they are unused. If arrays are passed as
+    # float32, norms needs to be recomputed on upcast chunks.
+    # TODO: use a float64 accumulator in row_norms to avoid the latter.
     if X_norm_squared is not None:
         XX = check_array(X_norm_squared)
         if XX.shape == (1, X.shape[0]):
@@ -231,10 +241,15 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
         elif XX.shape != (X.shape[0], 1):
             raise ValueError(
                 "Incompatible dimensions for X and X_norm_squared")
+        if XX.dtype == np.float32:
+            XX = None
+    elif X.dtype == np.float32:
+        XX = None
     else:
         XX = row_norms(X, squared=True)[:, np.newaxis]
 
-    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
+    if X is Y and XX is not None:
+        # shortcut in the common case euclidean_distances(X, X)
         YY = XX.T
     elif Y_norm_squared is not None:
         YY = np.atleast_2d(Y_norm_squared)
@@ -242,23 +257,99 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
         if YY.shape != (1, Y.shape[0]):
             raise ValueError(
                 "Incompatible dimensions for Y and Y_norm_squared")
+        if YY.dtype == np.float32:
+            YY = None
+    elif Y.dtype == np.float32:
+        YY = None
     else:
         YY = row_norms(Y, squared=True)[np.newaxis, :]
 
-    distances = safe_sparse_dot(X, Y.T, dense_output=True)
-    distances *= -2
-    distances += XX
-    distances += YY
+    if X.dtype == np.float32:
+        # To minimize precision issues with float32, we compute the distance
+        # matrix on chunks of X and Y upcast to float64
+        distances = _euclidean_distances_upcast(X, XX, Y, YY)
+    else:
+        # if dtype is already float64, no need to chunk and upcast
+        distances = - 2 * safe_sparse_dot(X, Y.T, dense_output=True)
+        distances += XX
+        distances += YY
     np.maximum(distances, 0, out=distances)
 
+    # Ensure that distances between vectors and themselves are set to 0.0.
+    # This may not be the case due to floating point rounding errors.
     if X is Y:
-        # Ensure that distances between vectors and themselves are set to 0.0.
-        # This may not be the case due to floating point rounding errors.
-        distances.flat[::distances.shape[0] + 1] = 0.0
+        np.fill_diagonal(distances, 0)
 
     return distances if squared else np.sqrt(distances, out=distances)
 
 
+def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
+    """Euclidean distances between X and Y
+
+    Assumes X and Y have float32 dtype.
+    Assumes XX and YY have float64 dtype or are None.
+
+    X and Y are upcast to float64 by chunks, which size is chosen to limit
+    memory increase by approximately 10% (at least 10MiB).
+    """
+    n_samples_X = X.shape[0]
+    n_samples_Y = Y.shape[0]
+    n_features = X.shape[1]
+
+    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)
+
+    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
+    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
+
+    # Allow 10% more memory than X, Y and the distance matrix take (at least
+    # 10MiB)
+    maxmem = max(
+        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
+         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
+        10 * 2**17)
+
+    # The increase amount of memory in 8-byte blocks is:
+    # - x_density * batch_size * n_features (copy of chunk of X)
+    # - y_density * batch_size * n_features (copy of chunk of Y)
+    # - batch_size * batch_size (chunk of distance matrix)
+    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
+    #                                 xd=x_density and yd=y_density
+    tmp = (x_density + y_density) * n_features
+    batch_size = (-tmp + np.sqrt(tmp**2 + 4 * maxmem)) / 2
+    batch_size = max(int(batch_size), 1)
+
+    x_batches = gen_batches(X.shape[0], batch_size)
+    y_batches = gen_batches(Y.shape[0], batch_size)
+
+    for i, x_slice in enumerate(x_batches):
+        X_chunk = X[x_slice].astype(np.float64)
+        if XX is None:
+            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
+        else:
+            XX_chunk = XX[x_slice]
+
+        for j, y_slice in enumerate(y_batches):
+            if X is Y and j < i:
+                # when X is Y the distance matrix is symmetric so we only need
+                # to compute half of it.
+                d = distances[y_slice, x_slice].T
+
+            else:
+                Y_chunk = Y[y_slice].astype(np.float64)
+                if YY is None:
+                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
+                else:
+                    YY_chunk = YY[:, y_slice]
+
+                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
+                d += XX_chunk
+                d += YY_chunk
+
+            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)
+
+    return distances
+
+
 def _argmin_min_reduce(dist, start):
     indices = dist.argmin(axis=1)
     values = dist[np.arange(dist.shape[0]), indices]

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/metrics/pairwise.py | 196 | 196 | 1 | 1 | 849
| sklearn/metrics/pairwise.py | 203 | 203 | 1 | 1 | 849
| sklearn/metrics/pairwise.py | 227 | 227 | 1 | 1 | 849
| sklearn/metrics/pairwise.py | 234 | 234 | 1 | 1 | 849
| sklearn/metrics/pairwise.py | 245 | 254 | 1 | 1 | 849


## Problem Statement

```
Numerical precision of euclidean_distances with float32
<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
I noticed that sklearn.metrics.pairwise.pairwise_distances function agrees with np.linalg.norm when using np.float64 arrays, but disagrees when using np.float32 arrays. See the code snippet below.

#### Steps/Code to Reproduce

\`\`\`python
import numpy as np
import scipy
import sklearn.metrics.pairwise

# create 64-bit vectors a and b that are very similar to each other
a_64 = np.array([61.221637725830078125, 71.60662841796875,    -65.7512664794921875],  dtype=np.float64)
b_64 = np.array([61.221637725830078125, 71.60894012451171875, -65.72847747802734375], dtype=np.float64)

# create 32-bit versions of a and b
a_32 = a_64.astype(np.float32)
b_32 = b_64.astype(np.float32)

# compute the distance from a to b using numpy, for both 64-bit and 32-bit
dist_64_np = np.array([np.linalg.norm(a_64 - b_64)], dtype=np.float64)
dist_32_np = np.array([np.linalg.norm(a_32 - b_32)], dtype=np.float32)

# compute the distance from a to b using sklearn, for both 64-bit and 32-bit
dist_64_sklearn = sklearn.metrics.pairwise.pairwise_distances([a_64], [b_64])
dist_32_sklearn = sklearn.metrics.pairwise.pairwise_distances([a_32], [b_32])

# note that the 64-bit sklearn results agree exactly with numpy, but the 32-bit results disagree
np.set_printoptions(precision=200)

print(dist_64_np)
print(dist_32_np)
print(dist_64_sklearn)
print(dist_32_sklearn)
\`\`\`

#### Expected Results
I expect that the results from sklearn.metrics.pairwise.pairwise_distances would agree with np.linalg.norm for both 64-bit and 32-bit. In other words, I expect the following output:
\`\`\`
[ 0.0229059506440019884643266578905240749008953571319580078125]
[ 0.02290595136582851409912109375]
[[ 0.0229059506440019884643266578905240749008953571319580078125]]
[[ 0.02290595136582851409912109375]]
\`\`\`

#### Actual Results
The code snippet above produces the following output for me:
\`\`\`
[ 0.0229059506440019884643266578905240749008953571319580078125]
[ 0.02290595136582851409912109375]
[[ 0.0229059506440019884643266578905240749008953571319580078125]]
[[ 0.03125]]
\`\`\`

#### Versions
\`\`\`
Darwin-16.6.0-x86_64-i386-64bit
('Python', '2.7.11 | 64-bit | (default, Jun 11 2016, 03:41:56) \n[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]')
('NumPy', '1.11.3')
('SciPy', '0.19.0')
('Scikit-Learn', '0.18.1')
\`\`\`
[WIP] Stable and fast float32 implementation of euclidean_distances
#### Reference Issues/PRs
Fixes #9354
Superseds PR #10069

#### What does this implement/fix? Explain your changes.
These commits implement a block-wise casting to float64 and uses the older code to compute the euclidean distance matrix on the blocks. This is done useing only a fixed amount of additional (temporary) memory.

#### Any other comments?
This code implements several optimizations:

* since the distance matrix is symmetric when `X is Y`, copy the blocks of the upper triangle to the lower triangle;
* compute the optimal block size that would use most of the allowed additional memory;
* cast blocks of `{X,Y}_norm_squared` to float64;
* precompute blocks of `X_norm_squared` if not given so it gets reused through the iterations over `Y`;
* swap `X` and `Y` when `X_norm_squared` is given, but not `Y_norm_squared`.

Note that all the optimizations listed here have proven useful in a benchmark. The hardcoded amount of additional memory of 10MB is also derived from a benchmark.

As a side bonus, this implementation should also support float16 out of the box, should scikit-learn support it at some point.
Add a test for numeric precision (see #9354)
Surprisingly bad precision, isn't it?

Note that the traditional computation sqrt(sum((x-y)**2)) gets the results exact.

<!--
Thanks for contributing a pull request! Please ensure you have taken a look at
the contribution guidelines: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#pull-request-checklist
-->

#### Reference Issues/PRs
<!--
Example: Fixes #1234. See also #3456.
Please use keywords (e.g., Fixes) to create link to the issues or pull requests
you resolved, so that they will automatically be closed when your pull request
is merged. See https://github.com/blog/1506-closing-issues-via-pull-requests
-->


#### What does this implement/fix? Explain your changes.


#### Any other comments?


<!--
Please be aware that we are a loose team of volunteers so patience is
necessary; assistance handling other issues is very welcome. We value
all user contributions, no matter how minor they are. If we are slow to
review, either the pull request needs some benchmarking, tinkering,
convincing, etc. or more likely the reviewers are simply busy. In either
case, we ask for your understanding during the review process.
For more information, see our FAQ on this topic:
http://scikit-learn.org/dev/faq.html#why-is-my-pull-request-not-getting-any-attention.

Thanks for contributing!
-->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/metrics/pairwise.py** | 164 | 259| 849 | 849 | 13865 | 
| 2 | **1 sklearn/metrics/pairwise.py** | 1202 | 1318| 1273 | 2122 | 13865 | 
| 3 | **1 sklearn/metrics/pairwise.py** | 1319 | 1360| 452 | 2574 | 13865 | 
| 4 | **1 sklearn/metrics/pairwise.py** | 1449 | 1488| 401 | 2975 | 13865 | 
| 5 | **1 sklearn/metrics/pairwise.py** | 599 | 617| 114 | 3089 | 13865 | 
| 6 | **1 sklearn/metrics/pairwise.py** | 262 | 361| 829 | 3918 | 13865 | 
| 7 | **1 sklearn/metrics/pairwise.py** | 364 | 441| 637 | 4555 | 13865 | 
| 8 | **1 sklearn/metrics/pairwise.py** | 1099 | 1121| 212 | 4767 | 13865 | 
| 9 | **1 sklearn/metrics/pairwise.py** | 36 | 58| 157 | 4924 | 13865 | 
| 10 | **1 sklearn/metrics/pairwise.py** | 1491 | 1517| 192 | 5116 | 13865 | 
| 11 | **1 sklearn/metrics/pairwise.py** | 61 | 127| 630 | 5746 | 13865 | 
| 12 | **1 sklearn/metrics/pairwise.py** | 1057 | 1069| 121 | 5867 | 13865 | 
| 13 | 2 sklearn/utils/extmath.py | 48 | 77| 195 | 6062 | 20181 | 
| 14 | 2 sklearn/utils/extmath.py | 1 | 45| 201 | 6263 | 20181 | 
| 15 | 3 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 947 | 7210 | 24561 | 
| 16 | **3 sklearn/metrics/pairwise.py** | 678 | 732| 433 | 7643 | 24561 | 
| 17 | **3 sklearn/metrics/pairwise.py** | 644 | 675| 219 | 7862 | 24561 | 
| 18 | **3 sklearn/metrics/pairwise.py** | 1363 | 1448| 946 | 8808 | 24561 | 
| 19 | **3 sklearn/metrics/pairwise.py** | 489 | 559| 629 | 9437 | 24561 | 
| 20 | 4 sklearn/cluster/optics_.py | 457 | 489| 348 | 9785 | 29463 | 
| 21 | **4 sklearn/metrics/pairwise.py** | 1 | 33| 163 | 9948 | 29463 | 
| 22 | **4 sklearn/metrics/pairwise.py** | 444 | 486| 445 | 10393 | 29463 | 
| 23 | **4 sklearn/metrics/pairwise.py** | 130 | 161| 302 | 10695 | 29463 | 
| 24 | 4 benchmarks/bench_plot_randomized_svd.py | 281 | 294| 168 | 10863 | 29463 | 
| 25 | 5 sklearn/neighbors/base.py | 698 | 777| 690 | 11553 | 37195 | 
| 26 | 6 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 12307 | 38414 | 
| 27 | 6 sklearn/neighbors/base.py | 1 | 58| 434 | 12741 | 38414 | 
| 28 | **6 sklearn/metrics/pairwise.py** | 1124 | 1153| 273 | 13014 | 38414 | 
| 29 | 7 sklearn/decomposition/nmf.py | 1 | 63| 356 | 13370 | 50321 | 
| 30 | 8 sklearn/cluster/birch.py | 6 | 35| 239 | 13609 | 55610 | 
| 31 | **8 sklearn/metrics/pairwise.py** | 620 | 641| 148 | 13757 | 55610 | 
| 32 | 9 sklearn/preprocessing/data.py | 1614 | 1644| 278 | 14035 | 80736 | 
| 33 | 10 sklearn/utils/fixes.py | 1 | 39| 195 | 14230 | 82576 | 
| 34 | 11 sklearn/metrics/cluster/bicluster.py | 1 | 27| 228 | 14458 | 83293 | 
| 35 | 12 sklearn/manifold/mds.py | 92 | 132| 424 | 14882 | 87163 | 
| 36 | 13 sklearn/metrics/__init__.py | 1 | 72| 547 | 15429 | 88118 | 
| 37 | **13 sklearn/metrics/pairwise.py** | 1156 | 1162| 137 | 15566 | 88118 | 
| 38 | 13 sklearn/preprocessing/data.py | 142 | 195| 596 | 16162 | 88118 | 
| 39 | 14 benchmarks/bench_plot_parallel_pairwise.py | 3 | 47| 299 | 16461 | 88441 | 
| 40 | 15 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 16824 | 89559 | 
| 41 | 16 sklearn/cluster/k_means_.py | 1178 | 1248| 767 | 17591 | 104678 | 
| 42 | 17 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 18123 | 106623 | 
| 43 | 17 sklearn/cluster/optics_.py | 403 | 454| 528 | 18651 | 106623 | 
| 44 | 18 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 19435 | 109731 | 
| 45 | 18 sklearn/neighbors/base.py | 163 | 248| 761 | 20196 | 109731 | 
| 46 | 18 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 21011 | 109731 | 
| 47 | 18 benchmarks/bench_plot_randomized_svd.py | 297 | 340| 517 | 21528 | 109731 | 
| 48 | 18 sklearn/cluster/optics_.py | 261 | 296| 282 | 21810 | 109731 | 
| 49 | 19 examples/covariance/plot_mahalanobis_distances.py | 79 | 134| 726 | 22536 | 111394 | 
| 50 | 19 sklearn/neighbors/base.py | 421 | 492| 628 | 23164 | 111394 | 
| 51 | 20 sklearn/covariance/robust_covariance.py | 413 | 509| 1054 | 24218 | 118469 | 
| 52 | 21 sklearn/metrics/cluster/unsupervised.py | 38 | 108| 737 | 24955 | 121724 | 
| 53 | **21 sklearn/metrics/pairwise.py** | 1630 | 1646| 185 | 25140 | 121724 | 
| 54 | 22 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 26454 | 123065 | 
| 55 | 23 sklearn/cluster/dbscan_.py | 1 | 138| 1234 | 27688 | 126636 | 
| 56 | 24 examples/plot_johnson_lindenstrauss_bound.py | 94 | 168| 709 | 28397 | 128479 | 
| 57 | 24 sklearn/metrics/cluster/unsupervised.py | 152 | 226| 757 | 29154 | 128479 | 
| 58 | 25 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 29665 | 129012 | 
| 59 | **25 sklearn/metrics/pairwise.py** | 1184 | 1199| 174 | 29839 | 129012 | 
| 60 | **25 sklearn/metrics/pairwise.py** | 562 | 596| 260 | 30099 | 129012 | 
| 61 | 26 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 84| 741 | 30840 | 130137 | 
| 62 | 27 sklearn/linear_model/least_angle.py | 475 | 632| 1591 | 32431 | 146635 | 
| 63 | 27 sklearn/neighbors/base.py | 250 | 291| 343 | 32774 | 146635 | 
| 64 | 28 benchmarks/bench_plot_neighbors.py | 26 | 109| 704 | 33478 | 148066 | 
| 65 | 29 sklearn/cluster/bicluster.py | 51 | 70| 168 | 33646 | 152971 | 
| 66 | 30 examples/manifold/plot_lle_digits.py | 78 | 169| 748 | 34394 | 154995 | 
| 67 | **30 sklearn/metrics/pairwise.py** | 735 | 759| 183 | 34577 | 154995 | 
| 68 | 31 benchmarks/bench_plot_nmf.py | 370 | 422| 514 | 35091 | 158891 | 
| 69 | 32 benchmarks/bench_plot_svd.py | 1 | 53| 388 | 35479 | 159570 | 
| 70 | 32 benchmarks/bench_plot_nmf.py | 1 | 46| 287 | 35766 | 159570 | 
| 71 | 32 sklearn/cluster/dbscan_.py | 193 | 317| 1233 | 36999 | 159570 | 
| 72 | 32 examples/covariance/plot_mahalanobis_distances.py | 1 | 78| 789 | 37788 | 159570 | 
| 73 | 32 sklearn/utils/extmath.py | 660 | 689| 258 | 38046 | 159570 | 
| 74 | **32 sklearn/metrics/pairwise.py** | 1072 | 1096| 198 | 38244 | 159570 | 
| 75 | 32 sklearn/linear_model/least_angle.py | 634 | 754| 1218 | 39462 | 159570 | 
| 76 | 32 sklearn/utils/extmath.py | 692 | 767| 648 | 40110 | 159570 | 
| 77 | 33 benchmarks/bench_sparsify.py | 1 | 82| 754 | 40864 | 160477 | 
| 78 | 33 benchmarks/bench_plot_randomized_svd.py | 380 | 431| 540 | 41404 | 160477 | 
| 79 | 34 sklearn/utils/__init__.py | 1 | 72| 606 | 42010 | 166681 | 
| 80 | 34 sklearn/cluster/k_means_.py | 581 | 626| 368 | 42378 | 166681 | 
| 81 | 35 examples/cross_decomposition/plot_compare_cross_decomposition.py | 1 | 82| 724 | 43102 | 168164 | 
| 82 | 36 sklearn/utils/estimator_checks.py | 1 | 62| 444 | 43546 | 190039 | 
| 83 | 37 benchmarks/bench_tsne_mnist.py | 69 | 169| 1011 | 44557 | 191465 | 
| 84 | 38 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 45224 | 192132 | 
| 85 | 38 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 45983 | 192132 | 
| 86 | 38 sklearn/cluster/bicluster.py | 29 | 48| 212 | 46195 | 192132 | 
| 87 | 39 examples/model_selection/plot_precision_recall.py | 101 | 205| 770 | 46965 | 194474 | 
| 88 | 39 benchmarks/bench_plot_svd.py | 56 | 83| 290 | 47255 | 194474 | 
| 89 | 39 examples/plot_johnson_lindenstrauss_bound.py | 170 | 208| 376 | 47631 | 194474 | 
| 90 | 40 sklearn/linear_model/base.py | 177 | 191| 155 | 47786 | 199235 | 


### Hint

```
Same results with python 3.5 :

\`\`\`
Darwin-15.6.0-x86_64-i386-64bit
Python 3.5.1 (v3.5.1:37a07cee5969, Dec  5 2015, 21:12:44) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)]
NumPy 1.11.0
SciPy 0.18.1
Scikit-Learn 0.17.1
\`\`\`

It happens only with euclidean distance and can be reproduced using directly `sklearn.metrics.pairwise.euclidean_distances` :

\`\`\`
import scipy
import sklearn.metrics.pairwise

# create 64-bit vectors a and b that are very similar to each other
a_64 = np.array([61.221637725830078125, 71.60662841796875,    -65.7512664794921875],  dtype=np.float64)
b_64 = np.array([61.221637725830078125, 71.60894012451171875, -65.72847747802734375], dtype=np.float64)

# create 32-bit versions of a and b
a_32 = a_64.astype(np.float32)
b_32 = b_64.astype(np.float32)

# compute the distance from a to b using sklearn, for both 64-bit and 32-bit
dist_64_sklearn = sklearn.metrics.pairwise.euclidean_distances([a_64], [b_64])
dist_32_sklearn = sklearn.metrics.pairwise.euclidean_distances([a_32], [b_32])

np.set_printoptions(precision=200)

print(dist_64_sklearn)
print(dist_32_sklearn)
\`\`\`

I couldn't track down further the error.
I hope this can help.


numpy might use a higher precision accumulator. yes, it looks like this
deserves fixing.

On 19 Jul 2017 12:05 am, "nvauquie" <notifications@github.com> wrote:

> Same results with python 3.5 :
>
> Darwin-15.6.0-x86_64-i386-64bit
> Python 3.5.1 (v3.5.1:37a07cee5969, Dec  5 2015, 21:12:44)
> [GCC 4.2.1 (Apple Inc. build 5666) (dot 3)]
> NumPy 1.11.0
> SciPy 0.18.1
> Scikit-Learn 0.17.1
>
> It happens only with euclidean distance and can be reproduced using
> directly sklearn.metrics.pairwise.euclidean_distances :
>
> import scipy
> import sklearn.metrics.pairwise
>
> # create 64-bit vectors a and b that are very similar to each other
> a_64 = np.array([61.221637725830078125, 71.60662841796875,    -65.7512664794921875],  dtype=np.float64)
> b_64 = np.array([61.221637725830078125, 71.60894012451171875, -65.72847747802734375], dtype=np.float64)
>
> # create 32-bit versions of a and b
> a_32 = a_64.astype(np.float32)
> b_32 = b_64.astype(np.float32)
>
> # compute the distance from a to b using sklearn, for both 64-bit and 32-bit
> dist_64_sklearn = sklearn.metrics.pairwise.euclidean_distances([a_64], [b_64])
> dist_32_sklearn = sklearn.metrics.pairwise.euclidean_distances([a_32], [b_32])
>
> np.set_printoptions(precision=200)
>
> print(dist_64_sklearn)
> print(dist_32_sklearn)
>
> I couldn't track down further the error.
> I hope this can help.
>
> —
> You are receiving this because you are subscribed to this thread.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/9354#issuecomment-316074315>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz65yy8Aq2FcsDAcWHT8qkkdXF_MfPks5sPLu_gaJpZM4OXbpZ>
> .
>

I'd like to work on this if possible 
Go for it!
So I think the problem lies around the fact that we are using `sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))` for computing euclidean distance 
Because if I try - ` (-2 * np.dot(X, Y.T) + (X * X).sum(axis=1) + (Y * Y).sum(axis=1)` I get the answer 0 for np.float32, while I get the correct ans for np.float 64.
@jnothman What do you think I should do then ? As mentioned in my comment above the problem is probably computing euclidean distance using `sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))`
So you're saying that dot is returning a less precise result than product-then-sum?
No, what I'm trying to say is dot is returning more precise result than product-then-sum
`-2 * np.dot(X, Y.T) + (X * X).sum(axis=1) + (Y * Y).sum(axis=1)` gives output  `[[0.]]`
while `np.sqrt(((X-Y) * (X-Y)).sum(axis=1))` gives output `[ 0.02290595]`
It is not clear what you are doing, partly because you are not posting a fully stand-alone snippet.

Quickly looking at your last post the two things you are trying to compare `[[0.]]` and `[0.022...]` do not have the same dimensions (maybe a copy and paste problem but again hard to know because we don't have a full snippet).
Ok sorry my bad
\`\`\`
import numpy as np
import scipy
from sklearn.metrics.pairwise import check_pairwise_arrays, row_norms
from sklearn.utils.extmath import safe_sparse_dot

# create 64-bit vectors a and b that are very similar to each other
a_64 = np.array([61.221637725830078125, 71.60662841796875,    -65.7512664794921875],  dtype=np.float64)
b_64 = np.array([61.221637725830078125, 71.60894012451171875, -65.72847747802734375], dtype=np.float64)

# create 32-bit versions of a and b
X = a_64.astype(np.float32)
Y = b_64.astype(np.float32)

X, Y = check_pairwise_arrays(X, Y)
XX = row_norms(X, squared=True)[:, np.newaxis]
YY = row_norms(Y, squared=True)[np.newaxis, :]

#Euclidean distance computed using product-then-sum
distances = safe_sparse_dot(X, Y.T, dense_output=True)
distances *= -2
distances += XX
distances += YY
print(np.sqrt(distances))

#Euclidean distance computed using (X-Y)^2
print(np.sqrt(row_norms(X-Y, squared=True)[:, np.newaxis]))

\`\`\`

**OUTPUT**
\`\`\`
[[ 0.03125]]
[[ 0.02290595136582851409912109375]]
\`\`\`
The first method is how it is computed by the euclidean distance function. 
Also to clarify what I meant above was the fact that sum-then-product has lower precision even when we use numpy functions to do it

Yes, I can replicate this. I see that doing the subtraction initially
allows the precision of the difference to be maintained. Doing the dot
product and then subtracting (or negating and adding), as we currently do,
loses this precision as the most significant figures are much larger than
the differences.

The current implementation is more memory efficient for a high number of
features. But I suppose euclidean distance becomes increasingly irrelevant
in high dimensions, so the memory is dominated by the number of output
values.

So I vote for adopting the more numerically stable implementation over the
d-asymptotically efficient implementation we currently have. An opinion,
@ogrisel? @agramfort?

And this is of course more of a concern since we recently allowed float32s
to be more commonplace across estimators.

So for this example product-then-sum works perfectly fine for np.float64, so a possible solution could be to convert the input to float64 then compute the result and return the result converted back to float32. I guess this would be more efficient, but not sure if this would work fine for some other example.
converting to float64 won't be more efficient in memory usage than
subtraction.

Oh yeah you are right sorry about that, but I think using float64 and then doing product-then-sum would be more efficient computationally if not memory wise.
And the reason for using product-then-sum was to have more computational efficiency and not memory efficiency.
sure, but I don't believe there is any reason to assume that it is in fact
more computationally efficient except by way of not having to realise an
intermediate array. Assuming we limit absolute working memory (e.g. by
chunking), why would taking the dot product, doubling and subtracting norms
be much more efficient than subtracting and squaring?

Provide benchmarks?

Ok so I created a python script to compare the time taken by subtraction-then-squaring and conversion to float64 then product-then-sum and it turns out if we choose an X and Y as very big vectors then the 2 results are very different. Also @jnothman you were right subtraction-then-squaring is faster. 
Here's the script that I wrote, if there's any problem please let me know 

\`\`\`
import numpy as np
import scipy
from sklearn.metrics.pairwise import check_pairwise_arrays, row_norms
from sklearn.utils.extmath import safe_sparse_dot
from timeit import default_timer as timer

for i in range(9):
	X = np.random.rand(1,3 * (10**i)).astype(np.float32)
	Y = np.random.rand(1,3 * (10**i)).astype(np.float32)

	X, Y = check_pairwise_arrays(X, Y)
	XX = row_norms(X, squared=True)[:, np.newaxis]
	YY = row_norms(Y, squared=True)[np.newaxis, :]

	#Euclidean distance computed using product-then-sum
	distances = safe_sparse_dot(X, Y.T, dense_output=True)
	distances *= -2
	distances += XX
	distances += YY

	ans1 = np.sqrt(distances)

	start = timer()
	ans2 = np.sqrt(row_norms(X-Y, squared=True)[:, np.newaxis])
	end = timer()
	if ans1 != ans2:
		print(end-start)

		start = timer()
		X = X.astype(np.float64)
		Y = Y.astype(np.float64)
		X, Y = check_pairwise_arrays(X, Y)
		XX = row_norms(X, squared=True)[:, np.newaxis]
		YY = row_norms(Y, squared=True)[np.newaxis, :]
		distances = safe_sparse_dot(X, Y.T, dense_output=True)
		distances *= -2
		distances += XX
		distances += YY
		distances = np.sqrt(distances)
		end = timer()
		print(end-start)
		print('')
		if abs(ans2 - distances) > 1e-3:
			# np.set_printoptions(precision=200)
			print(ans2)
			print(np.sqrt(distances))

			print(X, Y)
			break
\`\`\`
it's worth testing how it scales with the number of samples, not just the
number of features... taking norms may have the benefit of computing some
things once per sample, not once per pair of samples

On 20 Oct 2017 2:39 am, "Osaid Rehman Nasir" <notifications@github.com>
wrote:

> Ok so I created a python script to compare the time taken by
> subtraction-then-squaring and conversion to float64 then product-then-sum
> and it turns out if we choose an X and Y as very big vectors then the 2
> results are very different. Also @jnothman <https://github.com/jnothman>
> you were right subtraction-then-squaring is faster.
> Here's the script that I wrote, if there's any problem please let me know
>
> import numpy as np
> import scipy
> from sklearn.metrics.pairwise import check_pairwise_arrays, row_norms
> from sklearn.utils.extmath import safe_sparse_dot
> from timeit import default_timer as timer
>
> for i in range(9):
> 	X = np.random.rand(1,3 * (10**i)).astype(np.float32)
> 	Y = np.random.rand(1,3 * (10**i)).astype(np.float32)
>
> 	X, Y = check_pairwise_arrays(X, Y)
> 	XX = row_norms(X, squared=True)[:, np.newaxis]
> 	YY = row_norms(Y, squared=True)[np.newaxis, :]
>
> 	#Euclidean distance computed using product-then-sum
> 	distances = safe_sparse_dot(X, Y.T, dense_output=True)
> 	distances *= -2
> 	distances += XX
> 	distances += YY
>
> 	ans1 = np.sqrt(distances)
>
> 	start = timer()
> 	ans2 = np.sqrt(row_norms(X-Y, squared=True)[:, np.newaxis])
> 	end = timer()
> 	if ans1 != ans2:
> 		print(end-start)
>
> 		start = timer()
> 		X = X.astype(np.float64)
> 		Y = Y.astype(np.float64)
> 		X, Y = check_pairwise_arrays(X, Y)
> 		XX = row_norms(X, squared=True)[:, np.newaxis]
> 		YY = row_norms(Y, squared=True)[np.newaxis, :]
> 		distances = safe_sparse_dot(X, Y.T, dense_output=True)
> 		distances *= -2
> 		distances += XX
> 		distances += YY
> 		distances = np.sqrt(distances)
> 		end = timer()
> 		print(end-start)
> 		print('')
> 		if abs(ans2 - distances) > 1e-3:
> 			# np.set_printoptions(precision=200)
> 			print(ans2)
> 			print(np.sqrt(distances))
>
> 			print(X, Y)
> 			break
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/9354#issuecomment-337948154>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz6z5o2Ao_7V5-Lflb4HosMrHCeOrVks5st209gaJpZM4OXbpZ>
> .
>

anyway, would you like to submit a PR, @ragnerok?
yeah sure, what do you want me to do ?
provide a more stable implementation, also a test that would fail under the
current implementation, and ideally a benchmark that shows we do not lose
much from the change, in reasonable cases.

I wanted to ask if it is possible to find distance between each pair of rows with vectorisation. I cannot think about how to do it vectorised.
You mean difference (not distance) between pairs of rows? Sure you can do that if you're working with numpy arrays. If you have arrays with shapes (n_samples1, n_features) and (n_samples2, n_features), you just need to reshape it to (n_samples1, 1, n_features) and (1, n_samples2, n_features) and do the subtraction:
\`\`\`python
>>> X = np.random.randint(10, size=(10, 5))
>>> Y = np.random.randint(10, size=(11, 5))
X.reshape(-1, 1, X.shape[1]) - Y.reshape(1, -1, X.shape[1])
\`\`\`
Yeah thanks that really helped 😄 
I also wanted to ask if I provide a more stable implementation I won't be using X_norm_squared and Y_norm_squared. So do I remove them from the arguments as well or should I warn about it not being of any use ?
I think they will be deprecated, but we might need to first be assured that
there's no case where we should keep that version.

we're going to be quite careful in changing this. it's a widely used and
longstanding implementation. we should be sure not to slow any important
cases. we might need to do the operation in chunks to avoid high memory
usage (which is perhaps made trickier by the fact that this is called
within functions that chunk to minimise the output memory retirement from
pairwise distances).

I'd really like to hear from other core devs who know about computational
costs and numerical precision... @ogrisel, @lesteve, @rth...

On 5 Nov 2017 5:27 am, "Osaid Rehman Nasir" <notifications@github.com>
wrote:

> I also wanted to ask if I provide a more stable implementation I won't be
> using X_norm_squared and Y_norm_squared. So do I remove them from the
> arguments as well or should I warn about it not being of any use ?
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/9354#issuecomment-341919282>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz63izdpQGDEuW32m8Aob6rrsvV6q-ks5szKyHgaJpZM4OXbpZ>
> .
>

but it would be easier to discuss precisely if you open a PR

Ok I'll open up a PR then, with a very basic implementation of this function
The questions is what should be done about this for the 0.20 release. Could there be some simple / temporary improvements (event at the cost e.g. of memory usage) that could be considered?

The solution and analysis proposed in #11271 are definitely very valuable, but it might require some more discussion to make sure this is the optimal solution. In particular, I am concerned about the fact that now we have some pending discussion about the optimal global working memory in  https://github.com/scikit-learn/scikit-learn/issues/11506 depending on the CPU type etc while this would add yet another level of chunking and the complexity of the whole would be getting a bit of control IMO. But maybe it's just me, looking for a second opinion.

What do you think should be done about this issue for the release @jnothman @amueller @ogrisel ?
Stability trumps efficiency. Stability issues should be fixed even when
efficiency still needs tweaks.

working_memory's focus was to make things like silhouette with large sample
sizes work. It also improved efficiency, but that can be tweaked down the
line.

I strongly believe we should try to get a fix for euclidean_distances with
float32 in. We broke it in 0.19 by assuming that we could make
euclidean_distances work on 32 bit in a naive way.

I agree that we need a fix. My concern here is not efficiency but the added complexity in the code base.

Taking a step back, scipy's euclidean implementation seems to be [10 lines of C code](https://github.com/scipy/scipy/blob/5e22b2e447cec5588fb42303a1ae796ab2bf852d/scipy/spatial/src/distance_impl.h#L49) and for 32 bit, simply cast them to 64bit. I understand that it's not the fastest but it's conceptually easy to follow and understand.  In scikit-learn, we use the trick to make computations faster in BLAS, then there are possible improvements due in https://github.com/scikit-learn/scikit-learn/pull/10212  and now the possible chunked solution to euclidean distance in 32 bit.

I'm just looking for input about what the general direction on this topic should be (e.g try to upstream some of it to scipy etc). 
scipy doesn't seem concerned by copying the data...

Move to 0.21 following the PR.
Remove the blocker?
`sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))`

is numerically unstable, if dot(x,x) and dot(y,y) are of similar magnitude as dot(x,y) because of what is known as **catastrophic cancellation**.

This not only affect FP32 precision, but it is of course more prominent, and will fail much earlier.

Here is a simple test case that shows how bad this is even with double precision:
\`\`\`
import numpy
from sklearn.metrics.pairwise import euclidean_distances

a = numpy.array([[100000001, 100000000]])
b = numpy.array([[100000000, 100000001]])

print "skelarn:", euclidean_distances(a, b)[0,0]
print "correct:", numpy.sqrt(numpy.sum((a-b)**2))

a = numpy.array([[10001, 10000]], numpy.float32)
b = numpy.array([[10000, 10001]], numpy.float32)

print "skelarn:", euclidean_distances(a, b)[0,0]
print "correct:", numpy.sqrt(numpy.sum((a-b)**2))
\`\`\`
sklearn computes a distance of 0 here both times, rather than sqrt(2).

A discussion of the numerical issues for variance and covariance - and this trivially carries over to this approach of accelerating euclidean distance - can be found here:

> Erich Schubert, and Michael Gertz.
> **Numerically Stable Parallel Computation of (Co-)Variance.**
> In: Proceedings of the 30th International Conference on Scientific and Statistical Database Management (SSDBM), Bolzano-Bozen, Italy. 2018, 10:1–10:12

Actually the y coordinate can be removed from that test case, the correct distance then trivially becomes 1. I made a pull request that triggers this numeric problem:
\`\`\`
    XA = np.array([[10000]], np.float32)
    XB = np.array([[10001]], np.float32)
    assert_equal(euclidean_distances(XA, XB)[0,0], 1)
\`\`\`
I don't think my paper mentioned above provides a solution for this problem - just compute Euclidean distance as sqrt(sum(power())) and it is single-pass and has reasonable precision. The loss is in using the squares already, i.e., dot(x,x) itself already losing the precision.

@amueller as the problem may be more sever than expected, I suggest re-adding the blocker label...
Thanks for this very simple example.

The reason it is implemented this way is because it's way faster. See below:
\`\`\`python
x = np.random.random_sample((1000, 1000))

%timeit euclidean_distances(x,x)
20.6 ms ± 452 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit cdist(x,x)
695 ms ± 4.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
\`\`\`

Although the number of operations is of the same order in both methods (1.5x more in the second one), the speedup comes from the possibility to use well optimized BLAS libraries for matrix matrix multiplication.

This would be a huge slowdown for several estimators in scikit-learn.
Yes, but **just 3-4 digits of precision** with FP32, and 7-8 digits with FP64 *does* cause substantial imprecision, doesn't it? In particular, since such errors tend to amplify...
Well I'm not saying that it's fine right now. :)
I'm saying that we need to find a solution in between.
There is a PR (#11271) which proposes to cast on float64 to do the computations. In does not fix the problem for float64 but gives better precision for float32.

Do you have an example where using an estimator which uses euclidean_distances gives wrong results due to the loss of precision ?
I certainly still think this is a big deal and should be a blocker for 0.21. It was an issue introduced for 32 bit in 0.19, and it's not a nice state of affairs to leave. I wish we had resolved it earlier in 0.20, and I would be okay, or even keen, to see #11271 merged in the interim. The only issues in that PR that I know of surround optimisation of memory efficiency, which is a deep rabbit hole.

We've had this "fast" version for a long time, but always in float64. I know, @kno10, that it's got issues with precision. Do you have a good and fast heuristic for us to work out when that might be a problem and use a slower-but-surer solution?
> Yes, but just 3-4 digits of precision with FP32, and 7-8 digits with FP64 does cause substantial imprecision, doesn't it

Thanks for illustrating this issue with very simple example!

I don't think the issue is as widespread as you suggest, however -- it mostly affects samples whose mutual distance small with respect to their norms.

The below figure illustrates this, for 2e6 random sample pairs, where each 1D samples is in the interval [-100, 100]. The relative error between the scikit-learn and scipy implementation is plotted as a function of the distance between samples, normalized by their L2 norms, i.e.,
\`\`\`
d_norm(A, B) = d(A, B) / sqrt(‖A‖₂*‖B‖₂)
\`\`\`
(not sure it's the right parametrization, but just to get results somewhat invariant to the data scale),
![euclidean_distance_precision_1d](https://user-images.githubusercontent.com/630936/45919546-41ea1880-be97-11e8-9707-9279dfac4f5b.png)


For instance, 
  1. if one takes `[10000]` and `[10001]` the L2 normalized distance is 1e-4 and the relative error on the distance calculation will be 1e-8 in 64 bit, and >1 in 32 bit (Or 1e-8 and >1 in absolute value respectively). In 32 bit this case is indeed quite terrible.
  2. on the other hand for `[1]` and `[10001]`, the relative error will be ~1e-7 in 32 bit, or the maximum possible precision. 

The question is how often the case 1. will happen in practice in ML applications. 

Interestingly, if we go to 2D, again with a uniform random distribution, it will be difficult to find points that are very close,
![euclidean_distance_precision_2d](https://user-images.githubusercontent.com/630936/45919664-37308300-be99-11e8-8a01-5f936524aea5.png)

Of course, in reality our data will not be uniformly sampled, but for any distribution because of the curse of dimensionality the distance between any two points will slowly converge to very similar values (different from 0) as the dimentionality increases. While it's a general ML issue, here it may mitigate somewhat this accuracy problem, even for relatively low dimensionality. Below the results for `n_features=5`,
![euclidean_distance_precision_5d](https://user-images.githubusercontent.com/630936/45919716-3fd58900-be9a-11e8-9a5f-17c1a7c60102.png).

So for centered data, at least in 64 bit, it may not be so much of an issue in practice (assuming there are more then 2 features). The 50x computational speed-up (as illustrated above) may be worth it (in 64 bit). Of course one can always add 1e6 to some data normalized in [-1, 1] and say that the results are not accurate, but I would argue that the same applies to a number of numerical algorithms, and working with data expressed in the 6th significant digit is just looking for trouble.

(The code for the above figures can be found [here](https://github.com/rth/ipynb/blob/master/sklearn/euclidean_distance_accuracy.ipynb)).

Any fast approach using the dot(x,x)+dot(y,y)-2*dot(x,y) version will likely have the same issue for all I can tell, but you'd better ask some real expert on numerics for this. I believe you'll need to double the precision of the dot products to get to approx. the precision of the *input* data (and I'd assume that if a user provides float32 data, then they'll want float32 precision, with float64, they'll want float64 precision). You may be able to do this with some tricks (think of Kahan summation), but it will very likely cost you much more than you gained in the first place.

I can't tell how much overhead you get from converting float32 to float64 on the fly for using this approach. At least for float32, to my understanding, doing all the computations and storing the dot products as float64 should be fine.

IMHO, the performance gains (which are not exponential, just a constant factor) are not worth the loss in precision (which can bite you unexpectedly) and the proper way is to not use this problematic trick. It may, however, be well possible to further optimize code doing the "traditional" computation, for example to use AVX. Because sum( (x-y)**2 ) is all but difficult to implement in AVX.
At the minimum, I would suggest renaming the method to `approximate_euclidean_distances`, because of the sometimes low precision (which gets worse the closer two values are, which *may* be fine initially then begin to matter when converging to some optimum), so that users are aware of this issue.
@rth thanks for the illustrations. But what if you are trying to optimize, e.g., x towards some optimum. Most likely the optimum will not be at zero (if it would always be your data center, life would be great), and eventually the deltas you are computing for gradients etc. may have some very small differences.
Similarly, in clustering, clusters will not all have their centers close to zero, but in particular with many clusters, x ≈ center with a few digits is quite possible.
Overall however, I agree this issue needs fixing. In any case we need to document the precision issues of the current implementation as soon as possible.

In general though I don't think the this discussion should happen in scikit-learn. Euclidean distance is used in various fields of scientific computing and IMO scipy mailing list or issues is a better place to discuss it: that community has also more experience with such numerical precision issues. In fact what we have here is a fast but somewhat approximate algorithm. We may have to implement some fixes workarounds in the short term, but in the long term it would be good to know that this will be contributed there.

For 32 bit, https://github.com/scikit-learn/scikit-learn/pull/11271 may indeed be a solution, I'm just not so keen of multiple levels of chunking all through the library as that increases code complexity, and want to make sure there is no better way around it.
Thanks for your response @kno10! (My above comments doesn't take it into account yet) I'll respond a bit later.
Yes, convergence to some point outside of the origin may be an issue.

> IMHO, the performance gains (which are not exponential, just a constant factor) are not worth the loss in precision (which can bite you unexpectedly) and the proper way is to not use this problematic trick.

Well a >10x slow down for their calculation in 64 bit will have a very real effect on users.

> It may, however, be well possible to further optimize code doing the "traditional" computation, for example to use AVX. Because sum( (x-y)**2 ) is all but difficult to implement in AVX.

Tried a quick naive implementation with numba (which should use SSE),
\`\`\`py
@numba.jit(nopython=True, fastmath=True)              
def pairwise_distance_naive(A, B):
    n_samples_a, n_features_a = A.shape
    n_samples_b, n_features_b = B.shape
    assert n_features_a == n_features_b
    distance = np.empty((n_samples_a, n_samples_b), dtype=A.dtype)
    for i in range(n_samples_a):
        for j in range(n_samples_b):
            psum = 0.0
            for k in range(n_features_a):
                psum += (A[i, k] - B[j, k])**2
            distance[i, j] = math.sqrt(psum)
    return distance
\`\`\`
getting a similar speed to scipy `cdist` so far (but I'm not a numba expert), and also not sure about the effect of `fastmath`.

>  using the dot(x,x)+dot(y,y)-2*dot(x,y) version

Just for future reference, what we are currently doing is roughly the following (because there is a dimension that doesn't show in the above notation),
\`\`\`py
def quadratic_pairwise_distance(A, B):
    A2 = np.einsum('ij,ij->i', A, A)
    B2 = np.einsum('ij,ij->i', B, B)
    return np.sqrt(A2[:, None] + B2[None, :] - 2*np.dot(A, B.T))
\`\`\`
where both `einsum` and `dot` now use BLAS. I wonder, if aside from using BLAS, this also actually does the same number of mathematical operations as the first version above. 
>  I wonder, if aside from using BLAS, this also actually does the same number of mathematical operations as the first version above.

No. The ((x - y)**2.sum()) performs
*n_samples_x * n_samples_y * n_features * (1 substraction + 1 addition + 1 multiplication)*
 whereas the x.x + y.y -2x.y performs 
*n_samples_x * n_samples_y * n_features * (1 addition + 1 multiplication)*.
There is a 2/3 ratio for the number of operations between the 2 versions.
Following the above discussion,
 - Made a PR to optionally allow computing euclidean distances exactly https://github.com/scikit-learn/scikit-learn/pull/12136
 - Some WIP to see if we can detect and mitigate the problematic points in https://github.com/scikit-learn/scikit-learn/pull/12142

For 32 bit, we still need to merge https://github.com/scikit-learn/scikit-learn/pull/11271 in some form though IMO, the above PRs are somewhat orthogonal to it.
FYI: when fixing some issues in OPTICS, and refreshing the test to use reference results from ELKI, these fail with `metric="euclidean"` but succeed with `metric="minkowski"`. The numerical differences are large enough to cause a different processing order (just decreasing the threshold is not enough).

https://github.com/kno10/scikit-learn/blob/ab544709a392e4dc7b22e9fd60c4755baf3d3053/sklearn/cluster/tests/test_optics.py#L588
I'm really not caught up on this, but I'm surprised there's no existing solution. This seems to be a very common computation and it looks like we're reinventing the wheel. Has anyone tried reaching out to the wider scientific computing community?
Not yet, but I agree we should. The only thing I found about this in scipy was https://github.com/scipy/scipy/pull/2815 and linked issues.
I feel @jeremiedbb might have an idea?
Unfortunately not a satisfying one yet :(

We'd like to rely on a highly optimized library for this kind of computation, as we do for linear algebra with BLAS libraries such as OpenBLAS or MKL. But euclidean distance is not part of it. The dot trick is an attempt at doing that relying on BLAS level 3 matrix-matrix multiplication subroutine. But this is not precise and there is no way to make it more precise using the same method. We have to lower our expectancy either in term of speed or in term of precision.

I think in some situations, full precision is not mandatory and keeping the fast method is fine. This is when the distances are used for "find the closest" tasks. The precision issues in the fast method appear when the distances between points is small compared to their norm (in a ratio ~< 1e-4 for float 32 and ~< 1e-8 for float64). First for this situation to happen, the dataset needs to be quite dense. Then to have an ordering error, you need to have the two closest points within almost the same distance. Moreover, in that case, in a ML point of view, both would lead to almost equally good fits.

In the above situation, there is something we can do to lower the frequency of these wrong ordering (down to 0 ?). In the pairwise distance argmin situation. We can move the misordering to points which are not the closest. Essentially using the fact that one of the norm is not necessary to find the argmin, see [comment](https://github.com/scikit-learn/scikit-learn/pull/11950#issuecomment-429916562). It has 2 advantages. It's a more robust (up to now I haven't found a wrong ordering yet) and it is even faster because it avoids some computations.

One drawback, still in the same situation, if at the end we want the actual distances to the closest points, the distances computed with the above method can't be used. They are only partially computed and they are not precise anyway. We need to re-compute the distances from each point to it's closest point. But this is fast because for each point there is only one distance to compute.

I wonder what I described above covers all the use case of euclidean_distances in sklearn. But I suggest to do that wherever it can be applied. To do that we can add a new parameter to euclidean_distances to only compute the necessary part in order to chain it with argmin. Then use it in pairwise_distances_argmin and in pairwise_distances_argmin_min (re computing the actual min distances at the end in the latter).

When we can't do that, fall back to the slow yet precise one, or add a switch like in #12136.
We can try to optimize it a bit to lower the performance drop cause I agree that [this](https://github.com/scikit-learn/scikit-learn/pull/12136#issuecomment-439097748) does not seem optimal. I have a few ideas for that.

Another possibility to keep using BLAS is combining `axpy` with `nrm2` but this is far from optimal. Both are BLAS level 1 functions, and it involves a copy. This would only be faster in dimension > 100.
Ideally we'd like the euclidean distance to be included in BLAS...

Finally, there is another solution, consisting in upcasting. This is done in #11271 for float32. The advantage is that the speed is just half the current one and precision is kept. It does not solve the problem for float64 however. Maybe we can find a way to do a similar thing in cython for float64. I don't know exactly how but using 2 float64 numbers to kind of simulate a float128. I can give it a try to see if it's somewhat doable.
> Ideally we'd like the euclidean distance to be included in BLAS...

Is that something the libraries would consider? If OpenBLAS does it we would be in a pretty good situation already...

Also, what's the exact differences between us doing it and the BLAS doing it? Detecting the CPU capabilities and deciding which implementation to use, or something like that? Or just having compiled versions for more diverse architectures?
Or just more time/energy spend writing efficient implementations?
This is interesting: an alternative implementation of the fast unstable method but claiming to be much faster than sklearn:
https://github.com/droyed/eucl_dist
(doesn't solve this issue at all though lol)
This discussion seems related https://github.com/scipy/scipy/issues/5657
Here's what julia does: https://github.com/JuliaStats/Distances.jl/blob/master/README.md#precision-for-euclidean-and-sqeuclidean
It allows setting a precision threshold to force recalculation.
Answering my own question: OpenBLAS has what looks like hand-written assembly for each processor (not architecture!) and heutistics to choose kernels for different problem sizes. So I don't think it's an issue of getting it into openblas as much as finding someone to write/optimize all those kernels...
Thanks for the additional thoughts!

In a partial response,

> We'd like to rely on a highly optimized library for this kind of computation, as we do for linear algebra with BLAS libraries such as OpenBLAS or MKL.

Yeah, I also was hoping we could do more of this in BLAS. Last time I looked nothing in standard BLAS API looks close enough (but then I'm not an expert on those). [BLIS](https://github.com/flame/blis) might offer more flexibility but since we are not using it by default it's of somewhat limited use (though numpy might someday https://github.com/numpy/numpy/issues/7372) 

> Here's what julia does: It allows setting a precision threshold to force recalculation.

Great to know!


Should we open a separate issue for the faster approximate computation linked above? Seems interesting
Their speedup on CPU of x2-x4 might be due to https://github.com/scikit-learn/scikit-learn/pull/10212 .

I would rather open an issue on scipy once we have studied this question enough to come up with a reasonable solution there (and then possibly backport it) as I feel euclidean distance is something basic enough that should be of interest to many people outside of ML (and at the same time having the opinion of people there e.g. on accuracy issues would be helfpul).
It's up to 60x, right?
> This is interesting: an alternative implementation of the fast unstable method but claiming to be much faster than sklearn

hum not sure about that. They are benchmarking `%timeit pairwise_distances(a,b, 'sqeuclidean')`, which uses scipy's one. They should do `%timeit pairwise_distances(a,b, 'euclidean', metric_params={'squared': True})` and their speedup wouldn't be as good :)
As shown far earlier in the discussion, sklearn can be 35x faster than scipy
Yes, they benchmarks are only ~30% better better with `metric="euclidean"` (instead of `squeclidean`),

\`\`\`py
In [1]: from eucl_dist.cpu_dist import dist                                                                                                                  
    ... import numpy as np                                                                                                                                   
In [4]: rng = np.random.RandomState(1)                                                                                                                        
    ... a = rng.rand(1000, 300)                                                                                                                              
    ...b = rng.rand(1000, 300)                                                                                                                              

In [7]: from sklearn.metrics.pairwise import pairwise_distances                                                                                              
In [8]: %timeit pairwise_distances(a, b, 'sqeuclidean')                                                                                                      
214 ms ± 2.06 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

In [9]: %timeit pairwise_distances(a, b)                                                                                                                     
27.4 ms ± 2.48 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [10]: from eucl_dist.cpu_dist import dist                                                                                                                 
In [11]: %timeit dist(a, b, matmul='gemm', method='ext', precision='float32')                                                                                
20.8 ms ± 330 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [12]: %timeit dist(a, b, matmul='gemm', method='ext', precision='float64')                                                                                
20.4 ms ± 222 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
\`\`\`
> Is that something the libraries would consider? If OpenBLAS does it we would be in a pretty good situation already...

Doesn't sound straightforward. BLAS is a set of specs for linear algebra routines and there are several implementations of it. I don't know how open they are to adding new features not in the original specs. For that maybe blis would be more open but as said before, it's not the default for now.
Opened https://github.com/scikit-learn/scikit-learn/issues/12600 on the `sqeuclidean` vs `euclidean` handling in `pairwise_distances`.
I need some clarity about what we want for this. Do we want `pairwise_distances` to be close - in the sense of `all_close` - for both 'euclidean' and 'sqeuclidean' ?

It's a bit tricky. Because x is close to y does not mean x² is close to y². Precision is lost during squaring.

The julia workaround linked above is very interesting and is kind of straightforward to implement. However I suspect that it does not work as expected for 'sqeuclidean'. I suspect that you have to set the threshold way below to get the desired precision.

The issue with setting a very low threshold is that it induces a lot of re-computations and a huge drop of performances. However this is mitigated by the dimension of the dataset. The same threshold will trigger way less re-computations in high dimension (distances are bigger). 

Maybe we can have 2 implementations and switch depending on the dimension of the dataset. The slow but safe one for low dimensional ones (there not much difference between scipy and sklearn in that case anyway) and the fast + threshold one for high dimensional ones.

This will need some benchmarks to find when to switch, and set the threshold but this may be a glimmer of hope :)
Here are some benchmarks for speed comparison between scipy and sklearn. The benchmarks compare `sklearn.metrics.pairwise.euclidean_distances(X,X)` with `scipy.spatial.distance.cdist(X,X)` for Xs of all sizes. Number of samples goes from 2⁴ (16) to 2¹³ (8192), and number of features goes from 2⁰ (1) to 2¹³ (8192).

The value in each cell is the speedup of sklearn vs scipy, i.e. below 1 sklearn is slower and above 1 sklearn is faster.

The first benchmark is using the MKL implementation of BLAS and a single core.
![bench_euclidean_mkl_1](https://user-images.githubusercontent.com/34657725/48772816-c6092280-ecc5-11e8-94fe-68a7a5cdf304.png)

The second one is using the OpenBLAS implementation of BLAS and a single core. It's just to check that both MKL and OpenBLAS have the same behavior.
![bench_euclidean_openblas_1](https://user-images.githubusercontent.com/34657725/48772823-cacdd680-ecc5-11e8-95f7-0f9ca8baca9e.png)

The third one is using the MKL implementation of BLAS and 4 cores. The thing is that `euclidean_distances` is parallelized through a BLAS LEVEL 3 function but `cdist` only uses a BLAS LEVEL 1 function. Interestingly it almost doesn't change the frontier.
![bench_euclidean_mkl_4](https://user-images.githubusercontent.com/34657725/48774974-f18f0b80-eccb-11e8-925f-2a332891d957.png)


When n_samples is not too low (>100), it seems that the frontier is around 32 features. We could decide to use cdist when n_features < 32 and euclidean_distances when n_features > 32. This is faster and there no precision issue. This also has the advantage that when n_features is small, the julia threshold leads to a lot of re-computations. Using cdist avoids that.

When n_features > 32, we can keep the `euclidean_distances` implementation, updated with the julia threshold. Adding the threshold shouldn't slow `euclidean_distances` too much because the number of features is high enough so that only a few re-computations are necessary.



@jeremiedbb great, thank you for the analysis. The conclusion sounds like a great way forward to me.
Oh, I assume this was all for float64, right? What do we do with float32? upcast always? upcast for >32 features?
I've not read through the comments carefully (will soon), just FYI that float64 has it limitations, see #12128
@qinhanmin2014 yes, float64 precision has limitations, but it is precise enough for producing reliable fp32 results for all I can tell. The question is at which parameters an upcast to fp64 is actually cheaper than using cdist from scipy.
As seen in above benchmarks, even multi-core BLAS is *not* generally faster. This seems to mostly hold for high dimensional data (over 64 dimensions; before that the benefit is usually not worth the effort IMHO) - and since Euclidean distances are not that reliable in dense high dimensional data, that use case IMHO is not of highest importance. Many users will have less than 10 dimensions. In these cases, cdist seems to usually be faster?
> Oh, I assume this was all for float64, right?

Actually it's for both float32 and float64 (I mean very similar). I suggest to always use cdist when n_features < 32.

> The question is at which parameters an upcast to fp64 is actually cheaper than using cdist from scipy.

Upcasting will slowdown by a factor of ~2 so I guess around n_features=64.

> Many users will have less than 10 dimensions. 

But not everyone, so we still need to find a solution for high dimensional data.

Very nice analysis @jeremiedbb !

For low dimensional data it would definitely make sense to use cdist then.

Also, FYI scipy's cdist upcasts float32 to float64 https://github.com/scipy/scipy/issues/8771#issuecomment-384015674, I'm not sure if this is due to accuracy issues or something else. 

Overall, I think it could make sense to add the "algorithm" parameter to `euclidean_distance` as suggested in https://github.com/scikit-learn/scikit-learn/pull/12601#pullrequestreview-176076355, possibly with a default to "None" so that it could also be set via a  global option as in https://github.com/scikit-learn/scikit-learn/pull/12136.
There's also an interesting approach in Eigen3 to compute stable norms: https://eigen.tuxfamily.org/dox/StableNorm_8h_source.html (that I haven't really grokked yet)
Good Explanation, Improved my understanding
We haven't made any progress on this at the sprint and we probably should... and @rth is not around today.
I can join remotely if you set a time. Maybe in the beginning of afternoon?

To summarize the situation,

For precision issues in Euclidean distance calculations,
 - in the low dimensional case, as @jeremiedbb showed above, we should probably use cdist
 - in the high dimensional case and float32, we could choose between,
    - chunking, computing the distance in 64 bit and concatenating
    - falling back to cdist in cases when precision is an issue (how is an open question -- reaching out e.g. to scipy might be useful https://github.com/scikit-learn/scikit-learn/issues/9354#issuecomment-438522881 )

Then there are all the issues of inconsistencies between euclidean, sqeuclidean, minkowski, etc.
In terms of the precisions, @jeremiedbb, @amueller and I had a quick chat, mostly just milking Jeremie for his expertise. He is of the opinion that we don't need to worry so much about the instability issues in an ML context in high dimensions in float64. Jeremie also implied that it is hard to find an efficient test for whether good results have been returned (cf. #12142)

So I think we're happy with @rth's [preceding comment](https://github.com/scikit-learn/scikit-learn/issues/9354#issuecomment-468173901) with the upcasting for float32. Since cdist also upcasts to float64, we could reimplement cdist to take float32 (but with float64 accumulators?), or could use chunking, if we want less copying in low-dim float32.

Does @Celelibi want to change the PR in #11271, or should someone else (one of us?) produce a complete pull request?

And once this has been fixed, I think we should make sqeuclidean and minkowski(p in {0,1}) use our implementations. We've not discussed discrepancy with NearestNeighbors. Another sprint :)
After a quick discussion at the sprint we ended up on the following way:

- in high dimensional case (> 32 or > 64 choose the best): upcast by chunks to float64 when it's float32 and keep the 'fast' method. For this kind of data, numerical issues, on float64, are almost negligible (I'll provide benchmarks for that)

- in low dimensional case: implement the safe computation (instead of using scipy cdist because of the upcast) in sklearn.

(It's tempting to throw upcasting float32 into 0.20.3 also)
Ping when you feel it's ready for review!​

Thanks

@jnothman, Now that all tests pass, I think it's ready for review.
> This is certainly very precise, but perhaps not so elegant! I think it looks pretty good, but I'll have to look again later.

This is indeed not my proudest code. I'm open to any suggestion to refactor the code in addition to the small fix you suggested.

BTW, shall I make a new pull request with the changes in a new branch?
May I modify my branch and force push it?
Or maybe just add new commits to the current branch?

> Could you report some benchmarks here please?

Here you go.

### Optimal memory size
Here are the plots I used to choose the amount of 10MB of temporary memory. It measures the computation time with some various number of samples and features. Distinct X and Y are passed, no squared norm.
![multiplot_memsize](https://user-images.githubusercontent.com/6136274/41529630-0f92c430-72ee-11e8-9dad-c4c3f30498fa.png)
For 100 features, the optimal memory size seems to be about 5MB, but the computation time is quite short. While for 1000 features, it seems to be more between 10MB and 30MB. I thought about computing the optimal amount of memory from the shape of X and Y. But I'm not sure it's worth the added complexity.

Hm. after further investigation, it looks like optimal memory size is the one that produce a block size around 2048. So... maybe I could just add `bs = min(bs, 2048)` so that we get both a maximum of 10MB and a fast block size for small number of features?

### Norm squared precomputing
Here are some plots to see whether it's worth precomputing the norm squared of X and Y. The 3 plots on the left have a fixed number of samples (shown above the plots) and vary the number of features. The 3 plots on the right have a fixed number of features and vary the number of samples.
![multiplot_precompute_full](https://user-images.githubusercontent.com/6136274/41533633-c7a3da66-72fb-11e8-8d7f-159f87d3e4a9.png)
The reason why varying the number of features produce so much variations in the performance might be because it makes the block size vary too.

Let's zoom on the right part of the plots to see whether it's worth precomputing the squared norm.

![multiplot_precompute_zoom](https://user-images.githubusercontent.com/6136274/41533980-0b2499c8-72fd-11e8-9c63-e12ede299753.png)
For a small number of features and samples, it doesn't really matter. But if the number of samples or features is large, precomputing the squared norm of X does have a noticeable impact on the performance. On the other hand, precomputing the squared norm of Y doesn't change anything. It would indeed be computed anyway by the code for float64.

However, a possible improvement not implemented here could be to use some of the allowed additional memory to precompute and cache the norm squared of Y for some blocks (if not all). So that they could be reused during the next iteration over the blocks of X.

### Casting the given norm squared
When both `X_norm_squared` and `Y_norm_squared` are given, is it worth casting them to float64?
![multiplot_cast_zoom](https://user-images.githubusercontent.com/6136274/41535401-75be86f4-7302-11e8-8ce0-9457d0d6980e.png)
It seems pretty clear that it's always worth casting the squared norms when they are given. At least when the numbrer of samples is large enough. Otherwise it doesn't matter.

However, I'm not exactly sure why casting `Y_norm_squared` makes such a difference. It looks like the broadcasting+casting in `distances += YY` is suboptimal.

As before, a possible improvement not implemented could be to cache the casted blocks of the squared norm of `Y_norm_squared` so that they could be reused during the next iteration over the blocks of X.

### Swapping X and Y
Is it worth swapping X and Y when only `X_norm_squared` is given?
Let's plot the time taken when either `X_norm_squared` or `Y_norm_squared` is given and casted to float64, while the other is precomputed.
![multiplot_swap_zoom](https://user-images.githubusercontent.com/6136274/41536751-c4910104-7306-11e8-9c56-793e2f41a648.png)
I think it's pretty clear for a large number of features or samples that X and Y should be swapped when only `X_norm_squared` is given.

Is there any other benchmark you would want to see?

Overall, the gain provided by these optimizations is small, but real and consistent. It's up to you to decide whether it's worth the complexity of the code. ^^
> BTW, shall I make a new pull request with the changes in a new branch?
> May I modify my branch and force push it?
> Or maybe just add new commits to the current branch?

I had to rebase my branch and force push it anyway for the auto-merge to succeed and the tests to pass.

@jnothman wanna have a look at the benchmarks and discuss the mergability of those commits?
yes, this got forgotten, I'll admit.

but I may not find time over the next week. ping if necessary

@rth, you may be interested in this PR, btw.

IMO, we broke euclidean distances for float32 in 0.19 (thinking that
avoiding the upcast was a good idea), and should prioritise fixing it.

Very nice benchmarks and PR @Celelibi !

It would be useful to also test the net effect of this PR  on performance e.g. of KMeans / Birch as suggested in https://github.com/scikit-learn/scikit-learn/pull/10069#issuecomment-342347548 

I'm not yet  sure how this would interact with `pairwise_distances_chunked(.., metric="euclidean")` -- I'm wondering if could be possible to reuse some of that work, or at least make sure we don't chunk twice in this case. In any case it might make sense to use `with sklearn.config_context(working_memory=128):` context manager to defined the amount of memory per block.
> I'm not yet sure how this would interact with pairwise_distances_chunked(.., metric="euclidean")

I didn't forsee this. Well, they might chunk twice, which may have a bad impact on performance.

> I'm wondering if could be possible to reuse some of that work, or at least make sure we don't chunk twice in this case.

It wouldn't be easy to have the chunking done at only one place in the code. I mean the current code always use `Y` as a whole. Which means it should be casted entirely. Even if we fixed it to chunk both `X` and `Y`, the chunking code of `pairwise_distances_chunked` isn't really meant to be intermixed with some kind of preprocessing (nor should it IMO).

The best solution I could see right now would be to have some specialized chunked implementations for some metrics. Kind of the same way `pairwise_distance` only rely on `scipy.distance.{p,c}dist` when there isn't a better implementation.
What do you think about it?

BTW `pairwise_distances` might currently use `scipy.spatial.{c,p}dist`, which (in addition to being slow) handle float32 by casting first and returning a float64 result. This might be a problem with `sqeuclidean` metric which then behave differently from `euclidean` in that regard in addition to being a problem with the general support of float32.

> In any case it might make sense to use `with sklearn.config_context(working_memory=128):` context manager to defined the amount of memory per block.

Interesting, I didn't know about that. However, it looks like `utils.get_chunk_n_rows` is the only function to ever use that setting. Unfortunately I can't use that function since I have to _at least_ take into account the casted copy of `X` and the result chunk. But I could still use the amount of working memory that is set instead of a magic value.

> It would be useful to also test the net effect of this PR on performance e.g. of KMeans / Birch as suggested in #10069 (comment)

That comment was about deprecating `{X,Y}_norm_squared` and its impact on performance on the algorithms using it. But ok, why not. I haven't made a benchmark comparing the older code.
I think given that we seem to see that this operation works well with 10MB
working mem and the default working memory is 1GB, we should consider this
a negligible addition, and not use the working_memory business with it.​

I also think it's important to focus upon this as a bug fix, rather than
something that needs to be perfected in one shot.

@Celelibi Thanks for the detailed response!

> I think given that we seem to see that this operation works well with 10MB
working mem and the default working memory is 1GB we should consider this
a negligible addition, and not use the working_memory business with it.​


@jeremiedbb, @ogrisel mentioned that you run some benchmarks demonstrating that using a smaller working memory had higher performance on your system. Would you be able to share those results (the original benchmarks are in https://github.com/scikit-learn/scikit-learn/pull/10280#issuecomment-356419843)? Maybe in a separate issue. Thanks!
from my understanding this will take some more work. untagging 0.20.
Yes, it's pretty bad. Integrated these tests into https://github.com/scikit-learn/scikit-learn/pull/12142

Also (for other readers) most of the discussion about this is happening in https://github.com/scikit-learn/scikit-learn/issues/9354
```

## Patch

```diff
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -193,6 +193,7 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     Y_norm_squared : array-like, shape (n_samples_2, ), optional
         Pre-computed dot-products of vectors in Y (e.g.,
         ``(Y**2).sum(axis=1)``)
+        May be ignored in some cases, see the note below.
 
     squared : boolean, optional
         Return squared Euclidean distances.
@@ -200,10 +201,16 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     X_norm_squared : array-like, shape = [n_samples_1], optional
         Pre-computed dot-products of vectors in X (e.g.,
         ``(X**2).sum(axis=1)``)
+        May be ignored in some cases, see the note below.
+
+    Notes
+    -----
+    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
+    unused if they are passed as ``float32``.
 
     Returns
     -------
-    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)
+    distances : array, shape (n_samples_1, n_samples_2)
 
     Examples
     --------
@@ -224,6 +231,9 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     """
     X, Y = check_pairwise_arrays(X, Y)
 
+    # If norms are passed as float32, they are unused. If arrays are passed as
+    # float32, norms needs to be recomputed on upcast chunks.
+    # TODO: use a float64 accumulator in row_norms to avoid the latter.
     if X_norm_squared is not None:
         XX = check_array(X_norm_squared)
         if XX.shape == (1, X.shape[0]):
@@ -231,10 +241,15 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
         elif XX.shape != (X.shape[0], 1):
             raise ValueError(
                 "Incompatible dimensions for X and X_norm_squared")
+        if XX.dtype == np.float32:
+            XX = None
+    elif X.dtype == np.float32:
+        XX = None
     else:
         XX = row_norms(X, squared=True)[:, np.newaxis]
 
-    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
+    if X is Y and XX is not None:
+        # shortcut in the common case euclidean_distances(X, X)
         YY = XX.T
     elif Y_norm_squared is not None:
         YY = np.atleast_2d(Y_norm_squared)
@@ -242,23 +257,99 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
         if YY.shape != (1, Y.shape[0]):
             raise ValueError(
                 "Incompatible dimensions for Y and Y_norm_squared")
+        if YY.dtype == np.float32:
+            YY = None
+    elif Y.dtype == np.float32:
+        YY = None
     else:
         YY = row_norms(Y, squared=True)[np.newaxis, :]
 
-    distances = safe_sparse_dot(X, Y.T, dense_output=True)
-    distances *= -2
-    distances += XX
-    distances += YY
+    if X.dtype == np.float32:
+        # To minimize precision issues with float32, we compute the distance
+        # matrix on chunks of X and Y upcast to float64
+        distances = _euclidean_distances_upcast(X, XX, Y, YY)
+    else:
+        # if dtype is already float64, no need to chunk and upcast
+        distances = - 2 * safe_sparse_dot(X, Y.T, dense_output=True)
+        distances += XX
+        distances += YY
     np.maximum(distances, 0, out=distances)
 
+    # Ensure that distances between vectors and themselves are set to 0.0.
+    # This may not be the case due to floating point rounding errors.
     if X is Y:
-        # Ensure that distances between vectors and themselves are set to 0.0.
-        # This may not be the case due to floating point rounding errors.
-        distances.flat[::distances.shape[0] + 1] = 0.0
+        np.fill_diagonal(distances, 0)
 
     return distances if squared else np.sqrt(distances, out=distances)
 
 
+def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
+    """Euclidean distances between X and Y
+
+    Assumes X and Y have float32 dtype.
+    Assumes XX and YY have float64 dtype or are None.
+
+    X and Y are upcast to float64 by chunks, which size is chosen to limit
+    memory increase by approximately 10% (at least 10MiB).
+    """
+    n_samples_X = X.shape[0]
+    n_samples_Y = Y.shape[0]
+    n_features = X.shape[1]
+
+    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)
+
+    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
+    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
+
+    # Allow 10% more memory than X, Y and the distance matrix take (at least
+    # 10MiB)
+    maxmem = max(
+        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
+         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
+        10 * 2**17)
+
+    # The increase amount of memory in 8-byte blocks is:
+    # - x_density * batch_size * n_features (copy of chunk of X)
+    # - y_density * batch_size * n_features (copy of chunk of Y)
+    # - batch_size * batch_size (chunk of distance matrix)
+    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
+    #                                 xd=x_density and yd=y_density
+    tmp = (x_density + y_density) * n_features
+    batch_size = (-tmp + np.sqrt(tmp**2 + 4 * maxmem)) / 2
+    batch_size = max(int(batch_size), 1)
+
+    x_batches = gen_batches(X.shape[0], batch_size)
+    y_batches = gen_batches(Y.shape[0], batch_size)
+
+    for i, x_slice in enumerate(x_batches):
+        X_chunk = X[x_slice].astype(np.float64)
+        if XX is None:
+            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
+        else:
+            XX_chunk = XX[x_slice]
+
+        for j, y_slice in enumerate(y_batches):
+            if X is Y and j < i:
+                # when X is Y the distance matrix is symmetric so we only need
+                # to compute half of it.
+                d = distances[y_slice, x_slice].T
+
+            else:
+                Y_chunk = Y[y_slice].astype(np.float64)
+                if YY is None:
+                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
+                else:
+                    YY_chunk = YY[:, y_slice]
+
+                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
+                d += XX_chunk
+                d += YY_chunk
+
+            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)
+
+    return distances
+
+
 def _argmin_min_reduce(dist, start):
     indices = dist.argmin(axis=1)
     values = dist[np.arange(dist.shape[0]), indices]

```

## Test Patch

```diff
diff --git a/sklearn/metrics/tests/test_pairwise.py b/sklearn/metrics/tests/test_pairwise.py
--- a/sklearn/metrics/tests/test_pairwise.py
+++ b/sklearn/metrics/tests/test_pairwise.py
@@ -584,41 +584,115 @@ def test_pairwise_distances_chunked():
     assert_raises(StopIteration, next, gen)
 
 
-def test_euclidean_distances():
-    # Check the pairwise Euclidean distances computation
-    X = [[0]]
-    Y = [[1], [2]]
+@pytest.mark.parametrize("x_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+@pytest.mark.parametrize("y_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances_known_result(x_array_constr, y_array_constr):
+    # Check the pairwise Euclidean distances computation on known result
+    X = x_array_constr([[0]])
+    Y = y_array_constr([[1], [2]])
     D = euclidean_distances(X, Y)
-    assert_array_almost_equal(D, [[1., 2.]])
+    assert_allclose(D, [[1., 2.]])
 
-    X = csr_matrix(X)
-    Y = csr_matrix(Y)
-    D = euclidean_distances(X, Y)
-    assert_array_almost_equal(D, [[1., 2.]])
 
+@pytest.mark.parametrize("dtype", [np.float32, np.float64])
+@pytest.mark.parametrize("y_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances_with_norms(dtype, y_array_constr):
+    # check that we still get the right answers with {X,Y}_norm_squared
+    # and that we get a wrong answer with wrong {X,Y}_norm_squared
     rng = np.random.RandomState(0)
-    X = rng.random_sample((10, 4))
-    Y = rng.random_sample((20, 4))
-    X_norm_sq = (X ** 2).sum(axis=1).reshape(1, -1)
-    Y_norm_sq = (Y ** 2).sum(axis=1).reshape(1, -1)
+    X = rng.random_sample((10, 10)).astype(dtype, copy=False)
+    Y = rng.random_sample((20, 10)).astype(dtype, copy=False)
+
+    # norms will only be used if their dtype is float64
+    X_norm_sq = (X.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
+    Y_norm_sq = (Y.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
+
+    Y = y_array_constr(Y)
 
-    # check that we still get the right answers with {X,Y}_norm_squared
     D1 = euclidean_distances(X, Y)
     D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
     D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
     D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq,
                              Y_norm_squared=Y_norm_sq)
-    assert_array_almost_equal(D2, D1)
-    assert_array_almost_equal(D3, D1)
-    assert_array_almost_equal(D4, D1)
+    assert_allclose(D2, D1)
+    assert_allclose(D3, D1)
+    assert_allclose(D4, D1)
 
     # check we get the wrong answer with wrong {X,Y}_norm_squared
-    X_norm_sq *= 0.5
-    Y_norm_sq *= 0.5
     wrong_D = euclidean_distances(X, Y,
                                   X_norm_squared=np.zeros_like(X_norm_sq),
                                   Y_norm_squared=np.zeros_like(Y_norm_sq))
-    assert_greater(np.max(np.abs(wrong_D - D1)), .01)
+    with pytest.raises(AssertionError):
+        assert_allclose(wrong_D, D1)
+
+
+@pytest.mark.parametrize("dtype", [np.float32, np.float64])
+@pytest.mark.parametrize("x_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+@pytest.mark.parametrize("y_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances(dtype, x_array_constr, y_array_constr):
+    # check that euclidean distances gives same result as scipy cdist
+    # when X and Y != X are provided
+    rng = np.random.RandomState(0)
+    X = rng.random_sample((100, 10)).astype(dtype, copy=False)
+    X[X < 0.8] = 0
+    Y = rng.random_sample((10, 10)).astype(dtype, copy=False)
+    Y[Y < 0.8] = 0
+
+    expected = cdist(X, Y)
+
+    X = x_array_constr(X)
+    Y = y_array_constr(Y)
+    distances = euclidean_distances(X, Y)
+
+    # the default rtol=1e-7 is too close to the float32 precision
+    # and fails due too rounding errors.
+    assert_allclose(distances, expected, rtol=1e-6)
+    assert distances.dtype == dtype
+
+
+@pytest.mark.parametrize("dtype", [np.float32, np.float64])
+@pytest.mark.parametrize("x_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances_sym(dtype, x_array_constr):
+    # check that euclidean distances gives same result as scipy pdist
+    # when only X is provided
+    rng = np.random.RandomState(0)
+    X = rng.random_sample((100, 10)).astype(dtype, copy=False)
+    X[X < 0.8] = 0
+
+    expected = squareform(pdist(X))
+
+    X = x_array_constr(X)
+    distances = euclidean_distances(X)
+
+    # the default rtol=1e-7 is too close to the float32 precision
+    # and fails due too rounding errors.
+    assert_allclose(distances, expected, rtol=1e-6)
+    assert distances.dtype == dtype
+
+
+@pytest.mark.parametrize(
+    "dtype, eps, rtol",
+    [(np.float32, 1e-4, 1e-5),
+     pytest.param(
+         np.float64, 1e-8, 0.99,
+         marks=pytest.mark.xfail(reason='failing due to lack of precision'))])
+@pytest.mark.parametrize("dim", [1, 1000000])
+def test_euclidean_distances_extreme_values(dtype, eps, rtol, dim):
+    # check that euclidean distances is correct with float32 input thanks to
+    # upcasting. On float64 there are still precision issues.
+    X = np.array([[1.] * dim], dtype=dtype)
+    Y = np.array([[1. + eps] * dim], dtype=dtype)
+
+    distances = euclidean_distances(X, Y)
+    expected = cdist(X, Y)
+
+    assert_allclose(distances, expected, rtol=1e-5)
 
 
 def test_cosine_distances():

```


## Code snippets

### 1 - sklearn/metrics/pairwise.py:

Start line: 164, End line: 259

```python
# Pairwise distances
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    This formulation has two advantages over other ways of computing distances.
    First, it is computationally efficient when dealing with sparse data.
    Second, if one argument varies but the other remains unchanged, then
    `dot(x, x)` and/or `dot(y, y)` can be pre-computed.

    However, this is not the most precise way of doing this computation, and
    the distance matrix returned by this function may not be exactly
    symmetric as required by, e.g., ``scipy.spatial.distance`` functions.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_1, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_2, n_features)

    Y_norm_squared : array-like, shape (n_samples_2, ), optional
        Pre-computed dot-products of vectors in Y (e.g.,
        ``(Y**2).sum(axis=1)``)

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)

    Returns
    -------
    distances : {array, sparse matrix}, shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from sklearn.metrics.pairwise import euclidean_distances
    >>> X = [[0, 1], [1, 1]]
    >>> # distance between rows of X
    >>> euclidean_distances(X, X)
    array([[0., 1.],
           [1., 0.]])
    >>> # get distance to origin
    >>> euclidean_distances(X, [[0, 0]])
    array([[1.        ],
           [1.41421356]])

    See also
    --------
    paired_distances : distances betweens pairs of elements of X and Y.
    """
    X, Y = check_pairwise_arrays(X, Y)

    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if X is Y:  # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    distances = safe_sparse_dot(X, Y.T, dense_output=True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0

    return distances if squared else np.sqrt(distances, out=distances)
```
### 2 - sklearn/metrics/pairwise.py:

Start line: 1202, End line: 1318

```python
def pairwise_distances_chunked(X, Y=None, reduce_func=None,
                               metric='euclidean', n_jobs=None,
                               working_memory=None, **kwds):
    """Generate a distance matrix chunk by chunk with optional reduction

    In cases where not all of a pairwise distance matrix needs to be stored at
    once, this is used to calculate pairwise distances in
    ``working_memory``-sized chunks.  If ``reduce_func`` is given, it is run
    on each chunk and its return values are concatenated into lists, arrays
    or sparse matrices.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or,
        [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".

    reduce_func : callable, optional
        The function which is applied on each chunk of the distance matrix,
        reducing it to needed values.  ``reduce_func(D_chunk, start)``
        is called repeatedly, where ``D_chunk`` is a contiguous vertical
        slice of the pairwise distance matrix, starting at row ``start``.
        It should return an array, a list, or a sparse matrix of length
        ``D_chunk.shape[0]``, or a tuple of such objects.

        If None, pairwise_distances_chunked returns a generator of vertical
        chunks of the distance matrix.

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    working_memory : int, optional
        The sought maximum memory for temporary distance matrix chunks.
        When None (default), the value of
        ``sklearn.get_config()['working_memory']`` is used.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Yields
    ------
    D_chunk : array or sparse matrix
        A contiguous slice of distance matrix, optionally processed by
        ``reduce_func``.

    Examples
    --------
    Without reduce_func:

    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances_chunked
    >>> X = np.random.RandomState(0).rand(5, 3)
    >>> D_chunk = next(pairwise_distances_chunked(X))
    >>> D_chunk  # doctest: +ELLIPSIS
    array([[0.  ..., 0.29..., 0.41..., 0.19..., 0.57...],
           [0.29..., 0.  ..., 0.57..., 0.41..., 0.76...],
           [0.41..., 0.57..., 0.  ..., 0.44..., 0.90...],
           [0.19..., 0.41..., 0.44..., 0.  ..., 0.51...],
           [0.57..., 0.76..., 0.90..., 0.51..., 0.  ...]])

    Retrieve all neighbors and average distance within radius r:

    >>> r = .2
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r) for d in D_chunk]
    ...     avg_dist = (D_chunk * (D_chunk < r)).mean(axis=1)
    ...     return neigh, avg_dist
    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func)
    >>> neigh, avg_dist = next(gen)
    >>> neigh
    [array([0, 3]), array([1]), array([2]), array([0, 3]), array([4])]
    >>> avg_dist  # doctest: +ELLIPSIS
    array([0.039..., 0.        , 0.        , 0.039..., 0.        ])

    Where r is defined per sample, we need to make use of ``start``:

    >>> r = [.2, .4, .4, .3, .1]
    >>> def reduce_func(D_chunk, start):
    ...     neigh = [np.flatnonzero(d < r[i])
    ...              for i, d in enumerate(D_chunk, start)]
    ...     return neigh
    >>> neigh = next(pairwise_distances_chunked(X, reduce_func=reduce_func))
    >>> neigh
    [array([0, 3]), array([0, 1]), array([2]), array([0, 3]), array([4])]

    Force row-by-row generation by reducing ``working_memory``:

    >>> gen = pairwise_distances_chunked(X, reduce_func=reduce_func,
    ...                                  working_memory=0)
    >>> next(gen)
    [array([0, 3])]
    >>> next(gen)
    [array([0, 1])]
    """
    # ... other code
```
### 3 - sklearn/metrics/pairwise.py:

Start line: 1319, End line: 1360

```python
def pairwise_distances_chunked(X, Y=None, reduce_func=None,
                               metric='euclidean', n_jobs=None,
                               working_memory=None, **kwds):
    n_samples_X = _num_samples(X)
    if metric == 'precomputed':
        slices = (slice(0, n_samples_X),)
    else:
        if Y is None:
            Y = X
        # We get as many rows as possible within our working_memory budget to
        # store len(Y) distances in each row of output.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of distances will
        #    exceed working_memory.
        #  - this does not account for any temporary memory usage while
        #    calculating distances (e.g. difference of vectors in manhattan
        #    distance.
        chunk_n_rows = get_chunk_n_rows(row_bytes=8 * _num_samples(Y),
                                        max_n_rows=n_samples_X,
                                        working_memory=working_memory)
        slices = gen_batches(n_samples_X, chunk_n_rows)

    # precompute data-derived metric params
    params = _precompute_metric_params(X, Y, metric=metric, **kwds)
    kwds.update(**params)

    for sl in slices:
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        D_chunk = pairwise_distances(X_chunk, Y, metric=metric,
                                     n_jobs=n_jobs, **kwds)
        if ((X is Y or Y is None)
                and PAIRWISE_DISTANCE_FUNCTIONS.get(metric, None)
                is euclidean_distances):
            # zeroing diagonal, taking care of aliases of "euclidean",
            # i.e. "l2"
            D_chunk.flat[sl.start::_num_samples(X) + 1] = 0
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        yield D_chunk
```
### 4 - sklearn/metrics/pairwise.py:

Start line: 1449, End line: 1488

```python
def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=None, **kwds):
    if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
        raise ValueError("Unknown metric %s. "
                         "Valid metrics are %s, or 'precomputed', or a "
                         "callable" % (metric, _VALID_METRICS))

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)

        whom = ("`pairwise_distances`. Precomputed distance "
                " need to have non-negative values.")
        check_non_negative(X, whom=whom)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not"
                            " support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else None

        if dtype == bool and (X.dtype != bool or Y.dtype != bool):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        X, Y = check_pairwise_arrays(X, Y, dtype=dtype)

        # precompute data-derived metric params
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric,
                                                      **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
```
### 5 - sklearn/metrics/pairwise.py:

Start line: 599, End line: 617

```python
# Paired distances
def paired_euclidean_distances(X, Y):
    """
    Computes the paired euclidean distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    X, Y = check_paired_arrays(X, Y)
    return row_norms(X - Y)
```
### 6 - sklearn/metrics/pairwise.py:

Start line: 262, End line: 361

```python
def _argmin_min_reduce(dist, start):
    indices = dist.argmin(axis=1)
    values = dist[np.arange(dist.shape[0]), indices]
    return indices, values


def pairwise_distances_argmin_min(X, Y, axis=1, metric="euclidean",
                                  batch_size=None, metric_kwargs=None):
    """Compute minimum distances between one point and a set of points.

    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance). The minimal distances are
    also returned.

    This is mostly equivalent to calling:

        (pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis),
         pairwise_distances(X, Y=Y, metric=metric).min(axis=axis))

    but uses much less memory, and is faster for large arrays.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples1, n_features)
        Array containing points.

    Y : {array-like, sparse matrix}, shape (n_samples2, n_features)
        Arrays containing points.

    axis : int, optional, default 1
        Axis along which the argmin and distances are to be computed.

    metric : string or callable, default 'euclidean'
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

    batch_size : integer
        .. deprecated:: 0.20
            Deprecated for removal in 0.22.
            Use sklearn.set_config(working_memory=...) instead.

    metric_kwargs : dict, optional
        Keyword arguments to pass to specified metric function.

    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].

    distances : numpy.ndarray
        distances[i] is the distance between the i-th row in X and the
        argmin[i]-th row in Y.

    See also
    --------
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin
    """
    if batch_size is not None:
        warnings.warn("'batch_size' is ignored. It was deprecated in version "
                      "0.20 and will be removed in version 0.22. "
                      "Use sklearn.set_config(working_memory=...) instead.",
                      DeprecationWarning)
    X, Y = check_pairwise_arrays(X, Y)

    if metric_kwargs is None:
        metric_kwargs = {}

    if axis == 0:
        X, Y = Y, X

    indices, values = zip(*pairwise_distances_chunked(
        X, Y, reduce_func=_argmin_min_reduce, metric=metric,
        **metric_kwargs))
    indices = np.concatenate(indices)
    values = np.concatenate(values)

    return indices, values
```
### 7 - sklearn/metrics/pairwise.py:

Start line: 364, End line: 441

```python
def pairwise_distances_argmin(X, Y, axis=1, metric="euclidean",
                              batch_size=None, metric_kwargs=None):
    """Compute minimum distances between one point and a set of points.

    This function computes for each row in X, the index of the row of Y which
    is closest (according to the specified distance).

    This is mostly equivalent to calling:

        pairwise_distances(X, Y=Y, metric=metric).argmin(axis=axis)

    but uses much less memory, and is faster for large arrays.

    This function works with dense 2D arrays only.

    Parameters
    ----------
    X : array-like
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)

    Y : array-like
        Arrays containing points. Respective shapes (n_samples1, n_features)
        and (n_samples2, n_features)

    axis : int, optional, default 1
        Axis along which the argmin and distances are to be computed.

    metric : string or callable
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

    batch_size : integer
        .. deprecated:: 0.20
            Deprecated for removal in 0.22.
            Use sklearn.set_config(working_memory=...) instead.

    metric_kwargs : dict
        keyword arguments to pass to specified metric function.

    Returns
    -------
    argmin : numpy.ndarray
        Y[argmin[i], :] is the row in Y that is closest to X[i, :].

    See also
    --------
    sklearn.metrics.pairwise_distances
    sklearn.metrics.pairwise_distances_argmin_min
    """
    if metric_kwargs is None:
        metric_kwargs = {}

    return pairwise_distances_argmin_min(X, Y, axis, metric,
                                         metric_kwargs=metric_kwargs,
                                         batch_size=batch_size)[0]
```
### 8 - sklearn/metrics/pairwise.py:

Start line: 1099, End line: 1121

```python
def _dist_wrapper(dist_func, dist_matrix, slice_, *args, **kwargs):
    """Write in-place to a slice of a distance matrix"""
    dist_matrix[:, slice_] = dist_func(*args, **kwargs)


def _parallel_pairwise(X, Y, func, n_jobs, **kwds):
    """Break the pairwise matrix in n_jobs even slices
    and compute them in parallel"""

    if Y is None:
        Y = X

    if effective_n_jobs(n_jobs) == 1:
        return func(X, Y, **kwds)

    # enforce a threading backend to prevent data communication overhead
    fd = delayed(_dist_wrapper)
    ret = np.empty((X.shape[0], Y.shape[0]), dtype=X.dtype, order='F')
    Parallel(backend="threading", n_jobs=n_jobs)(
        fd(func, ret, s, X, Y[s], **kwds)
        for s in gen_even_slices(_num_samples(Y), effective_n_jobs(n_jobs)))

    return ret
```
### 9 - sklearn/metrics/pairwise.py:

Start line: 36, End line: 58

```python
# Utility Functions
def _return_float_dtype(X, Y):
    """
    1. If dtype of X and Y is float32, then dtype float32 is returned.
    2. Else dtype float is returned.
    """
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float

    return X, Y, dtype
```
### 10 - sklearn/metrics/pairwise.py:

Start line: 1491, End line: 1517

```python
# These distances recquire boolean arrays, when using scipy.spatial.distance
PAIRWISE_BOOLEAN_FUNCTIONS = [
    'dice',
    'jaccard',
    'kulsinski',
    'matching',
    'rogerstanimoto',
    'russellrao',
    'sokalmichener',
    'sokalsneath',
    'yule',
]


# Helper functions - distance
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'additive_chi2': additive_chi2_kernel,
    'chi2': chi2_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'laplacian': laplacian_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity, }
```
### 11 - sklearn/metrics/pairwise.py:

Start line: 61, End line: 127

```python
def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """ Set X and Y appropriately and checks inputs

    If Y is None, it is set as a pointer to X (i.e. not a copy).
    If Y is given, this does not happen.
    All distance metrics should use this function first to assert that the
    given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats (or dtype if provided). Finally, the function
    checks that the size of the second dimension of the two arrays is equal, or
    the equivalent check for a precomputed distance matrix.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    precomputed : bool
        True if X is to be treated as precomputed distances to the samples in
        Y.

    dtype : string, type, list of types or None (default=None)
        Data type required for X and Y. If None, the dtype will be an
        appropriate float type selected by _return_float_dtype.

        .. versionadded:: 0.18

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y, dtype_float = _return_float_dtype(X, Y)

    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype,
                            estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype,
                        estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype,
                        estimator=estimator)

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError("Precomputed metric requires shape "
                             "(n_queries, n_indexed). Got (%d, %d) "
                             "for %d indexed." %
                             (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices: "
                         "X.shape[1] == %d while Y.shape[1] == %d" % (
                             X.shape[1], Y.shape[1]))

    return X, Y
```
### 12 - sklearn/metrics/pairwise.py:

Start line: 1057, End line: 1069

```python
# Helper functions - distance
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cityblock': manhattan_distances,
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'haversine': haversine_distances,
    'l2': euclidean_distances,
    'l1': manhattan_distances,
    'manhattan': manhattan_distances,
    'precomputed': None,  # HACK: precomputed is always allowed, never called
}
```
### 16 - sklearn/metrics/pairwise.py:

Start line: 678, End line: 732

```python
def paired_distances(X, Y, metric="euclidean", **kwds):
    """
    Computes the paired distances between X and Y.

    Computes the distances between (X[0], Y[0]), (X[1], Y[1]), etc...

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        Array 1 for distance computation.

    Y : ndarray (n_samples, n_features)
        Array 2 for distance computation.

    metric : string or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        specified in PAIRED_DISTANCES, including "euclidean",
        "manhattan", or "cosine".
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    Returns
    -------
    distances : ndarray (n_samples, )

    Examples
    --------
    >>> from sklearn.metrics.pairwise import paired_distances
    >>> X = [[0, 1], [1, 1]]
    >>> Y = [[0, 1], [2, 1]]
    >>> paired_distances(X, Y)
    array([0., 1.])

    See also
    --------
    pairwise_distances : Computes the distance between every pair of samples
    """

    if metric in PAIRED_DISTANCES:
        func = PAIRED_DISTANCES[metric]
        return func(X, Y)
    elif callable(metric):
        # Check the matrix first (it is usually done by the metric)
        X, Y = check_paired_arrays(X, Y)
        distances = np.zeros(len(X))
        for i in range(len(X)):
            distances[i] = metric(X[i], Y[i])
        return distances
    else:
        raise ValueError('Unknown distance %s' % metric)
```
### 17 - sklearn/metrics/pairwise.py:

Start line: 644, End line: 675

```python
def paired_cosine_distances(X, Y):
    """
    Computes the paired cosine distances between X and Y

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray, shape (n_samples, )

    Notes
    ------
    The cosine distance is equivalent to the half the squared
    euclidean distance if each sample is normalized to unit norm
    """
    X, Y = check_paired_arrays(X, Y)
    return .5 * row_norms(normalize(X) - normalize(Y), squared=True)


PAIRED_DISTANCES = {
    'cosine': paired_cosine_distances,
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
    'l1': paired_manhattan_distances,
    'manhattan': paired_manhattan_distances,
    'cityblock': paired_manhattan_distances}
```
### 18 - sklearn/metrics/pairwise.py:

Start line: 1363, End line: 1448

```python
def pairwise_distances(X, Y=None, metric="euclidean", n_jobs=None, **kwds):
    """ Compute the distance matrix from a vector array X and optional Y.

    This method takes either a vector array or a distance matrix, and returns
    a distance matrix. If the input is a vector array, the distances are
    computed. If the input is a distances matrix, it is returned instead.

    This method provides a safe way to take a distance matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.

    If Y is given (default is None), then the returned matrix is the pairwise
    distance between the arrays from both X and Y.

    Valid values for metric are:

    - From scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']. These metrics support sparse matrix inputs.

    - From scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
      'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
      'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
      See the documentation for scipy.spatial.distance for details on these
      metrics. These metrics do not support sparse matrix inputs.

    Note that in the case of 'cityblock', 'cosine' and 'euclidean' (which are
    valid scipy.spatial.distance metrics), the scikit-learn implementation
    will be used, which is faster and has support for sparse matrices (except
    for 'cityblock'). For a verbose description of the metrics from
    scikit-learn, see the __doc__ of the sklearn.pairwise.distance_metrics
    function.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.

    Y : array [n_samples_b, n_features], optional
        An optional second feature array. Only allowed if
        metric != "precomputed".

    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]
        A distance matrix D such that D_{i, j} is the distance between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then D_{i, j} is the distance between the ith array
        from X and the jth array from Y.

    See also
    --------
    pairwise_distances_chunked : performs the same calculation as this
        function, but returns a generator of chunks of the distance matrix, in
        order to limit memory usage.
    paired_distances : Computes the distances between corresponding
                       elements of two arrays
    """
    # ... other code
```
### 19 - sklearn/metrics/pairwise.py:

Start line: 489, End line: 559

```python
def manhattan_distances(X, Y=None, sum_over_features=True):
    """ Compute the L1 distances between the vectors in X and Y.

    With sum_over_features equal to False it returns the componentwise
    distances.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like
        An array with shape (n_samples_X, n_features).

    Y : array_like, optional
        An array with shape (n_samples_Y, n_features).

    sum_over_features : bool, default=True
        If True the function returns the pairwise distance matrix
        else it returns the componentwise L1 pairwise-distances.
        Not supported for sparse matrix inputs.

    Returns
    -------
    D : array
        If sum_over_features is False shape is
        (n_samples_X * n_samples_Y, n_features) and D contains the
        componentwise L1 pairwise-distances (ie. absolute difference),
        else shape is (n_samples_X, n_samples_Y) and D contains
        the pairwise L1 distances.

    Examples
    --------
    >>> from sklearn.metrics.pairwise import manhattan_distances
    >>> manhattan_distances([[3]], [[3]])#doctest:+ELLIPSIS
    array([[0.]])
    >>> manhattan_distances([[3]], [[2]])#doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[2]], [[3]])#doctest:+ELLIPSIS
    array([[1.]])
    >>> manhattan_distances([[1, 2], [3, 4]],\
         [[1, 2], [0, 3]])#doctest:+ELLIPSIS
    array([[0., 2.],
           [4., 4.]])
    >>> import numpy as np
    >>> X = np.ones((1, 2))
    >>> y = np.full((2, 2), 2.)
    >>> manhattan_distances(X, y, sum_over_features=False)#doctest:+ELLIPSIS
    array([[1., 1.],
           [1., 1.]])
    """
    X, Y = check_pairwise_arrays(X, Y)

    if issparse(X) or issparse(Y):
        if not sum_over_features:
            raise TypeError("sum_over_features=%r not supported"
                            " for sparse matrices" % sum_over_features)

        X = csr_matrix(X, copy=False)
        Y = csr_matrix(Y, copy=False)
        D = np.zeros((X.shape[0], Y.shape[0]))
        _sparse_manhattan(X.data, X.indices, X.indptr,
                          Y.data, Y.indices, Y.indptr,
                          X.shape[1], D)
        return D

    if sum_over_features:
        return distance.cdist(X, Y, 'cityblock')

    D = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    D = np.abs(D, D)
    return D.reshape((-1, X.shape[1]))
```
### 21 - sklearn/metrics/pairwise.py:

Start line: 1, End line: 33

```python
# -*- coding: utf-8 -*-

import itertools
from functools import partial
import warnings

import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse

from ..utils.validation import _num_samples
from ..utils.validation import check_non_negative
from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..utils._joblib import Parallel
from ..utils._joblib import delayed
from ..utils._joblib import effective_n_jobs

from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
from ..exceptions import DataConversionWarning
```
### 22 - sklearn/metrics/pairwise.py:

Start line: 444, End line: 486

```python
def haversine_distances(X, Y=None):
    """Compute the Haversine distance between samples in X and Y

    The Haversine (or great circle) distance is the angular distance between
    two points on the surface of a sphere. The first distance of each point is
    assumed to be the latitude, the second is the longitude, given in radians.
    The dimension of the data must be 2.

    .. math::
       D(x, y) = 2\arcsin[\\sqrt{\\sin^2((x1 - y1) / 2)
                                + cos(x1)cos(y1)sin^2((x2 - y2) / 2)}]

    Parameters
    ----------
    X : array_like, shape (n_samples_1, 2)

    Y : array_like, shape (n_samples_2, 2), optional

    Returns
    -------
    distance : {array}, shape (n_samples_1, n_samples_2)

    Notes
    -----
    As the Earth is nearly spherical, the haversine formula provides a good
    approximation of the distance between two points of the Earth surface, with
    a less than 1% error on average.

    Examples
    --------
    We want to calculate the distance between the Ezeiza Airport
    (Buenos Aires, Argentina) and the Charles de Gaulle Airport (Paris, France)

    >>> from sklearn.metrics.pairwise import haversine_distances
    >>> bsas = [-34.83333, -58.5166646]
    >>> paris = [49.0083899664, 2.53844117956]
    >>> result = haversine_distances([bsas, paris])
    >>> result * 6371000/1000  # multiply by Earth radius to get kilometers
    array([[    0.        , 11279.45379464],
           [11279.45379464,     0.        ]])
    """
    from sklearn.neighbors import DistanceMetric
    return DistanceMetric.get_metric('haversine').pairwise(X, Y)
```
### 23 - sklearn/metrics/pairwise.py:

Start line: 130, End line: 161

```python
def check_paired_arrays(X, Y):
    """ Set X and Y appropriately and checks inputs for paired distances

    All paired distance metrics should use this function first to assert that
    the given parameters are correct and safe to use.

    Specifically, this function first ensures that both X and Y are arrays,
    then checks that they are at least two dimensional while ensuring that
    their elements are floats. Finally, the function checks that the size
    of the dimensions of the two arrays are equal.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples_a, n_features)

    Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_samples_a, n_features)
        An array equal to X, guaranteed to be a numpy array.

    safe_Y : {array-like, sparse matrix}, shape (n_samples_b, n_features)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.

    """
    X, Y = check_pairwise_arrays(X, Y)
    if X.shape != Y.shape:
        raise ValueError("X and Y should be of same shape. They were "
                         "respectively %r and %r long." % (X.shape, Y.shape))
    return X, Y
```
### 28 - sklearn/metrics/pairwise.py:

Start line: 1124, End line: 1153

```python
def _pairwise_callable(X, Y, metric, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}
    """
    X, Y = check_pairwise_arrays(X, Y)

    if X is Y:
        # Only calculate metric for upper triangle
        out = np.zeros((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

        # Make symmetric
        # NB: out += out.T will produce incorrect results
        out = out + out.T

        # Calculate diagonal
        # NB: nonzero diagonals are allowed for both metrics and kernels
        for i in range(X.shape[0]):
            x = X[i]
            out[i, i] = metric(x, x, **kwds)

    else:
        # Calculate all cells
        out = np.empty((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            out[i, j] = metric(X[i], Y[j], **kwds)

    return out
```
### 31 - sklearn/metrics/pairwise.py:

Start line: 620, End line: 641

```python
def paired_manhattan_distances(X, Y):
    """Compute the L1 distances between the vectors in X and Y.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    Y : array-like, shape (n_samples, n_features)

    Returns
    -------
    distances : ndarray (n_samples, )
    """
    X, Y = check_paired_arrays(X, Y)
    diff = X - Y
    if issparse(diff):
        diff.data = np.abs(diff.data)
        return np.squeeze(np.array(diff.sum(axis=1)))
    else:
        return np.abs(diff).sum(axis=-1)
```
### 37 - sklearn/metrics/pairwise.py:

Start line: 1156, End line: 1162

```python
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski',
                  'haversine']
```
### 53 - sklearn/metrics/pairwise.py:

Start line: 1630, End line: 1646

```python
def pairwise_kernels(X, Y=None, metric="linear", filter_params=False,
                     n_jobs=None, **kwds):
    # ... other code

    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return X
    elif isinstance(metric, GPKernel):
        func = metric.__call__
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        if filter_params:
            kwds = {k: kwds[k] for k in kwds
                    if k in KERNEL_PARAMS[metric]}
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(_pairwise_callable, metric=metric, **kwds)
    else:
        raise ValueError("Unknown kernel %r" % metric)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
```
### 59 - sklearn/metrics/pairwise.py:

Start line: 1184, End line: 1199

```python
def _precompute_metric_params(X, Y, metric=None, **kwds):
    """Precompute data-derived metric parameters if not provided
    """
    if metric == "seuclidean" and 'V' not in kwds:
        if X is Y:
            V = np.var(X, axis=0, ddof=1)
        else:
            V = np.var(np.vstack([X, Y]), axis=0, ddof=1)
        return {'V': V}
    if metric == "mahalanobis" and 'VI' not in kwds:
        if X is Y:
            VI = np.linalg.inv(np.cov(X.T)).T
        else:
            VI = np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T
        return {'VI': VI}
    return {}
```
### 60 - sklearn/metrics/pairwise.py:

Start line: 562, End line: 596

```python
def cosine_distances(X, Y=None):
    """Compute cosine distance between samples in X and Y.

    Cosine distance is defined as 1.0 minus the cosine similarity.

    Read more in the :ref:`User Guide <metrics>`.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    distance matrix : array
        An array with shape (n_samples_X, n_samples_Y).

    See also
    --------
    sklearn.metrics.pairwise.cosine_similarity
    scipy.spatial.distance.cosine : dense matrices only
    """
    # 1.0 - cosine_similarity(X, Y) without copy
    S = cosine_similarity(X, Y)
    S *= -1
    S += 1
    np.clip(S, 0, 2, out=S)
    if X is Y or Y is None:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        S[np.diag_indices_from(S)] = 0.0
    return S
```
### 67 - sklearn/metrics/pairwise.py:

Start line: 735, End line: 759

```python
# Kernels
def linear_kernel(X, Y=None, dense_output=True):
    """
    Compute the linear kernel between X and Y.

    Read more in the :ref:`User Guide <linear_kernel>`.

    Parameters
    ----------
    X : array of shape (n_samples_1, n_features)

    Y : array of shape (n_samples_2, n_features)

    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.20

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=dense_output)
```
### 74 - sklearn/metrics/pairwise.py:

Start line: 1072, End line: 1096

```python
def distance_metrics():
    """Valid metrics for pairwise_distances.

    This function simply returns the valid pairwise distance metrics.
    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid distance metrics, and the function they map to, are:

    ============     ====================================
    metric           Function
    ============     ====================================
    'cityblock'      metrics.pairwise.manhattan_distances
    'cosine'         metrics.pairwise.cosine_distances
    'euclidean'      metrics.pairwise.euclidean_distances
    'haversine'      metrics.pairwise.haversine_distances
    'l1'             metrics.pairwise.manhattan_distances
    'l2'             metrics.pairwise.euclidean_distances
    'manhattan'      metrics.pairwise.manhattan_distances
    ============     ====================================

    Read more in the :ref:`User Guide <metrics>`.

    """
    return PAIRWISE_DISTANCE_FUNCTIONS
```

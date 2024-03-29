# scikit-learn__scikit-learn-13910

| **scikit-learn/scikit-learn** | `eb93420e875ba14673157be7df305eb1fac7adce` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1470 |
| **Any found context length** | 1470 |
| **Avg pos** | 6.0 |
| **Min pos** | 2 |
| **Max pos** | 2 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -283,7 +283,7 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     return distances if squared else np.sqrt(distances, out=distances)
 
 
-def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
+def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None, batch_size=None):
     """Euclidean distances between X and Y
 
     Assumes X and Y have float32 dtype.
@@ -298,28 +298,28 @@ def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
 
     distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)
 
-    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
-    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
-
-    # Allow 10% more memory than X, Y and the distance matrix take (at least
-    # 10MiB)
-    maxmem = max(
-        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
-         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
-        10 * 2 ** 17)
-
-    # The increase amount of memory in 8-byte blocks is:
-    # - x_density * batch_size * n_features (copy of chunk of X)
-    # - y_density * batch_size * n_features (copy of chunk of Y)
-    # - batch_size * batch_size (chunk of distance matrix)
-    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
-    #                                 xd=x_density and yd=y_density
-    tmp = (x_density + y_density) * n_features
-    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
-    batch_size = max(int(batch_size), 1)
-
-    x_batches = gen_batches(X.shape[0], batch_size)
-    y_batches = gen_batches(Y.shape[0], batch_size)
+    if batch_size is None:
+        x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
+        y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
+
+        # Allow 10% more memory than X, Y and the distance matrix take (at
+        # least 10MiB)
+        maxmem = max(
+            ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
+             + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
+            10 * 2 ** 17)
+
+        # The increase amount of memory in 8-byte blocks is:
+        # - x_density * batch_size * n_features (copy of chunk of X)
+        # - y_density * batch_size * n_features (copy of chunk of Y)
+        # - batch_size * batch_size (chunk of distance matrix)
+        # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
+        #                                 xd=x_density and yd=y_density
+        tmp = (x_density + y_density) * n_features
+        batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
+        batch_size = max(int(batch_size), 1)
+
+    x_batches = gen_batches(n_samples_X, batch_size)
 
     for i, x_slice in enumerate(x_batches):
         X_chunk = X[x_slice].astype(np.float64)
@@ -328,6 +328,8 @@ def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
         else:
             XX_chunk = XX[x_slice]
 
+        y_batches = gen_batches(n_samples_Y, batch_size)
+
         for j, y_slice in enumerate(y_batches):
             if X is Y and j < i:
                 # when X is Y the distance matrix is symmetric so we only need

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/metrics/pairwise.py | 286 | 286 | 2 | 2 | 1470
| sklearn/metrics/pairwise.py | 301 | 322 | 2 | 2 | 1470
| sklearn/metrics/pairwise.py | 331 | 331 | 2 | 2 | 1470


## Problem Statement

```
Untreated overflow (?) for float32 in euclidean_distances new in sklearn 21.1
#### Description
I am using euclidean distances in a project and after updating, the result is wrong for just one of several datasets. When comparing it to scipy.spatial.distance.cdist one can see that in version 21.1 it behaves substantially different to 20.3.

The matrix is an ndarray with size (100,10000) with float32.

#### Steps/Code to Reproduce

\`\`\`python
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

X = np.load('wont.npy')

ed = euclidean_distances(X)
ed_ = cdist(X, X, metric='euclidean')

plt.plot(np.sort(ed.flatten()), label='euclidean_distances sklearn {}'.format(sklearn.__version__))
plt.plot(np.sort(ed_.flatten()), label='cdist')
plt.yscale('symlog', linthreshy=1E3)
plt.legend()
plt.show()

\`\`\`
The data are in this zip
[wont.zip](https://github.com/scikit-learn/scikit-learn/files/3194196/wont.zip)



#### Expected Results
Can be found when using sklearn 20.3, both behave identical.
[sklearn20.pdf](https://github.com/scikit-learn/scikit-learn/files/3194197/sklearn20.pdf)


#### Actual Results
When using version 21.1 has many 0 entries and some unreasonably high entries 
[sklearn_v21.pdf](https://github.com/scikit-learn/scikit-learn/files/3194198/sklearn_v21.pdf)


#### Versions
Sklearn 21
System:
    python: 3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
executable: /home/lenz/PycharmProjects/pyrolmm/venv_sklearn21/bin/python3
   machine: Linux-4.15.0-50-generic-x86_64-with-Ubuntu-18.04-bionic

BLAS:
    macros: HAVE_CBLAS=None, NO_ATLAS_INFO=-1
  lib_dirs: /usr/lib/x86_64-linux-gnu
cblas_libs: cblas

Python deps:
       pip: 9.0.1
setuptools: 39.0.1
   sklearn: 0.21.1
     numpy: 1.16.3
     scipy: 1.3.0
    Cython: None
    pandas: None

For sklearn 20.3 the versions are:
System:
    python: 3.6.7 (default, Oct 22 2018, 11:32:17)  [GCC 8.2.0]
executable: /home/lenz/PycharmProjects/pyrolmm/venv_sklearn20/bin/python3
   machine: Linux-4.15.0-50-generic-x86_64-with-Ubuntu-18.04-bionic

BLAS:
    macros: HAVE_CBLAS=None, NO_ATLAS_INFO=-1
  lib_dirs: /usr/lib/x86_64-linux-gnu
cblas_libs: cblas

Python deps:
       pip: 9.0.1
setuptools: 39.0.1
   sklearn: 0.20.3
     numpy: 1.16.3
     scipy: 1.3.0
    Cython: None
    pandas: None



<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/plot_anomaly_comparison.py | 81 | 153| 763 | 763 | 1524 | 
| **-> 2 <-** | **2 sklearn/metrics/pairwise.py** | 286 | 356| 707 | 1470 | 16239 | 
| 3 | 3 examples/covariance/plot_mahalanobis_distances.py | 79 | 134| 726 | 2196 | 17902 | 
| 4 | 4 benchmarks/bench_plot_nmf.py | 370 | 422| 514 | 2710 | 21798 | 
| 5 | **4 sklearn/metrics/pairwise.py** | 251 | 283| 342 | 3052 | 21798 | 
| 6 | 5 examples/plot_johnson_lindenstrauss_bound.py | 94 | 168| 709 | 3761 | 23641 | 
| 7 | 6 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 4545 | 26749 | 
| 8 | 7 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 5056 | 27282 | 
| 9 | 8 sklearn/manifold/mds.py | 92 | 132| 424 | 5480 | 31152 | 
| 10 | 9 benchmarks/bench_plot_randomized_svd.py | 281 | 294| 168 | 5648 | 35532 | 
| 11 | 10 examples/cluster/plot_cluster_comparison.py | 1 | 80| 610 | 6258 | 37215 | 
| 12 | 10 benchmarks/bench_plot_randomized_svd.py | 380 | 431| 540 | 6798 | 37215 | 
| 13 | 11 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 7161 | 38333 | 
| 14 | 12 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 7737 | 39468 | 
| 15 | 12 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 947 | 8684 | 39468 | 
| 16 | **12 sklearn/metrics/pairwise.py** | 1530 | 1570| 410 | 9094 | 39468 | 
| 17 | 12 examples/covariance/plot_mahalanobis_distances.py | 1 | 78| 789 | 9883 | 39468 | 
| 18 | **12 sklearn/metrics/pairwise.py** | 1 | 33| 163 | 10046 | 39468 | 
| 19 | 12 examples/cluster/plot_cluster_comparison.py | 94 | 196| 893 | 10939 | 39468 | 
| 20 | 12 benchmarks/bench_plot_randomized_svd.py | 297 | 340| 517 | 11456 | 39468 | 
| 21 | 13 sklearn/cluster/optics_.py | 495 | 527| 348 | 11804 | 47992 | 
| 22 | **13 sklearn/metrics/pairwise.py** | 1573 | 1598| 191 | 11995 | 47992 | 
| 23 | 14 examples/manifold/plot_lle_digits.py | 78 | 169| 748 | 12743 | 50016 | 
| 24 | 14 sklearn/manifold/mds.py | 234 | 276| 436 | 13179 | 50016 | 
| 25 | 15 examples/manifold/plot_mds.py | 1 | 89| 743 | 13922 | 50782 | 
| 26 | 15 examples/plot_anomaly_comparison.py | 1 | 80| 718 | 14640 | 50782 | 
| 27 | 16 benchmarks/bench_lof.py | 36 | 107| 650 | 15290 | 51698 | 
| 28 | 17 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 16075 | 53378 | 
| 29 | 17 examples/plot_johnson_lindenstrauss_bound.py | 170 | 208| 376 | 16451 | 53378 | 
| 30 | **17 sklearn/metrics/pairwise.py** | 1237 | 1243| 137 | 16588 | 53378 | 
| 31 | 18 examples/applications/plot_outlier_detection_housing.py | 1 | 78| 739 | 17327 | 54779 | 
| 32 | 18 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 18142 | 54779 | 
| 33 | 18 examples/cluster/plot_cluster_comparison.py | 82 | 92| 180 | 18322 | 54779 | 
| 34 | 19 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 18989 | 55446 | 
| 35 | 19 examples/applications/plot_outlier_detection_housing.py | 79 | 134| 633 | 19622 | 55446 | 
| 36 | 19 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 20181 | 55446 | 
| 37 | 20 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 20935 | 56665 | 
| 38 | 21 benchmarks/bench_plot_svd.py | 1 | 53| 388 | 21323 | 57344 | 
| 39 | 21 benchmarks/bench_plot_randomized_svd.py | 131 | 178| 522 | 21845 | 57344 | 
| 40 | 21 examples/cluster/plot_agglomerative_clustering_metrics.py | 1 | 91| 733 | 22578 | 57344 | 
| 41 | 22 examples/neighbors/plot_nca_classification.py | 1 | 89| 727 | 23305 | 58079 | 
| 42 | 23 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 775 | 24080 | 59903 | 
| 43 | 23 benchmarks/bench_plot_svd.py | 56 | 83| 290 | 24370 | 59903 | 
| 44 | 24 examples/manifold/plot_compare_methods.py | 1 | 89| 766 | 25136 | 60973 | 
| 45 | 25 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 86| 769 | 25905 | 62126 | 
| 46 | **25 sklearn/metrics/pairwise.py** | 164 | 249| 780 | 26685 | 62126 | 
| 47 | 26 benchmarks/bench_plot_neighbors.py | 26 | 109| 704 | 27389 | 63557 | 
| 48 | 27 examples/neighbors/plot_lof_novelty_detection.py | 68 | 84| 177 | 27566 | 64487 | 
| 49 | 28 examples/cross_decomposition/plot_compare_cross_decomposition.py | 1 | 82| 724 | 28290 | 65970 | 
| 50 | 29 sklearn/utils/estimator_checks.py | 1551 | 1623| 655 | 28945 | 87696 | 
| 51 | 30 benchmarks/bench_glmnet.py | 47 | 129| 796 | 29741 | 88783 | 
| 52 | 30 sklearn/utils/estimator_checks.py | 1304 | 1360| 575 | 30316 | 88783 | 
| 53 | 30 benchmarks/bench_plot_randomized_svd.py | 434 | 456| 194 | 30510 | 88783 | 
| 54 | 31 examples/cluster/plot_ward_structured_vs_unstructured.py | 1 | 94| 689 | 31199 | 89512 | 
| 55 | 32 examples/cluster/plot_cluster_iris.py | 1 | 93| 751 | 31950 | 90271 | 
| 56 | 33 benchmarks/bench_plot_incremental_pca.py | 104 | 151| 468 | 32418 | 91627 | 
| 57 | 34 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 33732 | 92968 | 
| 58 | 34 sklearn/utils/estimator_checks.py | 1157 | 1225| 601 | 34333 | 92968 | 
| 59 | 34 benchmarks/bench_plot_randomized_svd.py | 343 | 377| 423 | 34756 | 92968 | 
| 60 | **34 sklearn/metrics/pairwise.py** | 1400 | 1441| 452 | 35208 | 92968 | 
| 61 | 35 benchmarks/bench_plot_fastkmeans.py | 92 | 137| 495 | 35703 | 94179 | 
| 62 | 36 sklearn/cluster/hierarchical.py | 10 | 26| 126 | 35829 | 103046 | 
| 63 | **36 sklearn/metrics/pairwise.py** | 1132 | 1144| 121 | 35950 | 103046 | 
| 64 | 37 sklearn/metrics/cluster/unsupervised.py | 228 | 236| 140 | 36090 | 106301 | 
| 65 | 37 sklearn/utils/estimator_checks.py | 495 | 543| 491 | 36581 | 106301 | 
| 66 | 38 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 126| 555 | 37136 | 107307 | 
| 67 | 39 examples/impute/plot_missing_values.py | 51 | 95| 440 | 37576 | 108537 | 
| 68 | 40 examples/text/plot_document_classification_20newsgroups.py | 247 | 323| 658 | 38234 | 111072 | 
| 69 | 40 examples/preprocessing/plot_all_scaling.py | 182 | 216| 381 | 38615 | 111072 | 
| 70 | 41 examples/linear_model/plot_lasso_dense_vs_sparse_data.py | 1 | 67| 511 | 39126 | 111583 | 
| 71 | 42 examples/classification/plot_lda.py | 38 | 69| 325 | 39451 | 112167 | 
| 72 | 43 examples/neighbors/plot_species_kde.py | 1 | 78| 690 | 40141 | 113208 | 
| 73 | 43 examples/impute/plot_iterative_imputer_variants_comparison.py | 87 | 133| 384 | 40525 | 113208 | 
| 74 | 43 examples/covariance/plot_mahalanobis_distances.py | 135 | 145| 147 | 40672 | 113208 | 
| 75 | 43 sklearn/utils/estimator_checks.py | 1 | 62| 452 | 41124 | 113208 | 
| 76 | 44 examples/datasets/plot_random_dataset.py | 1 | 68| 645 | 41769 | 113853 | 
| 77 | 44 benchmarks/bench_plot_neighbors.py | 111 | 186| 663 | 42432 | 113853 | 
| 78 | 44 examples/covariance/plot_robust_vs_empirical_covariance.py | 142 | 154| 140 | 42572 | 113853 | 
| 79 | 44 examples/neighbors/plot_lof_novelty_detection.py | 1 | 67| 753 | 43325 | 113853 | 
| 80 | 44 examples/preprocessing/plot_scaling_importance.py | 82 | 134| 457 | 43782 | 113853 | 
| 81 | 44 benchmarks/bench_plot_randomized_svd.py | 199 | 223| 284 | 44066 | 113853 | 
| 82 | 45 examples/cluster/plot_digits_agglomeration.py | 1 | 62| 410 | 44476 | 114271 | 
| 83 | 45 examples/manifold/plot_lle_digits.py | 170 | 247| 608 | 45084 | 114271 | 
| 84 | 46 benchmarks/bench_plot_parallel_pairwise.py | 3 | 47| 299 | 45383 | 114594 | 
| 85 | 47 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 59 | 124| 700 | 46083 | 115776 | 
| 86 | 47 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 46842 | 115776 | 
| 87 | 48 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 95 | 109| 174 | 47016 | 116760 | 
| 88 | 49 examples/covariance/plot_sparse_cov.py | 91 | 139| 435 | 47451 | 117991 | 
| 89 | 50 sklearn/manifold/t_sne.py | 11 | 29| 125 | 47576 | 126398 | 
| 90 | **50 sklearn/metrics/pairwise.py** | 674 | 692| 114 | 47690 | 126398 | 
| 91 | 50 sklearn/cluster/optics_.py | 803 | 890| 757 | 48447 | 126398 | 
| 92 | 51 examples/cluster/plot_kmeans_digits.py | 73 | 127| 565 | 49012 | 127442 | 
| 93 | 52 examples/linear_model/plot_ols_ridge_variance.py | 1 | 68| 484 | 49496 | 127934 | 
| 94 | 53 benchmarks/bench_mnist.py | 84 | 105| 306 | 49802 | 129644 | 
| 95 | 54 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 50299 | 130673 | 
| 96 | 55 benchmarks/bench_glm.py | 1 | 58| 400 | 50699 | 131073 | 
| 97 | 56 examples/covariance/plot_lw_vs_oas.py | 1 | 84| 786 | 51485 | 131859 | 
| 98 | 57 examples/linear_model/plot_sgd_comparison.py | 1 | 60| 472 | 51957 | 132355 | 
| 99 | 57 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 94| 785 | 52742 | 132355 | 
| 100 | 58 examples/bicluster/plot_bicluster_newsgroups.py | 1 | 87| 745 | 53487 | 133713 | 
| 101 | 59 examples/decomposition/plot_pca_vs_fa_model_selection.py | 88 | 126| 437 | 53924 | 134834 | 
| 102 | 60 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 1 | 52| 391 | 54315 | 135893 | 
| 103 | 61 benchmarks/bench_isolation_forest.py | 54 | 161| 1032 | 55347 | 137361 | 
| 104 | 62 examples/cluster/plot_agglomerative_clustering.py | 1 | 81| 671 | 56018 | 138056 | 
| 105 | 63 examples/classification/plot_classifier_comparison.py | 79 | 145| 710 | 56728 | 139408 | 
| 106 | 64 examples/preprocessing/plot_map_data_to_normal.py | 99 | 138| 477 | 57205 | 140643 | 
| 107 | 65 examples/text/plot_hashing_vs_dict_vectorizer.py | 1 | 110| 794 | 57999 | 141454 | 
| 108 | **65 sklearn/metrics/pairwise.py** | 564 | 634| 629 | 58628 | 141454 | 
| 109 | 66 sklearn/datasets/twenty_newsgroups.py | 237 | 305| 620 | 59248 | 145062 | 
| 110 | 66 examples/preprocessing/plot_discretization_classification.py | 109 | 192| 870 | 60118 | 145062 | 
| 111 | 67 benchmarks/bench_feature_expansions.py | 1 | 50| 467 | 60585 | 145529 | 
| 112 | **67 sklearn/metrics/pairwise.py** | 36 | 58| 157 | 60742 | 145529 | 
| 113 | 68 examples/neighbors/plot_nca_dim_reduction.py | 1 | 102| 788 | 61530 | 146325 | 
| 114 | 68 examples/classification/plot_classifier_comparison.py | 1 | 78| 633 | 62163 | 146325 | 
| 115 | 69 benchmarks/bench_random_projections.py | 66 | 82| 174 | 62337 | 148060 | 
| 116 | 69 examples/neighbors/plot_species_kde.py | 80 | 116| 324 | 62661 | 148060 | 
| 117 | 70 examples/classification/plot_lda_qda.py | 33 | 43| 122 | 62783 | 149658 | 
| 118 | 70 examples/plot_johnson_lindenstrauss_bound.py | 1 | 92| 758 | 63541 | 149658 | 
| 119 | 70 examples/impute/plot_missing_values.py | 98 | 143| 391 | 63932 | 149658 | 
| 120 | 71 sklearn/decomposition/online_lda.py | 95 | 132| 445 | 64377 | 156177 | 
| 121 | 72 sklearn/datasets/species_distributions.py | 227 | 265| 363 | 64740 | 158293 | 
| 122 | 73 sklearn/datasets/rcv1.py | 1 | 68| 848 | 65588 | 161115 | 
| 123 | 74 benchmarks/bench_plot_lasso_path.py | 83 | 116| 347 | 65935 | 162089 | 
| 124 | 75 sklearn/covariance/elliptic_envelope.py | 155 | 189| 247 | 66182 | 163745 | 
| 125 | 76 sklearn/metrics/__init__.py | 74 | 136| 407 | 66589 | 164700 | 
| 126 | 77 sklearn/covariance/robust_covariance.py | 413 | 509| 1054 | 67643 | 171775 | 
| 127 | 78 sklearn/cluster/dbscan_.py | 139 | 190| 538 | 68181 | 175346 | 
| 128 | 78 sklearn/utils/estimator_checks.py | 1531 | 1548| 185 | 68366 | 175346 | 
| 129 | 79 examples/manifold/plot_t_sne_perplexity.py | 79 | 127| 440 | 68806 | 176468 | 
| 130 | 80 examples/applications/wikipedia_principal_eigenvector.py | 187 | 230| 376 | 69182 | 178365 | 
| 131 | 80 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 69688 | 178365 | 
| 132 | 81 examples/cluster/plot_digits_linkage.py | 1 | 38| 229 | 69917 | 179094 | 
| 133 | 81 examples/manifold/plot_t_sne_perplexity.py | 1 | 77| 661 | 70578 | 179094 | 
| 134 | 81 sklearn/utils/estimator_checks.py | 2355 | 2399| 427 | 71005 | 179094 | 
| 135 | 82 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 71537 | 181039 | 
| 136 | 83 examples/preprocessing/plot_discretization.py | 1 | 87| 779 | 72316 | 181848 | 
| 137 | 84 benchmarks/bench_tree.py | 64 | 125| 523 | 72839 | 182718 | 
| 138 | 84 examples/covariance/plot_robust_vs_empirical_covariance.py | 1 | 76| 755 | 73594 | 182718 | 
| 139 | 84 examples/preprocessing/plot_all_scaling.py | 311 | 356| 353 | 73947 | 182718 | 
| 140 | 84 sklearn/utils/estimator_checks.py | 2067 | 2081| 211 | 74158 | 182718 | 
| 141 | 85 examples/compose/plot_transformed_target.py | 99 | 178| 746 | 74904 | 184532 | 
| 142 | 85 benchmarks/bench_plot_randomized_svd.py | 111 | 128| 156 | 75060 | 184532 | 
| 143 | 85 benchmarks/bench_lof.py | 1 | 35| 266 | 75326 | 184532 | 
| 144 | 86 examples/neighbors/plot_kde_1d.py | 73 | 152| 790 | 76116 | 186100 | 
| 145 | 87 examples/neighbors/plot_nca_illustration.py | 1 | 99| 758 | 76874 | 186866 | 
| 146 | 87 sklearn/datasets/species_distributions.py | 1 | 74| 541 | 77415 | 186866 | 
| 147 | **87 sklearn/metrics/pairwise.py** | 719 | 750| 218 | 77633 | 186866 | 
| 148 | 88 benchmarks/bench_hist_gradient_boosting.py | 158 | 242| 750 | 78383 | 189092 | 
| 149 | 89 benchmarks/bench_multilabel_metrics.py | 134 | 189| 543 | 78926 | 190728 | 
| 150 | 90 sklearn/datasets/__init__.py | 55 | 103| 341 | 79267 | 191586 | 


### Hint

```
So it is because of the dtype, so it is probably some overflow. 
It does not give any warning or error though, and this did not happen before.
[float32.pdf](https://github.com/scikit-learn/scikit-learn/files/3194307/float32.pdf)



\`\`\`python
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

X = np.random.uniform(0,2,(100,10000))

ed = euclidean_distances(X)
title_ed = 'euc dist type: {}'.format(X.dtype)
X = X.astype('float32')
ed_ = euclidean_distances(X)
title_ed_ = 'euc dist type: {}'.format(X.dtype)

plt.plot(np.sort(ed.flatten()), label=title_ed)
plt.plot(np.sort(ed_.flatten()), label=title_ed_)
plt.yscale('symlog', linthreshy=1E3)
plt.legend()
plt.show()
\`\`\`
Thanks for reporting this @lenz3000. I can reproduce with the above example. It is likely due to https://github.com/scikit-learn/scikit-learn/pull/13554 which improves the numerical precision of `euclidean_distances` in some edge cases, but it looks like it has some side effects. It would be worth invesigating what is happening in this example (were the data is reasonably normalized).

cc @jeremiedbb
```

## Patch

```diff
diff --git a/sklearn/metrics/pairwise.py b/sklearn/metrics/pairwise.py
--- a/sklearn/metrics/pairwise.py
+++ b/sklearn/metrics/pairwise.py
@@ -283,7 +283,7 @@ def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
     return distances if squared else np.sqrt(distances, out=distances)
 
 
-def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
+def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None, batch_size=None):
     """Euclidean distances between X and Y
 
     Assumes X and Y have float32 dtype.
@@ -298,28 +298,28 @@ def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
 
     distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)
 
-    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
-    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
-
-    # Allow 10% more memory than X, Y and the distance matrix take (at least
-    # 10MiB)
-    maxmem = max(
-        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
-         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
-        10 * 2 ** 17)
-
-    # The increase amount of memory in 8-byte blocks is:
-    # - x_density * batch_size * n_features (copy of chunk of X)
-    # - y_density * batch_size * n_features (copy of chunk of Y)
-    # - batch_size * batch_size (chunk of distance matrix)
-    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
-    #                                 xd=x_density and yd=y_density
-    tmp = (x_density + y_density) * n_features
-    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
-    batch_size = max(int(batch_size), 1)
-
-    x_batches = gen_batches(X.shape[0], batch_size)
-    y_batches = gen_batches(Y.shape[0], batch_size)
+    if batch_size is None:
+        x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
+        y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
+
+        # Allow 10% more memory than X, Y and the distance matrix take (at
+        # least 10MiB)
+        maxmem = max(
+            ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
+             + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
+            10 * 2 ** 17)
+
+        # The increase amount of memory in 8-byte blocks is:
+        # - x_density * batch_size * n_features (copy of chunk of X)
+        # - y_density * batch_size * n_features (copy of chunk of Y)
+        # - batch_size * batch_size (chunk of distance matrix)
+        # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
+        #                                 xd=x_density and yd=y_density
+        tmp = (x_density + y_density) * n_features
+        batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
+        batch_size = max(int(batch_size), 1)
+
+    x_batches = gen_batches(n_samples_X, batch_size)
 
     for i, x_slice in enumerate(x_batches):
         X_chunk = X[x_slice].astype(np.float64)
@@ -328,6 +328,8 @@ def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
         else:
             XX_chunk = XX[x_slice]
 
+        y_batches = gen_batches(n_samples_Y, batch_size)
+
         for j, y_slice in enumerate(y_batches):
             if X is Y and j < i:
                 # when X is Y the distance matrix is symmetric so we only need

```

## Test Patch

```diff
diff --git a/sklearn/metrics/tests/test_pairwise.py b/sklearn/metrics/tests/test_pairwise.py
--- a/sklearn/metrics/tests/test_pairwise.py
+++ b/sklearn/metrics/tests/test_pairwise.py
@@ -48,6 +48,7 @@
 from sklearn.metrics.pairwise import paired_distances
 from sklearn.metrics.pairwise import paired_euclidean_distances
 from sklearn.metrics.pairwise import paired_manhattan_distances
+from sklearn.metrics.pairwise import _euclidean_distances_upcast
 from sklearn.preprocessing import normalize
 from sklearn.exceptions import DataConversionWarning
 
@@ -687,6 +688,52 @@ def test_euclidean_distances_sym(dtype, x_array_constr):
     assert distances.dtype == dtype
 
 
+@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
+@pytest.mark.parametrize("x_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+@pytest.mark.parametrize("y_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances_upcast(batch_size, x_array_constr,
+                                    y_array_constr):
+    # check batches handling when Y != X (#13910)
+    rng = np.random.RandomState(0)
+    X = rng.random_sample((100, 10)).astype(np.float32)
+    X[X < 0.8] = 0
+    Y = rng.random_sample((10, 10)).astype(np.float32)
+    Y[Y < 0.8] = 0
+
+    expected = cdist(X, Y)
+
+    X = x_array_constr(X)
+    Y = y_array_constr(Y)
+    distances = _euclidean_distances_upcast(X, Y=Y, batch_size=batch_size)
+    distances = np.sqrt(np.maximum(distances, 0))
+
+    # the default rtol=1e-7 is too close to the float32 precision
+    # and fails due too rounding errors.
+    assert_allclose(distances, expected, rtol=1e-6)
+
+
+@pytest.mark.parametrize("batch_size", [None, 5, 7, 101])
+@pytest.mark.parametrize("x_array_constr", [np.array, csr_matrix],
+                         ids=["dense", "sparse"])
+def test_euclidean_distances_upcast_sym(batch_size, x_array_constr):
+    # check batches handling when X is Y (#13910)
+    rng = np.random.RandomState(0)
+    X = rng.random_sample((100, 10)).astype(np.float32)
+    X[X < 0.8] = 0
+
+    expected = squareform(pdist(X))
+
+    X = x_array_constr(X)
+    distances = _euclidean_distances_upcast(X, Y=X, batch_size=batch_size)
+    distances = np.sqrt(np.maximum(distances, 0))
+
+    # the default rtol=1e-7 is too close to the float32 precision
+    # and fails due too rounding errors.
+    assert_allclose(distances, expected, rtol=1e-6)
+
+
 @pytest.mark.parametrize(
     "dtype, eps, rtol",
     [(np.float32, 1e-4, 1e-5),

```


## Code snippets

### 1 - examples/plot_anomaly_comparison.py:

Start line: 81, End line: 153

```python
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(behaviour='new',
                                         contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# Compare given classifiers under given settings
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    # Add outliers
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                       size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
```
### 2 - sklearn/metrics/pairwise.py:

Start line: 286, End line: 356

```python
def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
    """Euclidean distances between X and Y

    Assumes X and Y have float32 dtype.
    Assumes XX and YY have float64 dtype or are None.

    X and Y are upcast to float64 by chunks, which size is chosen to limit
    memory increase by approximately 10% (at least 10MiB).
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)

    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1

    # Allow 10% more memory than X, Y and the distance matrix take (at least
    # 10MiB)
    maxmem = max(
        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
        10 * 2 ** 17)

    # The increase amount of memory in 8-byte blocks is:
    # - x_density * batch_size * n_features (copy of chunk of X)
    # - y_density * batch_size * n_features (copy of chunk of Y)
    # - batch_size * batch_size (chunk of distance matrix)
    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
    #                                 xd=x_density and yd=y_density
    tmp = (x_density + y_density) * n_features
    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
    batch_size = max(int(batch_size), 1)

    x_batches = gen_batches(X.shape[0], batch_size)
    y_batches = gen_batches(Y.shape[0], batch_size)

    for i, x_slice in enumerate(x_batches):
        X_chunk = X[x_slice].astype(np.float64)
        if XX is None:
            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]

        for j, y_slice in enumerate(y_batches):
            if X is Y and j < i:
                # when X is Y the distance matrix is symmetric so we only need
                # to compute half of it.
                d = distances[y_slice, x_slice].T

            else:
                Y_chunk = Y[y_slice].astype(np.float64)
                if YY is None:
                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
                else:
                    YY_chunk = YY[:, y_slice]

                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
                d += XX_chunk
                d += YY_chunk

            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)

    return distances


def _argmin_min_reduce(dist, start):
    indices = dist.argmin(axis=1)
    values = dist[np.arange(dist.shape[0]), indices]
    return indices, values
```
### 3 - examples/covariance/plot_mahalanobis_distances.py:

Start line: 79, End line: 134

```python
emp_cov = EmpiricalCovariance().fit(X)

# #############################################################################
# Display results
fig = plt.figure()
plt.subplots_adjust(hspace=-.1, wspace=.4, top=.95, bottom=.05)

# Show data set
subfig1 = plt.subplot(3, 1, 1)
inlier_plot = subfig1.scatter(X[:, 0], X[:, 1],
                              color='black', label='inliers')
outlier_plot = subfig1.scatter(X[:, 0][-n_outliers:], X[:, 1][-n_outliers:],
                               color='red', label='outliers')
subfig1.set_xlim(subfig1.get_xlim()[0], 11.)
subfig1.set_title("Mahalanobis distances of a contaminated data set:")

# Show contours of the distance functions
xx, yy = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
                     np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
zz = np.c_[xx.ravel(), yy.ravel()]

mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = subfig1.contour(xx, yy, np.sqrt(mahal_emp_cov),
                                  cmap=plt.cm.PuBu_r,
                                  linestyles='dashed')

mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = subfig1.contour(xx, yy, np.sqrt(mahal_robust_cov),
                                 cmap=plt.cm.YlOrBr_r, linestyles='dotted')

subfig1.legend([emp_cov_contour.collections[1], robust_contour.collections[1],
                inlier_plot, outlier_plot],
               ['MLE dist', 'robust dist', 'inliers', 'outliers'],
               loc="upper right", borderaxespad=0)
plt.xticks(())
plt.yticks(())

# Plot the scores for each point
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
subfig2 = plt.subplot(2, 2, 3)
subfig2.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=.25)
subfig2.plot(np.full(n_samples - n_outliers, 1.26),
             emp_mahal[:-n_outliers], '+k', markeredgewidth=1)
subfig2.plot(np.full(n_outliers, 2.26),
             emp_mahal[-n_outliers:], '+k', markeredgewidth=1)
subfig2.axes.set_xticklabels(('inliers', 'outliers'), size=15)
subfig2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
subfig2.set_title("1. from non-robust estimates\n(Maximum Likelihood)")
plt.yticks(())

robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
subfig3 = plt.subplot(2, 2, 4)
subfig3.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]],
                widths=.25)
```
### 4 - benchmarks/bench_plot_nmf.py:

Start line: 370, End line: 422

```python
def load_20news():
    print("Loading 20 newsgroups dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                                 remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(dataset.data)
    return tfidf


def load_faces():
    print("Loading Olivetti face dataset")
    print("-----------------------------")
    from sklearn.datasets import fetch_olivetti_faces
    faces = fetch_olivetti_faces(shuffle=True)
    return faces.data


def build_clfs(cd_iters, pg_iters, mu_iters):
    clfs = [("Coordinate Descent", NMF, cd_iters, {'solver': 'cd'}),
            ("Projected Gradient", _PGNMF, pg_iters, {'solver': 'pg'}),
            ("Multiplicative Update", NMF, mu_iters, {'solver': 'mu'}),
            ]
    return clfs


if __name__ == '__main__':
    alpha = 0.
    l1_ratio = 0.5
    n_components = 10
    tol = 1e-15

    # first benchmark on 20 newsgroup dataset: sparse, shape(11314, 39116)
    plot_name = "20 Newsgroups sparse dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 6)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_20news = load_20news()
    run_bench(X_20news, clfs, plot_name, n_components, tol, alpha, l1_ratio)

    # second benchmark on Olivetti faces dataset: dense, shape(400, 4096)
    plot_name = "Olivetti Faces dense dataset"
    cd_iters = np.arange(1, 30)
    pg_iters = np.arange(1, 12)
    mu_iters = np.arange(1, 30)
    clfs = build_clfs(cd_iters, pg_iters, mu_iters)
    X_faces = load_faces()
    run_bench(X_faces, clfs, plot_name, n_components, tol, alpha, l1_ratio,)

    plt.show()
```
### 5 - sklearn/metrics/pairwise.py:

Start line: 251, End line: 283

```python
def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False,
                        X_norm_squared=None):
    # ... other code

    if X is Y and XX is not None:
        # shortcut in the common case euclidean_distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]

    if X.dtype == np.float32:
        # To minimize precision issues with float32, we compute the distance
        # matrix on chunks of X and Y upcast to float64
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        # if dtype is already float64, no need to chunk and upcast
        distances = - 2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)
```
### 6 - examples/plot_johnson_lindenstrauss_bound.py:

Start line: 94, End line: 168

```python
import sys
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances

# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}

# Part 1: plot the theoretical dependency between n_components_min and
# n_samples

# range of admissible distortions
eps_range = np.linspace(0.1, 0.99, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

# range of number of samples (observation) to embed
n_samples_range = np.logspace(1, 9, 9)

plt.figure()
for eps, color in zip(eps_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
    plt.loglog(n_samples_range, min_n_components, color=color)

plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
plt.xlabel("Number of observations to eps-embed")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Lindenstrauss bounds:\nn_samples vs n_components")

# range of admissible distortions
eps_range = np.linspace(0.01, 0.99, 100)

# range of number of samples (observation) to embed
n_samples_range = np.logspace(2, 6, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

plt.figure()
for n_samples, color in zip(n_samples_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
    plt.semilogy(eps_range, min_n_components, color=color)

plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
plt.xlabel("Distortion eps")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")

# Part 2: perform sparse random projection of some digits images which are
# quite low dimensional and dense or documents of the 20 newsgroups dataset
# which is both high dimensional and sparse

if '--twenty-newsgroups' in sys.argv:
    # Need an internet connection hence not enabled by default
    data = fetch_20newsgroups_vectorized().data[:500]
else:
    data = load_digits().data[:500]

n_samples, n_features = data.shape
print("Embedding %d samples with dim %d using various random projections"
      % (n_samples, n_features))

n_components_range = np.array([300, 1000, 10000])
dists = euclidean_distances(data, squared=True).ravel()

# select only non-identical samples pairs
nonzero = dists != 0
dists = dists[nonzero]
```
### 7 - examples/preprocessing/plot_all_scaling.py:

Start line: 1, End line: 105

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
=============================================================
Compare the effect of different scalers on data with outliers
=============================================================

Feature 0 (median income in a block) and feature 5 (number of households) of
the `California housing dataset
<https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html>`_ have very
different scales and contain some very large outliers. These two
characteristics lead to difficulties to visualize the data and, more
importantly, they can degrade the predictive performance of many machine
learning algorithms. Unscaled data can also slow down or even prevent the
convergence of many gradient-based estimators.

Indeed many estimators are designed with the assumption that each feature takes
values close to zero or more importantly that all features vary on comparable
scales. In particular, metric-based and gradient-based estimators often assume
approximately standardized data (centered features with unit variances). A
notable exception are decision tree-based estimators that are robust to
arbitrary scaling of the data.

This example uses different scalers, transformers, and normalizers to bring the
data within a pre-defined range.

Scalers are linear (or more precisely affine) transformers and differ from each
other in the way to estimate the parameters used to shift and scale each
feature.

``QuantileTransformer`` provides non-linear transformations in which distances
between marginal outliers and inliers are shrunk. ``PowerTransformer`` provides
non-linear transformations in which data is mapped to a normal distribution to
stabilize variance and minimize skewness.

Unlike the previous transformations, normalization refers to a per sample
transformation instead of a per feature transformation.

The following code is a bit verbose, feel free to jump directly to the analysis
of the results_.

"""

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.datasets import fetch_california_housing

print(__doc__)

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target

# Take only 2 features to make visualization easier
# Feature of 0 has a long tail distribution.
# Feature 5 has a few but very large outliers.

X = X_full[:, [0, 5]]

distributions = [
    ('Unscaled data', X),
    ('Data after standard scaling',
        StandardScaler().fit_transform(X)),
    ('Data after min-max scaling',
        MinMaxScaler().fit_transform(X)),
    ('Data after max-abs scaling',
        MaxAbsScaler().fit_transform(X)),
    ('Data after robust scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('Data after power transformation (Yeo-Johnson)',
     PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('Data after power transformation (Box-Cox)',
     PowerTransformer(method='box-cox').fit_transform(X)),
    ('Data after quantile transformation (gaussian pdf)',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('Data after quantile transformation (uniform pdf)',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('Data after sample-wise L2 normalizing',
        Normalizer().fit_transform(X)),
]

# scale the output between 0 and 1 for the colorbar
y = minmax_scale(y_full)

# plasma does not exist in matplotlib < 1.5
```
### 8 - examples/cluster/plot_kmeans_assumptions.py:

Start line: 1, End line: 65

```python
"""
====================================
Demonstration of k-means assumptions
====================================

This example is meant to illustrate situations where k-means will produce
unintuitive and possibly unexpected clusters. In the first three plots, the
input data does not conform to some implicit assumption that k-means makes and
undesirable clusters are produced as a result. In the last plot, k-means
returns intuitive clusters despite unevenly sized blobs.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
```
### 9 - sklearn/manifold/mds.py:

Start line: 92, End line: 132

```python
def _smacof_single(dissimilarities, metric=True, n_components=2, init=None,
                   max_iter=300, verbose=0, eps=1e-3, random_state=None):
    # ... other code
    for it in range(max_iter):
        # Compute distance and monotonic regression
        dis = euclidean_distances(X)

        if metric:
            disparities = dissimilarities
        else:
            dis_flat = dis.ravel()
            # dissimilarities with 0 are considered as missing values
            dis_flat_w = dis_flat[sim_flat != 0]

            # Compute the disparities using a monotonic regression
            disparities_flat = ir.fit_transform(sim_flat_w, dis_flat_w)
            disparities = dis_flat.copy()
            disparities[sim_flat != 0] = disparities_flat
            disparities = disparities.reshape((n_samples, n_samples))
            disparities *= np.sqrt((n_samples * (n_samples - 1) / 2) /
                                   (disparities ** 2).sum())

        # Compute stress
        stress = ((dis.ravel() - disparities.ravel()) ** 2).sum() / 2

        # Update X using the Guttman transform
        dis[dis == 0] = 1e-5
        ratio = disparities / dis
        B = - ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
        X = 1. / n_samples * np.dot(B, X)

        dis = np.sqrt((X ** 2).sum(axis=1)).sum()
        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))
        if old_stress is not None:
            if(old_stress - stress / dis) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break
        old_stress = stress / dis

    return X, stress, it + 1
```
### 10 - benchmarks/bench_plot_randomized_svd.py:

Start line: 281, End line: 294

```python
def scalable_frobenius_norm_discrepancy(X, U, s, V):
    # if the input is not too big, just call scipy
    if X.shape[0] * X.shape[1] < MAX_MEMORY:
        A = X - U.dot(np.diag(s).dot(V))
        return norm_diff(A, norm='fro')

    print("... computing fro norm by batches...")
    batch_size = 1000
    Vhat = np.diag(s).dot(V)
    cum_norm = .0
    for batch in gen_batches(X.shape[0], batch_size):
        M = X[batch, :] - U[batch, :].dot(Vhat)
        cum_norm += norm_diff(M, norm='fro', msg=False)
    return np.sqrt(cum_norm)
```
### 16 - sklearn/metrics/pairwise.py:

Start line: 1530, End line: 1570

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

        if (dtype == bool and
                (X.dtype != bool or (Y is not None and Y.dtype != bool))):
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
### 18 - sklearn/metrics/pairwise.py:

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

Start line: 1573, End line: 1598

```python
# These distances require boolean arrays, when using scipy.spatial.distance
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
### 30 - sklearn/metrics/pairwise.py:

Start line: 1237, End line: 1243

```python
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski',
                  'haversine']
```
### 46 - sklearn/metrics/pairwise.py:

Start line: 164, End line: 249

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
        May be ignored in some cases, see the note below.

    squared : boolean, optional
        Return squared Euclidean distances.

    X_norm_squared : array-like, shape = [n_samples_1], optional
        Pre-computed dot-products of vectors in X (e.g.,
        ``(X**2).sum(axis=1)``)
        May be ignored in some cases, see the note below.

    Notes
    -----
    To achieve better accuracy, `X_norm_squared` and `Y_norm_squared` may be
    unused if they are passed as ``float32``.

    Returns
    -------
    distances : array, shape (n_samples_1, n_samples_2)

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

    # If norms are passed as float32, they are unused. If arrays are passed as
    # float32, norms needs to be recomputed on upcast chunks.
    # TODO: use a float64 accumulator in row_norms to avoid the latter.
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
        if XX.dtype == np.float32:
            XX = None
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    # ... other code
```
### 60 - sklearn/metrics/pairwise.py:

Start line: 1400, End line: 1441

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
### 63 - sklearn/metrics/pairwise.py:

Start line: 1132, End line: 1144

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
### 90 - sklearn/metrics/pairwise.py:

Start line: 674, End line: 692

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
### 108 - sklearn/metrics/pairwise.py:

Start line: 564, End line: 634

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
### 112 - sklearn/metrics/pairwise.py:

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
### 147 - sklearn/metrics/pairwise.py:

Start line: 719, End line: 750

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
    -----
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

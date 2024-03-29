# scikit-learn__scikit-learn-15512

| **scikit-learn/scikit-learn** | `b8a4da8baa1137f173e7035f104067c7d2ffde22` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 8456 |
| **Any found context length** | 1074 |
| **Avg pos** | 19.0 |
| **Min pos** | 2 |
| **Max pos** | 17 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/cluster/_affinity_propagation.py b/sklearn/cluster/_affinity_propagation.py
--- a/sklearn/cluster/_affinity_propagation.py
+++ b/sklearn/cluster/_affinity_propagation.py
@@ -194,17 +194,19 @@ def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
             unconverged = (np.sum((se == convergence_iter) + (se == 0))
                            != n_samples)
             if (not unconverged and (K > 0)) or (it == max_iter):
+                never_converged = False
                 if verbose:
                     print("Converged after %d iterations." % it)
                 break
     else:
+        never_converged = True
         if verbose:
             print("Did not converge")
 
     I = np.flatnonzero(E)
     K = I.size  # Identify exemplars
 
-    if K > 0:
+    if K > 0 and not never_converged:
         c = np.argmax(S[:, I], axis=1)
         c[I] = np.arange(K)  # Identify clusters
         # Refine the final set of exemplars and clusters and return results
@@ -408,6 +410,7 @@ def predict(self, X):
             Cluster labels.
         """
         check_is_fitted(self)
+        X = check_array(X)
         if not hasattr(self, "cluster_centers_"):
             raise ValueError("Predict method is not supported when "
                              "affinity='precomputed'.")

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/cluster/_affinity_propagation.py | 197 | 197 | 2 | 1 | 1074
| sklearn/cluster/_affinity_propagation.py | 411 | 411 | 17 | 1 | 8456


## Problem Statement

```
Return values of non converged affinity propagation clustering
The affinity propagation Documentation states: 
"When the algorithm does not converge, it returns an empty array as cluster_center_indices and -1 as label for each training sample."

Example:
\`\`\`python
from sklearn.cluster import AffinityPropagation
import pandas as pd

data = pd.DataFrame([[1,0,0,0,0,0],[0,1,1,1,0,0],[0,0,1,0,0,1]])
af = AffinityPropagation(affinity='euclidean', verbose=True, copy=False, max_iter=2).fit(data)

print(af.cluster_centers_indices_)
print(af.labels_)

\`\`\`
I would expect that the clustering here (which does not converge) prints first an empty List and then [-1,-1,-1], however, I get [2] as cluster center and [0,0,0] as cluster labels. 
The only way I currently know if the clustering fails is if I use the verbose option, however that is very unhandy. A hacky solution is to check if max_iter == n_iter_ but it could have converged exactly 15 iterations before max_iter (although unlikely).
I am not sure if this is intended behavior and the documentation is wrong?

For my use-case within a bigger script, I would prefer to get back -1 values or have a property to check if it has converged, as otherwise, a user might not be aware that the clustering never converged.


#### Versions
System:
    python: 3.6.7 | packaged by conda-forge | (default, Nov 21 2018, 02:32:25)  [GCC 4.8.2 20140120 (Red Hat 4.8.2-15)]
executable: /home/jenniferh/Programs/anaconda3/envs/TF_RDKit_1_19/bin/python
   machine: Linux-4.15.0-52-generic-x86_64-with-debian-stretch-sid
BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /home/jenniferh/Programs/anaconda3/envs/TF_RDKit_1_19/lib
cblas_libs: mkl_rt, pthread
Python deps:
    pip: 18.1
   setuptools: 40.6.3
   sklearn: 0.20.3
   numpy: 1.15.4
   scipy: 1.2.0
   Cython: 0.29.2
   pandas: 0.23.4



```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/cluster/_affinity_propagation.py** | 207 | 231| 278 | 278 | 3589 | 
| **-> 2 <-** | **1 sklearn/cluster/_affinity_propagation.py** | 121 | 205| 796 | 1074 | 3589 | 
| 3 | **1 sklearn/cluster/_affinity_propagation.py** | 234 | 334| 838 | 1912 | 3589 | 
| 4 | 2 examples/cluster/plot_affinity_propagation.py | 1 | 63| 524 | 2436 | 4113 | 
| 5 | 3 sklearn/utils/estimator_checks.py | 1608 | 1664| 564 | 3000 | 29079 | 
| 6 | **3 sklearn/cluster/_affinity_propagation.py** | 336 | 394| 471 | 3471 | 29079 | 
| 7 | **3 sklearn/cluster/_affinity_propagation.py** | 33 | 119| 730 | 4201 | 29079 | 
| 8 | 4 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 4712 | 29612 | 
| 9 | **4 sklearn/cluster/_affinity_propagation.py** | 1 | 30| 171 | 4883 | 29612 | 
| 10 | 5 examples/cluster/plot_cluster_iris.py | 1 | 93| 751 | 5634 | 30371 | 
| 11 | 6 examples/cluster/plot_cluster_comparison.py | 1 | 80| 610 | 6244 | 32054 | 
| 12 | 7 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 6750 | 33113 | 
| 13 | **7 sklearn/cluster/_affinity_propagation.py** | 423 | 444| 174 | 6924 | 33113 | 
| 14 | 7 examples/cluster/plot_cluster_comparison.py | 94 | 196| 893 | 7817 | 33113 | 
| 15 | 8 benchmarks/bench_rcv1_logreg_convergence.py | 67 | 92| 201 | 8018 | 35059 | 
| 16 | 8 benchmarks/bench_rcv1_logreg_convergence.py | 95 | 121| 218 | 8236 | 35059 | 
| **-> 17 <-** | **8 sklearn/cluster/_affinity_propagation.py** | 396 | 421| 220 | 8456 | 35059 | 
| 18 | 9 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 8819 | 36177 | 
| 19 | 9 sklearn/utils/estimator_checks.py | 211 | 236| 230 | 9049 | 36177 | 
| 20 | 10 sklearn/cluster/_k_means.py | 1263 | 1326| 659 | 9708 | 51363 | 
| 21 | 11 examples/cluster/plot_agglomerative_clustering.py | 1 | 81| 671 | 10379 | 52058 | 
| 22 | 12 sklearn/cluster/_spectral.py | 250 | 550| 262 | 10641 | 57005 | 
| 23 | 13 examples/cluster/plot_dbscan.py | 1 | 76| 623 | 11264 | 57628 | 
| 24 | 13 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 1 | 52| 391 | 11655 | 57628 | 
| 25 | 14 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 12214 | 58763 | 
| 26 | 15 sklearn/cluster/_optics.py | 809 | 896| 757 | 12971 | 67335 | 
| 27 | 16 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 126| 555 | 13526 | 68341 | 
| 28 | 17 examples/cluster/plot_inductive_clustering.py | 59 | 121| 493 | 14019 | 69250 | 
| 29 | 17 examples/cluster/plot_cluster_comparison.py | 82 | 92| 180 | 14199 | 69250 | 
| 30 | 18 sklearn/cluster/_hierarchical.py | 758 | 775| 169 | 14368 | 78254 | 
| 31 | 18 benchmarks/bench_rcv1_logreg_convergence.py | 124 | 139| 148 | 14516 | 78254 | 
| 32 | 19 examples/cluster/plot_kmeans_silhouette_analysis.py | 55 | 140| 902 | 15418 | 79660 | 
| 33 | 20 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 50 | 116| 669 | 16087 | 80670 | 
| 34 | 21 sklearn/metrics/cluster/_supervised.py | 1 | 31| 134 | 16221 | 89369 | 
| 35 | 21 sklearn/cluster/_hierarchical.py | 653 | 756| 969 | 17190 | 89369 | 
| 36 | 21 sklearn/cluster/_k_means.py | 649 | 790| 1319 | 18509 | 89369 | 
| 37 | 21 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 19085 | 89369 | 
| 38 | 21 sklearn/utils/estimator_checks.py | 1667 | 1679| 131 | 19216 | 89369 | 
| 39 | 21 sklearn/metrics/cluster/_supervised.py | 225 | 240| 227 | 19443 | 89369 | 
| 40 | 22 examples/cluster/plot_digits_linkage.py | 55 | 72| 189 | 19632 | 90090 | 
| 41 | 22 examples/cluster/plot_kmeans_silhouette_analysis.py | 1 | 53| 504 | 20136 | 90090 | 
| 42 | 22 sklearn/cluster/_k_means.py | 1557 | 1634| 722 | 20858 | 90090 | 
| 43 | 22 examples/cluster/plot_agglomerative_clustering_metrics.py | 1 | 91| 733 | 21591 | 90090 | 
| 44 | 23 examples/cluster/plot_ward_structured_vs_unstructured.py | 1 | 94| 687 | 22278 | 90817 | 
| 45 | 24 examples/cluster/plot_mini_batch_kmeans.py | 87 | 117| 308 | 22586 | 91848 | 
| 46 | 24 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 55 | 68| 136 | 22722 | 91848 | 
| 47 | 24 sklearn/cluster/_spectral.py | 159 | 249| 871 | 23593 | 91848 | 
| 48 | 25 examples/bicluster/plot_spectral_coclustering.py | 1 | 59| 390 | 23983 | 92264 | 
| 49 | 25 examples/cluster/plot_inductive_clustering.py | 1 | 35| 250 | 24233 | 92264 | 
| 50 | 25 sklearn/cluster/_k_means.py | 418 | 458| 407 | 24640 | 92264 | 
| 51 | 26 examples/cluster/plot_kmeans_digits.py | 73 | 127| 565 | 25205 | 93316 | 
| 52 | 26 sklearn/cluster/_hierarchical.py | 10 | 26| 132 | 25337 | 93316 | 
| 53 | 26 examples/cluster/plot_adjusted_for_chance_measures.py | 1 | 31| 193 | 25530 | 93316 | 
| 54 | 27 examples/cluster/plot_digits_agglomeration.py | 1 | 62| 410 | 25940 | 93734 | 
| 55 | 28 examples/plot_anomaly_comparison.py | 81 | 152| 757 | 26697 | 95252 | 
| 56 | 28 sklearn/metrics/cluster/_supervised.py | 739 | 769| 391 | 27088 | 95252 | 
| 57 | 28 sklearn/metrics/cluster/_supervised.py | 34 | 61| 208 | 27296 | 95252 | 
| 58 | 29 examples/cluster/plot_optics.py | 75 | 99| 375 | 27671 | 96420 | 
| 59 | 29 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 28203 | 96420 | 
| 60 | 30 benchmarks/bench_plot_neighbors.py | 26 | 109| 704 | 28907 | 97855 | 
| 61 | 30 examples/cluster/plot_kmeans_digits.py | 1 | 56| 327 | 29234 | 97855 | 
| 62 | 31 sklearn/semi_supervised/_label_propagation.py | 1 | 71| 604 | 29838 | 101946 | 
| 63 | 31 examples/cluster/plot_digits_linkage.py | 1 | 36| 221 | 30059 | 101946 | 
| 64 | 31 examples/cluster/plot_inductive_clustering.py | 38 | 56| 149 | 30208 | 101946 | 
| 65 | 31 sklearn/cluster/_k_means.py | 104 | 135| 319 | 30527 | 101946 | 
| 66 | 32 examples/cluster/plot_mean_shift.py | 1 | 57| 396 | 30923 | 102342 | 
| 67 | 32 sklearn/cluster/_k_means.py | 37 | 103| 585 | 31508 | 102342 | 
| 68 | 33 sklearn/cluster/__init__.py | 1 | 43| 337 | 31845 | 102679 | 
| 69 | 33 sklearn/cluster/_k_means.py | 915 | 982| 651 | 32496 | 102679 | 
| 70 | 34 sklearn/cluster/_bicluster.py | 277 | 305| 280 | 32776 | 107478 | 
| 71 | 35 examples/text/plot_document_clustering.py | 89 | 192| 790 | 33566 | 109349 | 
| 72 | 36 examples/decomposition/plot_faces_decomposition.py | 63 | 141| 623 | 34189 | 110796 | 
| 73 | 36 sklearn/cluster/_k_means.py | 1 | 34| 193 | 34382 | 110796 | 
| 74 | 37 examples/cluster/plot_agglomerative_dendrogram.py | 41 | 53| 104 | 34486 | 111159 | 
| 75 | 38 examples/cluster/plot_segmentation_toy.py | 1 | 102| 830 | 35316 | 112042 | 
| 76 | 38 examples/text/plot_document_clustering.py | 194 | 226| 296 | 35612 | 112042 | 
| 77 | 39 sklearn/cluster/_mean_shift.py | 349 | 461| 935 | 36547 | 115933 | 
| 78 | 39 sklearn/cluster/_k_means.py | 291 | 300| 184 | 36731 | 115933 | 
| 79 | 39 sklearn/cluster/_mean_shift.py | 86 | 106| 230 | 36961 | 115933 | 
| 80 | 39 examples/cluster/plot_optics.py | 1 | 74| 753 | 37714 | 115933 | 
| 81 | 39 sklearn/cluster/_bicluster.py | 437 | 464| 272 | 37986 | 115933 | 
| 82 | 40 examples/semi_supervised/plot_label_propagation_digits.py | 1 | 93| 585 | 38571 | 116538 | 
| 83 | 40 sklearn/metrics/cluster/_supervised.py | 136 | 224| 761 | 39332 | 116538 | 
| 84 | 40 sklearn/cluster/_k_means.py | 303 | 329| 305 | 39637 | 116538 | 
| 85 | 40 benchmarks/bench_rcv1_logreg_convergence.py | 34 | 64| 225 | 39862 | 116538 | 
| 86 | 40 examples/cluster/plot_mini_batch_kmeans.py | 1 | 86| 723 | 40585 | 116538 | 
| 87 | 40 sklearn/cluster/_k_means.py | 792 | 818| 258 | 40843 | 116538 | 
| 88 | 40 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 41216 | 116538 | 
| 89 | 40 sklearn/cluster/_k_means.py | 820 | 914| 822 | 42038 | 116538 | 
| 90 | 40 sklearn/utils/estimator_checks.py | 1860 | 1932| 655 | 42693 | 116538 | 
| 91 | 40 sklearn/metrics/cluster/_supervised.py | 325 | 392| 582 | 43275 | 116538 | 
| 92 | 40 sklearn/cluster/_hierarchical.py | 425 | 515| 822 | 44097 | 116538 | 
| 93 | 41 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 95 | 109| 174 | 44271 | 117522 | 
| 94 | 42 sklearn/metrics/cluster/_bicluster.py | 1 | 27| 228 | 44499 | 118239 | 
| 95 | 42 sklearn/cluster/_hierarchical.py | 216 | 289| 657 | 45156 | 118239 | 
| 96 | 42 sklearn/metrics/cluster/_supervised.py | 853 | 869| 229 | 45385 | 118239 | 
| 97 | 43 sklearn/metrics/cluster/_unsupervised.py | 1 | 35| 167 | 45552 | 121595 | 
| 98 | 44 sklearn/linear_model/_least_angle.py | 479 | 636| 1591 | 47143 | 137875 | 
| 99 | 45 examples/bicluster/plot_spectral_biclustering.py | 1 | 68| 441 | 47584 | 138342 | 
| 100 | 45 examples/cluster/plot_kmeans_digits.py | 59 | 71| 160 | 47744 | 138342 | 
| 101 | 45 sklearn/cluster/_spectral.py | 276 | 435| 1529 | 49273 | 138342 | 
| 102 | 45 sklearn/cluster/_k_means.py | 138 | 151| 143 | 49416 | 138342 | 
| 103 | 46 examples/calibration/plot_calibration.py | 1 | 82| 752 | 50168 | 139571 | 
| 104 | 47 examples/applications/plot_outlier_detection_housing.py | 79 | 134| 633 | 50801 | 140972 | 
| 105 | 47 sklearn/cluster/_spectral.py | 456 | 521| 607 | 51408 | 140972 | 
| 106 | 47 sklearn/cluster/_hierarchical.py | 81 | 129| 449 | 51857 | 140972 | 
| 107 | 48 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 52524 | 141639 | 
| 108 | 49 examples/neighbors/plot_lof_novelty_detection.py | 68 | 84| 177 | 52701 | 142569 | 
| 109 | 49 examples/cluster/plot_digits_linkage.py | 38 | 52| 150 | 52851 | 142569 | 
| 110 | 49 sklearn/cluster/_k_means.py | 1329 | 1465| 1221 | 54072 | 142569 | 
| 111 | 49 sklearn/cluster/_hierarchical.py | 902 | 1010| 1040 | 55112 | 142569 | 
| 112 | 50 examples/impute/plot_missing_values.py | 55 | 105| 512 | 55624 | 143917 | 
| 113 | 50 sklearn/cluster/_hierarchical.py | 516 | 577| 584 | 56208 | 143917 | 
| 114 | 50 sklearn/cluster/_spectral.py | 523 | 551| 227 | 56435 | 143917 | 
| 115 | 50 sklearn/metrics/cluster/_unsupervised.py | 231 | 247| 232 | 56667 | 143917 | 
| 116 | 51 sklearn/cluster/_birch.py | 447 | 496| 448 | 57115 | 149088 | 
| 117 | 51 examples/cluster/plot_adjusted_for_chance_measures.py | 33 | 55| 231 | 57346 | 149088 | 
| 118 | 52 examples/bicluster/plot_bicluster_newsgroups.py | 1 | 87| 737 | 58083 | 150438 | 
| 119 | 53 benchmarks/bench_plot_randomized_svd.py | 379 | 430| 540 | 58623 | 154815 | 
| 120 | 53 sklearn/cluster/_birch.py | 295 | 319| 269 | 58892 | 154815 | 
| 121 | 53 sklearn/utils/estimator_checks.py | 1368 | 1436| 594 | 59486 | 154815 | 
| 122 | 53 sklearn/utils/estimator_checks.py | 1 | 59| 434 | 59920 | 154815 | 
| 123 | 53 benchmarks/bench_plot_neighbors.py | 111 | 186| 663 | 60583 | 154815 | 
| 124 | 53 sklearn/cluster/_hierarchical.py | 1012 | 1044| 241 | 60824 | 154815 | 
| 125 | 54 sklearn/neighbors/_nca.py | 1 | 27| 136 | 60960 | 159036 | 
| 126 | 54 sklearn/cluster/_spectral.py | 106 | 156| 480 | 61440 | 159036 | 
| 127 | 54 sklearn/cluster/_k_means.py | 1190 | 1260| 767 | 62207 | 159036 | 
| 128 | 55 examples/model_selection/plot_cv_indices.py | 105 | 151| 388 | 62595 | 160358 | 
| 129 | 55 sklearn/utils/estimator_checks.py | 2092 | 2137| 493 | 63088 | 160358 | 
| 130 | 55 examples/calibration/plot_calibration.py | 84 | 120| 401 | 63489 | 160358 | 
| 131 | 55 sklearn/cluster/_bicluster.py | 177 | 276| 992 | 64481 | 160358 | 
| 132 | 55 sklearn/cluster/_spectral.py | 437 | 454| 199 | 64680 | 160358 | 
| 133 | 56 sklearn/metrics/cluster/__init__.py | 1 | 35| 371 | 65051 | 160729 | 
| 134 | 56 sklearn/cluster/_hierarchical.py | 28 | 78| 411 | 65462 | 160729 | 
| 135 | 56 sklearn/cluster/_k_means.py | 1467 | 1480| 169 | 65631 | 160729 | 
| 136 | 57 benchmarks/bench_plot_fastkmeans.py | 92 | 137| 495 | 66126 | 161937 | 
| 137 | 58 sklearn/impute/_iterative.py | 2 | 709| 167 | 66293 | 168107 | 
| 138 | 59 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 61| 494 | 66787 | 169122 | 
| 139 | 60 benchmarks/bench_tree.py | 64 | 125| 523 | 67310 | 169992 | 
| 140 | 61 benchmarks/bench_isolation_forest.py | 54 | 160| 1025 | 68335 | 171453 | 
| 141 | 62 examples/cluster/plot_birch_vs_minibatchkmeans.py | 87 | 103| 182 | 68517 | 172441 | 
| 142 | 62 sklearn/cluster/_bicluster.py | 123 | 174| 526 | 69043 | 172441 | 
| 143 | 63 benchmarks/bench_lof.py | 36 | 107| 650 | 69693 | 173357 | 
| 144 | 64 examples/text/plot_document_classification_20newsgroups.py | 249 | 329| 673 | 70366 | 175957 | 
| 145 | 64 sklearn/cluster/_birch.py | 536 | 566| 234 | 70600 | 175957 | 
| 146 | 64 examples/cluster/plot_agglomerative_dendrogram.py | 11 | 38| 197 | 70797 | 175957 | 
| 147 | 64 sklearn/cluster/_optics.py | 899 | 926| 211 | 71008 | 175957 | 
| 148 | 65 sklearn/neighbors/_classification.py | 493 | 579| 764 | 71772 | 180952 | 
| 149 | 66 examples/classification/plot_lda.py | 38 | 69| 325 | 72097 | 181536 | 
| 150 | 66 sklearn/metrics/cluster/_supervised.py | 565 | 648| 828 | 72925 | 181536 | 
| 151 | 67 sklearn/utils/fixes.py | 56 | 155| 734 | 73659 | 183828 | 
| 152 | 67 sklearn/utils/estimator_checks.py | 2311 | 2347| 375 | 74034 | 183828 | 
| 153 | 68 examples/neighbors/plot_nearest_centroid.py | 1 | 59| 495 | 74529 | 184323 | 
| 154 | 69 sklearn/covariance/_robust_covariance.py | 96 | 180| 804 | 75333 | 191391 | 
| 155 | 69 sklearn/metrics/cluster/_supervised.py | 395 | 462| 561 | 75894 | 191391 | 
| 156 | 70 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 76679 | 193071 | 
| 157 | 70 sklearn/cluster/_hierarchical.py | 881 | 899| 141 | 76820 | 193071 | 
| 158 | 70 sklearn/cluster/_bicluster.py | 466 | 503| 338 | 77158 | 193071 | 
| 159 | 70 sklearn/cluster/_optics.py | 505 | 537| 348 | 77506 | 193071 | 
| 160 | 71 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 78020 | 196963 | 
| 161 | 71 sklearn/cluster/_mean_shift.py | 1 | 26| 163 | 78183 | 196963 | 


### Hint

```
@JenniferHemmerich this affinity propagation code is not often updated. If you have time to improve its documentation and fix corner cases like the one you report please send us PR. I'll try to find the time to review the changes. thanks
Working on this for the wmlds scikit learn sprint (pair programming with @akeshavan)
```

## Patch

```diff
diff --git a/sklearn/cluster/_affinity_propagation.py b/sklearn/cluster/_affinity_propagation.py
--- a/sklearn/cluster/_affinity_propagation.py
+++ b/sklearn/cluster/_affinity_propagation.py
@@ -194,17 +194,19 @@ def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
             unconverged = (np.sum((se == convergence_iter) + (se == 0))
                            != n_samples)
             if (not unconverged and (K > 0)) or (it == max_iter):
+                never_converged = False
                 if verbose:
                     print("Converged after %d iterations." % it)
                 break
     else:
+        never_converged = True
         if verbose:
             print("Did not converge")
 
     I = np.flatnonzero(E)
     K = I.size  # Identify exemplars
 
-    if K > 0:
+    if K > 0 and not never_converged:
         c = np.argmax(S[:, I], axis=1)
         c[I] = np.arange(K)  # Identify clusters
         # Refine the final set of exemplars and clusters and return results
@@ -408,6 +410,7 @@ def predict(self, X):
             Cluster labels.
         """
         check_is_fitted(self)
+        X = check_array(X)
         if not hasattr(self, "cluster_centers_"):
             raise ValueError("Predict method is not supported when "
                              "affinity='precomputed'.")

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_affinity_propagation.py b/sklearn/cluster/tests/test_affinity_propagation.py
--- a/sklearn/cluster/tests/test_affinity_propagation.py
+++ b/sklearn/cluster/tests/test_affinity_propagation.py
@@ -152,6 +152,14 @@ def test_affinity_propagation_predict_non_convergence():
     assert_array_equal(np.array([-1, -1, -1]), y)
 
 
+def test_affinity_propagation_non_convergence_regressiontest():
+    X = np.array([[1, 0, 0, 0, 0, 0],
+                  [0, 1, 1, 1, 0, 0],
+                  [0, 0, 1, 0, 0, 1]])
+    af = AffinityPropagation(affinity='euclidean', max_iter=2).fit(X)
+    assert_array_equal(np.array([-1, -1, -1]), af.labels_)
+
+
 def test_equal_similarities_and_preferences():
     # Unequal distances
     X = np.array([[0, 0], [1, 1], [-2, -2]])

```


## Code snippets

### 1 - sklearn/cluster/_affinity_propagation.py:

Start line: 207, End line: 231

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    # ... other code

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn("Affinity propagation did not converge, this model "
                      "will not have any cluster centers.", ConvergenceWarning)
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []

    if return_n_iter:
        return cluster_centers_indices, labels, it + 1
    else:
        return cluster_centers_indices, labels
```
### 2 - sklearn/cluster/_affinity_propagation.py:

Start line: 121, End line: 205

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    # ... other code

    if (n_samples == 1 or
            _equal_similarities_and_preferences(S, preference)):
        # It makes no sense to run the algorithm in this case, so return 1 or
        # n_samples clusters, depending on preferences
        warnings.warn("All samples have mutually equal similarities. "
                      "Returning arbitrary cluster center(s).")
        if preference.flat[0] >= S.flat[n_samples - 1]:
            return ((np.arange(n_samples), np.arange(n_samples), 0)
                    if return_n_iter
                    else (np.arange(n_samples), np.arange(n_samples)))
        else:
            return ((np.array([0]), np.array([0] * n_samples), 0)
                    if return_n_iter
                    else (np.array([0]), np.array([0] * n_samples)))

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages
    # Intermediate results
    tmp = np.zeros((n_samples, n_samples))

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # tmp = A + S; compute responsibilities
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)

        # tmp = Rnew
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2

        # Damping
        tmp *= 1 - damping
        R *= damping
        R += tmp

        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA

        # Damping
        tmp *= 1 - damping
        A *= damping
        A -= tmp

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.flatnonzero(E)
    K = I.size  # Identify exemplars
    # ... other code
```
### 3 - sklearn/cluster/_affinity_propagation.py:

Start line: 234, End line: 334

```python
###############################################################################

class AffinityPropagation(ClusterMixin, BaseEstimator):
    """Perform Affinity Propagation Clustering of data.

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------
    damping : float, optional, default: 0.5
        Damping factor (between 0.5 and 1) is the extent to
        which the current value is maintained relative to
        incoming values (weighted 1 - damping). This in order
        to avoid numerical oscillations when updating these
        values (messages).

    max_iter : int, optional, default: 200
        Maximum number of iterations.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    copy : boolean, optional, default: True
        Make a copy of input data.

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number
        of exemplars, ie of clusters, is influenced by the input
        preferences value. If the preferences are not passed as arguments,
        they will be set to the median of the input similarities.

    affinity : string, optional, default=``euclidean``
        Which affinity to use. At the moment ``precomputed`` and
        ``euclidean`` are supported. ``euclidean`` uses the
        negative squared euclidean distance between points.

    verbose : boolean, optional, default: False
        Whether to be verbose.


    Attributes
    ----------
    cluster_centers_indices_ : array, shape (n_clusters,)
        Indices of cluster centers

    cluster_centers_ : array, shape (n_clusters, n_features)
        Cluster centers (if affinity != ``precomputed``).

    labels_ : array, shape (n_samples,)
        Labels of each point

    affinity_matrix_ : array, shape (n_samples, n_samples)
        Stores the affinity matrix used in ``fit``.

    n_iter_ : int
        Number of iterations taken to converge.

    Examples
    --------
    >>> from sklearn.cluster import AffinityPropagation
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AffinityPropagation().fit(X)
    >>> clustering
    AffinityPropagation()
    >>> clustering.labels_
    array([0, 0, 0, 1, 1, 1])
    >>> clustering.predict([[0, 0], [4, 4]])
    array([0, 1])
    >>> clustering.cluster_centers_
    array([[1, 2],
           [4, 2]])

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    The algorithmic complexity of affinity propagation is quadratic
    in the number of points.

    When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
    array and all training samples will be labelled as ``-1``. In addition,
    ``predict`` will then label every sample as ``-1``.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, ``fit`` will result in
    a single cluster center and label ``0`` for every sample. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------

    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
```
### 4 - examples/cluster/plot_affinity_propagation.py:

Start line: 1, End line: 63

```python
"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```
### 5 - sklearn/utils/estimator_checks.py:

Start line: 1608, End line: 1664

```python
@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])

    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)
    if name == 'AffinityPropagation':
        clusterer.set_params(preference=-100)
        clusterer.set_params(max_iter=100)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert pred.shape == (n_samples,)
    assert adjusted_rand_score(pred, y) > 0.4
    if _safe_tags(clusterer, 'non_deterministic'):
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype('int32'), np.dtype('int64')]
    assert pred2.dtype in [np.dtype('int32'), np.dtype('int64')]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(labels_sorted[0],
                                                labels_sorted[-1] + 1))

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true
```
### 6 - sklearn/cluster/_affinity_propagation.py:

Start line: 336, End line: 394

```python
class AffinityPropagation(ClusterMixin, BaseEstimator):

    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"

    def fit(self, X, y=None):
        """Fit the clustering from features, or affinity matrix.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            array-like, shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self

        """
        if self.affinity == "precomputed":
            accept_sparse = False
        else:
            accept_sparse = 'csr'
        X = check_array(X, accept_sparse=accept_sparse)
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=True)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.cluster_centers_indices_, self.labels_, self.n_iter_ = \
            affinity_propagation(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose, return_n_iter=True)

        if self.affinity != "precomputed":
            self.cluster_centers_ = X[self.cluster_centers_indices_].copy()

        return self
```
### 7 - sklearn/cluster/_affinity_propagation.py:

Start line: 33, End line: 119

```python
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False,
                         return_n_iter=False):
    """Perform Affinity Propagation Clustering of data

    Read more in the :ref:`User Guide <affinity_propagation>`.

    Parameters
    ----------

    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations

    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    verbose : boolean, optional, default: False
        The verbosity level

    return_n_iter : bool, default False
        Whether or not to return the number of iterations.

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    n_iter : int
        number of iterations run. Returned only if `return_n_iter` is
        set to True.

    Notes
    -----
    For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
    <sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

    When the algorithm does not converge, it returns an empty array as
    ``cluster_center_indices`` and ``-1`` as label for each training sample.

    When all training samples have equal similarities and equal preferences,
    the assignment of cluster centers and labels depends on the preference.
    If the preference is smaller than the similarities, a single cluster center
    and label ``0`` for every sample will be returned. Otherwise, every
    training sample becomes its own cluster center and is assigned a unique
    label.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    preference = np.array(preference)
    # ... other code
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
### 9 - sklearn/cluster/_affinity_propagation.py:

Start line: 1, End line: 30

```python
"""Affinity Propagation clustering algorithm."""

import numpy as np
import warnings

from ..exceptions import ConvergenceWarning
from ..base import BaseEstimator, ClusterMixin
from ..utils import as_float_array, check_array
from ..utils.validation import check_is_fitted
from ..metrics import euclidean_distances
from ..metrics import pairwise_distances_argmin


def _equal_similarities_and_preferences(S, preference):
    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        # Create mask to ignore diagonal of S
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)

        return np.all(S[mask].flat == S[mask].flat[0])

    return all_equal_preferences() and all_equal_similarities()
```
### 10 - examples/cluster/plot_cluster_iris.py:

Start line: 1, End line: 93

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler

import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()
```
### 13 - sklearn/cluster/_affinity_propagation.py:

Start line: 423, End line: 444

```python
class AffinityPropagation(ClusterMixin, BaseEstimator):

    def fit_predict(self, X, y=None):
        """Fit the clustering from features or affinity matrix, and return
        cluster labels.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features), or \
            array-like, shape (n_samples, n_samples)
            Training instances to cluster, or similarities / affinities between
            instances if ``affinity='precomputed'``. If a sparse feature matrix
            is provided, it will be converted into a sparse ``csr_matrix``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
```
### 17 - sklearn/cluster/_affinity_propagation.py:

Start line: 396, End line: 421

```python
class AffinityPropagation(ClusterMixin, BaseEstimator):

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            New data to predict. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            Cluster labels.
        """
        check_is_fitted(self)
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Predict method is not supported when "
                             "affinity='precomputed'.")

        if self.cluster_centers_.shape[0] > 0:
            return pairwise_distances_argmin(X, self.cluster_centers_)
        else:
            warnings.warn("This model does not have any cluster centers "
                          "because affinity propagation did not converge. "
                          "Labeling every sample as '-1'.", ConvergenceWarning)
            return np.array([-1] * X.shape[0])
```

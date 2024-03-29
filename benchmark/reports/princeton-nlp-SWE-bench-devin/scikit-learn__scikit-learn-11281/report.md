# scikit-learn__scikit-learn-11281

| **scikit-learn/scikit-learn** | `4143356c3c51831300789e4fdf795d83716dbab6` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 32275 |
| **Any found context length** | 32275 |
| **Avg pos** | 270.0 |
| **Min pos** | 90 |
| **Max pos** | 90 |
| **Top file pos** | 4 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/mixture/base.py b/sklearn/mixture/base.py
--- a/sklearn/mixture/base.py
+++ b/sklearn/mixture/base.py
@@ -172,7 +172,7 @@ def _initialize(self, X, resp):
     def fit(self, X, y=None):
         """Estimate model parameters with the EM algorithm.
 
-        The method fit the model `n_init` times and set the parameters with
+        The method fits the model `n_init` times and set the parameters with
         which the model has the largest likelihood or lower bound. Within each
         trial, the method iterates between E-step and M-step for `max_iter`
         times until the change of likelihood or lower bound is less than
@@ -188,6 +188,32 @@ def fit(self, X, y=None):
         -------
         self
         """
+        self.fit_predict(X, y)
+        return self
+
+    def fit_predict(self, X, y=None):
+        """Estimate model parameters using X and predict the labels for X.
+
+        The method fits the model n_init times and sets the parameters with
+        which the model has the largest likelihood or lower bound. Within each
+        trial, the method iterates between E-step and M-step for `max_iter`
+        times until the change of likelihood or lower bound is less than
+        `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
+        predicts the most probable label for the input data points.
+
+        .. versionadded:: 0.20
+
+        Parameters
+        ----------
+        X : array-like, shape (n_samples, n_features)
+            List of n_features-dimensional data points. Each row
+            corresponds to a single data point.
+
+        Returns
+        -------
+        labels : array, shape (n_samples,)
+            Component labels.
+        """
         X = _check_X(X, self.n_components, ensure_min_samples=2)
         self._check_initial_parameters(X)
 
@@ -240,7 +266,7 @@ def fit(self, X, y=None):
         self._set_parameters(best_params)
         self.n_iter_ = best_n_iter
 
-        return self
+        return log_resp.argmax(axis=1)
 
     def _e_step(self, X):
         """E step.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/mixture/base.py | 175 | 175 | 90 | 4 | 32275
| sklearn/mixture/base.py | 191 | 191 | 90 | 4 | 32275
| sklearn/mixture/base.py | 243 | 243 | 90 | 4 | 32275


## Problem Statement

```
Should mixture models have a clusterer-compatible interface
Mixture models are currently a bit different. They are basically clusterers, except they are probabilistic, and are applied to inductive problems unlike many clusterers. But they are unlike clusterers in API:
* they have an `n_components` parameter, with identical purpose to `n_clusters`
* they do not store the `labels_` of the training data
* they do not have a `fit_predict` method

And they are almost entirely documented separately.

Should we make the MMs more like clusterers?

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/mixture/gaussian_mixture.py | 435 | 581| 1312 | 1312 | 6261 | 
| 2 | 2 sklearn/mixture/gmm.py | 133 | 257| 1207 | 2519 | 13825 | 
| 3 | 3 sklearn/mixture/bayesian_mixture.py | 66 | 307| 2354 | 4873 | 21097 | 
| 4 | **4 sklearn/mixture/base.py** | 67 | 86| 170 | 5043 | 24668 | 
| 5 | 5 sklearn/mixture/__init__.py | 1 | 23| 167 | 5210 | 24835 | 
| 6 | 6 sklearn/mixture/dpgmm.py | 128 | 219| 781 | 5991 | 33335 | 
| 7 | 6 sklearn/mixture/dpgmm.py | 1 | 44| 261 | 6252 | 33335 | 
| 8 | 6 sklearn/mixture/gmm.py | 1 | 72| 502 | 6754 | 33335 | 
| 9 | 7 examples/mixture/plot_gmm_selection.py | 1 | 77| 657 | 7411 | 34226 | 
| 10 | 7 sklearn/mixture/gmm.py | 490 | 581| 862 | 8273 | 34226 | 
| 11 | 7 sklearn/mixture/gmm.py | 281 | 305| 253 | 8526 | 34226 | 
| 12 | 7 sklearn/mixture/dpgmm.py | 549 | 619| 772 | 9298 | 34226 | 
| 13 | 7 sklearn/mixture/dpgmm.py | 299 | 306| 123 | 9421 | 34226 | 
| 14 | 8 examples/mixture/plot_gmm.py | 1 | 36| 289 | 9710 | 35059 | 
| 15 | 8 sklearn/mixture/gmm.py | 667 | 687| 219 | 9929 | 35059 | 
| 16 | 8 sklearn/mixture/bayesian_mixture.py | 751 | 786| 363 | 10292 | 35059 | 
| 17 | 8 sklearn/mixture/bayesian_mixture.py | 309 | 329| 270 | 10562 | 35059 | 
| 18 | 8 sklearn/mixture/gmm.py | 605 | 621| 203 | 10765 | 35059 | 
| 19 | 8 sklearn/mixture/gaussian_mixture.py | 1 | 51| 326 | 11091 | 35059 | 
| 20 | 8 sklearn/mixture/dpgmm.py | 648 | 741| 799 | 11890 | 35059 | 
| 21 | 8 sklearn/mixture/gmm.py | 259 | 279| 212 | 12102 | 35059 | 
| 22 | **8 sklearn/mixture/base.py** | 1 | 38| 189 | 12291 | 35059 | 
| 23 | 8 sklearn/mixture/gaussian_mixture.py | 691 | 707| 196 | 12487 | 35059 | 
| 24 | 8 sklearn/mixture/dpgmm.py | 622 | 645| 289 | 12776 | 35059 | 
| 25 | 8 sklearn/mixture/gaussian_mixture.py | 583 | 597| 197 | 12973 | 35059 | 
| 26 | 8 sklearn/mixture/bayesian_mixture.py | 687 | 700| 177 | 13150 | 35059 | 
| 27 | 8 examples/mixture/plot_gmm.py | 68 | 89| 235 | 13385 | 35059 | 
| 28 | 9 examples/mixture/plot_gmm_sin.py | 103 | 155| 644 | 14029 | 36633 | 
| 29 | 9 sklearn/mixture/gaussian_mixture.py | 674 | 689| 148 | 14177 | 36633 | 
| 30 | 9 sklearn/mixture/bayesian_mixture.py | 649 | 672| 219 | 14396 | 36633 | 
| 31 | **9 sklearn/mixture/base.py** | 264 | 288| 150 | 14546 | 36633 | 
| 32 | 9 sklearn/mixture/dpgmm.py | 819 | 842| 339 | 14885 | 36633 | 
| 33 | 9 sklearn/mixture/gmm.py | 346 | 392| 286 | 15171 | 36633 | 
| 34 | 9 sklearn/mixture/gmm.py | 307 | 344| 318 | 15489 | 36633 | 
| 35 | 9 sklearn/mixture/dpgmm.py | 743 | 750| 211 | 15700 | 36633 | 
| 36 | 9 sklearn/mixture/dpgmm.py | 308 | 325| 249 | 15949 | 36633 | 
| 37 | 9 examples/mixture/plot_gmm_sin.py | 1 | 54| 464 | 16413 | 36633 | 
| 38 | 9 sklearn/mixture/dpgmm.py | 247 | 297| 510 | 16923 | 36633 | 
| 39 | 10 examples/mixture/plot_gmm_covariances.py | 68 | 135| 599 | 17522 | 37850 | 
| 40 | 10 examples/mixture/plot_gmm_covariances.py | 1 | 44| 282 | 17804 | 37850 | 
| 41 | 11 sklearn/utils/estimator_checks.py | 1183 | 1241| 600 | 18404 | 57487 | 
| 42 | 11 sklearn/mixture/gaussian_mixture.py | 599 | 620| 211 | 18615 | 57487 | 
| 43 | 11 sklearn/mixture/gaussian_mixture.py | 709 | 721| 171 | 18786 | 57487 | 
| 44 | 11 sklearn/mixture/gmm.py | 637 | 664| 177 | 18963 | 57487 | 
| 45 | 11 sklearn/mixture/bayesian_mixture.py | 1 | 18| 142 | 19105 | 57487 | 
| 46 | 11 sklearn/mixture/gaussian_mixture.py | 723 | 751| 181 | 19286 | 57487 | 
| 47 | 12 examples/mixture/plot_gmm_pdf.py | 1 | 51| 398 | 19684 | 57885 | 
| 48 | 12 sklearn/mixture/dpgmm.py | 473 | 500| 282 | 19966 | 57885 | 
| 49 | **12 sklearn/mixture/base.py** | 343 | 361| 165 | 20131 | 57885 | 
| 50 | 13 sklearn/base.py | 379 | 400| 131 | 20262 | 62178 | 
| 51 | 13 sklearn/mixture/gaussian_mixture.py | 622 | 653| 309 | 20571 | 62178 | 
| 52 | 13 sklearn/mixture/bayesian_mixture.py | 674 | 685| 154 | 20725 | 62178 | 
| 53 | 13 sklearn/mixture/gaussian_mixture.py | 655 | 672| 174 | 20899 | 62178 | 
| 54 | 13 sklearn/mixture/gmm.py | 623 | 635| 173 | 21072 | 62178 | 
| 55 | 13 sklearn/mixture/bayesian_mixture.py | 488 | 500| 119 | 21191 | 62178 | 
| 56 | 13 sklearn/mixture/dpgmm.py | 776 | 817| 433 | 21624 | 62178 | 
| 57 | 13 sklearn/mixture/bayesian_mixture.py | 369 | 393| 207 | 21831 | 62178 | 
| 58 | 13 sklearn/mixture/bayesian_mixture.py | 454 | 468| 125 | 21956 | 62178 | 
| 59 | 13 sklearn/mixture/gmm.py | 436 | 453| 138 | 22094 | 62178 | 
| 60 | 14 examples/mixture/plot_concentration_prior.py | 1 | 40| 331 | 22425 | 63640 | 
| 61 | 14 sklearn/mixture/dpgmm.py | 327 | 380| 804 | 23229 | 63640 | 
| 62 | 14 examples/mixture/plot_gmm_selection.py | 78 | 99| 234 | 23463 | 63640 | 
| 63 | 14 sklearn/mixture/bayesian_mixture.py | 331 | 355| 236 | 23699 | 63640 | 
| 64 | 14 sklearn/mixture/dpgmm.py | 220 | 245| 282 | 23981 | 63640 | 
| 65 | 14 sklearn/mixture/bayesian_mixture.py | 357 | 367| 123 | 24104 | 63640 | 
| 66 | **14 sklearn/mixture/base.py** | 124 | 170| 324 | 24428 | 63640 | 
| 67 | 15 examples/cluster/plot_cluster_comparison.py | 1 | 86| 698 | 25126 | 65194 | 
| 68 | **15 sklearn/mixture/base.py** | 451 | 484| 294 | 25420 | 65194 | 
| 69 | 15 sklearn/mixture/bayesian_mixture.py | 470 | 486| 166 | 25586 | 65194 | 
| 70 | 15 sklearn/mixture/gmm.py | 720 | 743| 283 | 25869 | 65194 | 
| 71 | 15 sklearn/mixture/gmm.py | 394 | 434| 346 | 26215 | 65194 | 
| 72 | 16 sklearn/cluster/__init__.py | 1 | 37| 284 | 26499 | 65478 | 
| 73 | **16 sklearn/mixture/base.py** | 245 | 262| 149 | 26648 | 65478 | 
| 74 | 16 sklearn/mixture/gmm.py | 111 | 130| 174 | 26822 | 65478 | 
| 75 | 16 sklearn/mixture/gmm.py | 784 | 801| 206 | 27028 | 65478 | 
| 76 | 16 sklearn/mixture/dpgmm.py | 502 | 547| 400 | 27428 | 65478 | 
| 77 | 16 sklearn/mixture/gmm.py | 455 | 488| 281 | 27709 | 65478 | 
| 78 | 16 sklearn/mixture/dpgmm.py | 752 | 774| 278 | 27987 | 65478 | 
| 79 | 16 sklearn/mixture/bayesian_mixture.py | 395 | 411| 169 | 28156 | 65478 | 
| 80 | 17 sklearn/cluster/mean_shift_.py | 1 | 28| 180 | 28336 | 69048 | 
| 81 | 17 sklearn/mixture/bayesian_mixture.py | 620 | 647| 260 | 28596 | 69048 | 
| 82 | **17 sklearn/mixture/base.py** | 88 | 122| 289 | 28885 | 69048 | 
| 83 | 18 sklearn/cluster/k_means_.py | 1165 | 1234| 759 | 29644 | 83633 | 
| 84 | 18 examples/mixture/plot_gmm.py | 39 | 65| 309 | 29953 | 83633 | 
| 85 | 18 sklearn/mixture/bayesian_mixture.py | 413 | 452| 419 | 30372 | 83633 | 
| 86 | 18 sklearn/mixture/dpgmm.py | 109 | 125| 208 | 30580 | 83633 | 
| 87 | 18 sklearn/mixture/gmm.py | 704 | 717| 159 | 30739 | 83633 | 
| 88 | 18 examples/mixture/plot_concentration_prior.py | 87 | 137| 571 | 31310 | 83633 | 
| 89 | 19 sklearn/cluster/_feature_agglomeration.py | 1 | 59| 390 | 31700 | 84204 | 
| **-> 90 <-** | **19 sklearn/mixture/base.py** | 172 | 243| 575 | 32275 | 84204 | 
| 91 | 19 sklearn/base.py | 403 | 434| 222 | 32497 | 84204 | 
| 92 | 20 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 33008 | 84737 | 
| 93 | 21 sklearn/cluster/bicluster.py | 416 | 443| 279 | 33287 | 89298 | 
| 94 | 21 sklearn/cluster/k_means_.py | 1498 | 1575| 730 | 34017 | 89298 | 
| 95 | 21 sklearn/mixture/bayesian_mixture.py | 702 | 749| 470 | 34487 | 89298 | 
| 96 | 21 sklearn/mixture/bayesian_mixture.py | 502 | 525| 239 | 34726 | 89298 | 
| 97 | 21 sklearn/utils/estimator_checks.py | 215 | 238| 203 | 34929 | 89298 | 
| 98 | 21 sklearn/mixture/bayesian_mixture.py | 561 | 589| 282 | 35211 | 89298 | 
| 99 | 22 sklearn/datasets/samples_generator.py | 156 | 222| 778 | 35989 | 103310 | 
| 100 | 22 sklearn/cluster/bicluster.py | 7 | 27| 132 | 36121 | 103310 | 
| 101 | 22 sklearn/mixture/dpgmm.py | 413 | 435| 314 | 36435 | 103310 | 
| 102 | 22 sklearn/mixture/dpgmm.py | 844 | 860| 233 | 36668 | 103310 | 
| 103 | 22 sklearn/cluster/bicluster.py | 300 | 414| 147 | 36815 | 103310 | 
| 104 | 23 sklearn/cluster/hierarchical.py | 10 | 27| 126 | 36941 | 111400 | 
| 105 | 24 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 123| 542 | 37483 | 112393 | 
| 106 | 25 sklearn/metrics/cluster/supervised.py | 687 | 704| 232 | 37715 | 120316 | 
| 107 | 25 sklearn/metrics/cluster/supervised.py | 530 | 608| 800 | 38515 | 120316 | 
| 108 | 26 examples/cluster/plot_mini_batch_kmeans.py | 86 | 116| 312 | 38827 | 121364 | 
| 109 | 26 sklearn/cluster/k_means_.py | 904 | 930| 258 | 39085 | 121364 | 
| 110 | 26 sklearn/cluster/bicluster.py | 269 | 297| 286 | 39371 | 121364 | 
| 111 | 26 sklearn/mixture/gmm.py | 837 | 854| 189 | 39560 | 121364 | 
| 112 | 26 sklearn/mixture/dpgmm.py | 460 | 471| 176 | 39736 | 121364 | 
| 113 | 26 sklearn/mixture/gaussian_mixture.py | 54 | 74| 133 | 39869 | 121364 | 
| 114 | 26 sklearn/mixture/gmm.py | 583 | 603| 159 | 40028 | 121364 | 
| 115 | 27 sklearn/metrics/cluster/__init__.py | 1 | 33| 347 | 40375 | 121711 | 
| 116 | 27 sklearn/mixture/dpgmm.py | 395 | 411| 172 | 40547 | 121711 | 
| 117 | 27 sklearn/mixture/gmm.py | 819 | 834| 189 | 40736 | 121711 | 
| 118 | 27 sklearn/metrics/cluster/supervised.py | 1 | 31| 130 | 40866 | 121711 | 
| 119 | 28 sklearn/cluster/spectral.py | 409 | 425| 187 | 41053 | 125978 | 
| 120 | 28 sklearn/mixture/gmm.py | 689 | 701| 138 | 41191 | 125978 | 
| 121 | 28 sklearn/cluster/k_means_.py | 771 | 902| 1177 | 42368 | 125978 | 
| 122 | 28 sklearn/mixture/gmm.py | 804 | 816| 163 | 42531 | 125978 | 
| 123 | 28 sklearn/cluster/hierarchical.py | 738 | 826| 759 | 43290 | 125978 | 
| 124 | **28 sklearn/mixture/base.py** | 412 | 449| 251 | 43541 | 125978 | 
| 125 | 29 sklearn/cluster/birch.py | 540 | 553| 140 | 43681 | 131269 | 
| 126 | 29 sklearn/cluster/bicluster.py | 445 | 482| 336 | 44017 | 131269 | 
| 127 | 29 sklearn/metrics/cluster/supervised.py | 707 | 784| 735 | 44752 | 131269 | 
| 128 | 30 sklearn/naive_bayes.py | 630 | 709| 730 | 45482 | 139678 | 
| 129 | 30 sklearn/mixture/gaussian_mixture.py | 339 | 378| 320 | 45802 | 139678 | 
| 130 | 30 sklearn/utils/estimator_checks.py | 1244 | 1258| 146 | 45948 | 139678 | 
| 131 | 30 examples/mixture/plot_gmm_sin.py | 86 | 100| 152 | 46100 | 139678 | 
| 132 | 30 sklearn/mixture/bayesian_mixture.py | 527 | 559| 316 | 46416 | 139678 | 
| 133 | 30 sklearn/cluster/hierarchical.py | 654 | 736| 726 | 47142 | 139678 | 
| 134 | 30 sklearn/cluster/hierarchical.py | 907 | 938| 239 | 47381 | 139678 | 
| 135 | 30 examples/mixture/plot_gmm_sin.py | 57 | 83| 314 | 47695 | 139678 | 
| 136 | 30 sklearn/naive_bayes.py | 839 | 908| 705 | 48400 | 139678 | 
| 137 | **30 sklearn/mixture/base.py** | 290 | 307| 152 | 48552 | 139678 | 
| 138 | 30 sklearn/cluster/hierarchical.py | 829 | 905| 744 | 49296 | 139678 | 
| 139 | 30 sklearn/cluster/birch.py | 324 | 426| 990 | 50286 | 139678 | 
| 140 | 30 sklearn/base.py | 512 | 550| 230 | 50516 | 139678 | 
| 141 | 30 sklearn/cluster/k_means_.py | 540 | 580| 407 | 50923 | 139678 | 
| 142 | 31 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 51499 | 140813 | 
| 143 | 31 sklearn/cluster/k_means_.py | 107 | 142| 330 | 51829 | 140813 | 
| 144 | 31 sklearn/cluster/birch.py | 451 | 500| 447 | 52276 | 140813 | 
| 145 | 32 sklearn/linear_model/base.py | 270 | 309| 339 | 52615 | 145781 | 
| 146 | 32 sklearn/mixture/dpgmm.py | 382 | 393| 130 | 52745 | 145781 | 
| 147 | 32 sklearn/cluster/k_means_.py | 1412 | 1425| 176 | 52921 | 145781 | 
| 148 | 33 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 53427 | 146840 | 
| 149 | 33 sklearn/mixture/bayesian_mixture.py | 591 | 618| 270 | 53697 | 146840 | 
| 150 | 33 sklearn/metrics/cluster/supervised.py | 787 | 860| 762 | 54459 | 146840 | 
| 151 | 33 sklearn/metrics/cluster/supervised.py | 611 | 686| 751 | 55210 | 146840 | 
| 152 | **33 sklearn/mixture/base.py** | 363 | 410| 389 | 55599 | 146840 | 
| 153 | 33 sklearn/base.py | 553 | 607| 281 | 55880 | 146840 | 
| 154 | 33 sklearn/mixture/gmm.py | 746 | 781| 433 | 56313 | 146840 | 
| 155 | **33 sklearn/mixture/base.py** | 486 | 504| 235 | 56548 | 146840 | 
| 156 | 33 sklearn/cluster/bicluster.py | 89 | 126| 313 | 56861 | 146840 | 
| 157 | 34 sklearn/multiclass.py | 1 | 63| 489 | 57350 | 153327 | 
| 158 | 34 sklearn/cluster/k_means_.py | 371 | 422| 571 | 57921 | 153327 | 
| 159 | 35 examples/cluster/plot_agglomerative_clustering_metrics.py | 92 | 130| 363 | 58284 | 154445 | 
| 160 | 35 sklearn/cluster/mean_shift_.py | 379 | 423| 310 | 58594 | 154445 | 
| 161 | 35 examples/cluster/plot_cluster_comparison.py | 88 | 186| 856 | 59450 | 154445 | 
| 162 | 36 sklearn/manifold/mds.py | 355 | 370| 150 | 59600 | 158272 | 
| 163 | 37 examples/covariance/plot_outlier_detection.py | 77 | 130| 632 | 60232 | 159491 | 
| 164 | 37 sklearn/cluster/k_means_.py | 1303 | 1410| 866 | 61098 | 159491 | 
| 165 | 37 sklearn/cluster/birch.py | 297 | 321| 270 | 61368 | 159491 | 
| 166 | **37 sklearn/mixture/base.py** | 309 | 341| 244 | 61612 | 159491 | 
| 167 | 37 sklearn/cluster/birch.py | 598 | 639| 373 | 61985 | 159491 | 
| 168 | 37 sklearn/multiclass.py | 219 | 273| 509 | 62494 | 159491 | 
| 169 | 38 sklearn/cluster/affinity_propagation_.py | 234 | 317| 680 | 63174 | 162674 | 
| 170 | 38 sklearn/cluster/k_means_.py | 1 | 37| 206 | 63380 | 162674 | 
| 171 | 39 examples/cluster/plot_birch_vs_minibatchkmeans.py | 1 | 87| 769 | 64149 | 163671 | 
| 172 | 39 sklearn/mixture/dpgmm.py | 47 | 58| 125 | 64274 | 163671 | 
| 173 | 39 sklearn/multiclass.py | 515 | 565| 425 | 64699 | 163671 | 
| 174 | 39 sklearn/cluster/spectral.py | 427 | 480| 462 | 65161 | 163671 | 
| 175 | 39 sklearn/cluster/spectral.py | 250 | 272| 248 | 65409 | 163671 | 
| 176 | 39 sklearn/cluster/k_means_.py | 1013 | 1037| 189 | 65598 | 163671 | 
| 177 | 39 examples/cluster/plot_adjusted_for_chance_measures.py | 1 | 31| 193 | 65791 | 163671 | 
| 178 | 39 sklearn/utils/estimator_checks.py | 1 | 80| 686 | 66477 | 163671 | 
| 179 | 39 sklearn/cluster/spectral.py | 275 | 407| 1221 | 67698 | 163671 | 
| 180 | 39 sklearn/cluster/k_means_.py | 1577 | 1607| 246 | 67944 | 163671 | 
| 181 | 39 sklearn/cluster/k_means_.py | 1237 | 1300| 659 | 68603 | 163671 | 
| 182 | 39 examples/cluster/plot_mini_batch_kmeans.py | 1 | 85| 736 | 69339 | 163671 | 
| 183 | 39 sklearn/mixture/gaussian_mixture.py | 381 | 432| 553 | 69892 | 163671 | 
| 184 | 39 sklearn/mixture/gaussian_mixture.py | 140 | 169| 236 | 70128 | 163671 | 
| 185 | 39 sklearn/mixture/gaussian_mixture.py | 289 | 336| 443 | 70571 | 163671 | 
| 186 | 39 sklearn/manifold/mds.py | 234 | 276| 433 | 71004 | 163671 | 
| 187 | 40 sklearn/neighbors/base.py | 833 | 887| 457 | 71461 | 170951 | 
| 188 | 40 sklearn/cluster/k_means_.py | 145 | 158| 143 | 71604 | 170951 | 
| 189 | 40 sklearn/naive_bayes.py | 429 | 444| 169 | 71773 | 170951 | 
| 190 | 41 sklearn/metrics/cluster/setup.py | 1 | 24| 127 | 71900 | 171078 | 
| 191 | 41 sklearn/cluster/k_means_.py | 986 | 1011| 224 | 72124 | 171078 | 
| 192 | 41 sklearn/mixture/gaussian_mixture.py | 98 | 137| 318 | 72442 | 171078 | 
| 193 | 41 sklearn/cluster/birch.py | 152 | 162| 137 | 72579 | 171078 | 
| 194 | 41 examples/mixture/plot_gmm_covariances.py | 47 | 66| 290 | 72869 | 171078 | 
| 195 | 41 examples/cluster/plot_agglomerative_clustering_metrics.py | 1 | 91| 733 | 73602 | 171078 | 
| 196 | 41 sklearn/cluster/k_means_.py | 40 | 106| 586 | 74188 | 171078 | 
| 197 | 41 sklearn/cluster/k_means_.py | 425 | 451| 304 | 74492 | 171078 | 
| 198 | 42 sklearn/gaussian_process/kernels.py | 383 | 417| 189 | 74681 | 186328 | 
| 199 | 42 sklearn/cluster/hierarchical.py | 581 | 601| 134 | 74815 | 186328 | 
| 200 | 43 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 80| 760 | 75575 | 187351 | 
| 201 | 44 sklearn/decomposition/online_lda.py | 137 | 262| 1213 | 76788 | 194009 | 
| 202 | 44 sklearn/multiclass.py | 375 | 410| 300 | 77088 | 194009 | 
| 203 | 44 sklearn/metrics/cluster/supervised.py | 217 | 288| 565 | 77653 | 194009 | 
| 204 | 44 sklearn/cluster/affinity_propagation_.py | 1 | 30| 171 | 77824 | 194009 | 


### Hint

```
In my opinion, yes.

I wanted to compare K-Means, GMM and HDBSCAN and was very disappointed that GMM does not have a `fit_predict` method. The HDBSCAN examples use `fit_predict`, so I was expecting GMM to have the same interface.
I think we should add ``fit_predict`` at least. I wouldn't rename ``n_components``.
I would like to work on this!
@Eight1911 go for it. It is probably relatively simple but maybe not entirely trivial.
@Eight1911 Mind if I take a look at this?
@Eight1911 Do you mind if I jump in as well?
```

## Patch

```diff
diff --git a/sklearn/mixture/base.py b/sklearn/mixture/base.py
--- a/sklearn/mixture/base.py
+++ b/sklearn/mixture/base.py
@@ -172,7 +172,7 @@ def _initialize(self, X, resp):
     def fit(self, X, y=None):
         """Estimate model parameters with the EM algorithm.
 
-        The method fit the model `n_init` times and set the parameters with
+        The method fits the model `n_init` times and set the parameters with
         which the model has the largest likelihood or lower bound. Within each
         trial, the method iterates between E-step and M-step for `max_iter`
         times until the change of likelihood or lower bound is less than
@@ -188,6 +188,32 @@ def fit(self, X, y=None):
         -------
         self
         """
+        self.fit_predict(X, y)
+        return self
+
+    def fit_predict(self, X, y=None):
+        """Estimate model parameters using X and predict the labels for X.
+
+        The method fits the model n_init times and sets the parameters with
+        which the model has the largest likelihood or lower bound. Within each
+        trial, the method iterates between E-step and M-step for `max_iter`
+        times until the change of likelihood or lower bound is less than
+        `tol`, otherwise, a `ConvergenceWarning` is raised. After fitting, it
+        predicts the most probable label for the input data points.
+
+        .. versionadded:: 0.20
+
+        Parameters
+        ----------
+        X : array-like, shape (n_samples, n_features)
+            List of n_features-dimensional data points. Each row
+            corresponds to a single data point.
+
+        Returns
+        -------
+        labels : array, shape (n_samples,)
+            Component labels.
+        """
         X = _check_X(X, self.n_components, ensure_min_samples=2)
         self._check_initial_parameters(X)
 
@@ -240,7 +266,7 @@ def fit(self, X, y=None):
         self._set_parameters(best_params)
         self.n_iter_ = best_n_iter
 
-        return self
+        return log_resp.argmax(axis=1)
 
     def _e_step(self, X):
         """E step.

```

## Test Patch

```diff
diff --git a/sklearn/mixture/tests/test_bayesian_mixture.py b/sklearn/mixture/tests/test_bayesian_mixture.py
--- a/sklearn/mixture/tests/test_bayesian_mixture.py
+++ b/sklearn/mixture/tests/test_bayesian_mixture.py
@@ -1,12 +1,16 @@
 # Author: Wei Xue <xuewei4d@gmail.com>
 #         Thierry Guillemot <thierry.guillemot.work@gmail.com>
 # License: BSD 3 clause
+import copy
 
 import numpy as np
 from scipy.special import gammaln
 
 from sklearn.utils.testing import assert_raise_message
 from sklearn.utils.testing import assert_almost_equal
+from sklearn.utils.testing import assert_array_equal
+
+from sklearn.metrics.cluster import adjusted_rand_score
 
 from sklearn.mixture.bayesian_mixture import _log_dirichlet_norm
 from sklearn.mixture.bayesian_mixture import _log_wishart_norm
@@ -14,7 +18,7 @@
 from sklearn.mixture import BayesianGaussianMixture
 
 from sklearn.mixture.tests.test_gaussian_mixture import RandomData
-from sklearn.exceptions import ConvergenceWarning
+from sklearn.exceptions import ConvergenceWarning, NotFittedError
 from sklearn.utils.testing import assert_greater_equal, ignore_warnings
 
 
@@ -419,3 +423,49 @@ def test_invariant_translation():
             assert_almost_equal(bgmm1.means_, bgmm2.means_ - 100)
             assert_almost_equal(bgmm1.weights_, bgmm2.weights_)
             assert_almost_equal(bgmm1.covariances_, bgmm2.covariances_)
+
+
+def test_bayesian_mixture_fit_predict():
+    rng = np.random.RandomState(0)
+    rand_data = RandomData(rng, scale=7)
+    n_components = 2 * rand_data.n_components
+
+    for covar_type in COVARIANCE_TYPE:
+        bgmm1 = BayesianGaussianMixture(n_components=n_components,
+                                        max_iter=100, random_state=rng,
+                                        tol=1e-3, reg_covar=0)
+        bgmm1.covariance_type = covar_type
+        bgmm2 = copy.deepcopy(bgmm1)
+        X = rand_data.X[covar_type]
+
+        Y_pred1 = bgmm1.fit(X).predict(X)
+        Y_pred2 = bgmm2.fit_predict(X)
+        assert_array_equal(Y_pred1, Y_pred2)
+
+
+def test_bayesian_mixture_predict_predict_proba():
+    # this is the same test as test_gaussian_mixture_predict_predict_proba()
+    rng = np.random.RandomState(0)
+    rand_data = RandomData(rng)
+    for prior_type in PRIOR_TYPE:
+        for covar_type in COVARIANCE_TYPE:
+            X = rand_data.X[covar_type]
+            Y = rand_data.Y
+            bgmm = BayesianGaussianMixture(
+                n_components=rand_data.n_components,
+                random_state=rng,
+                weight_concentration_prior_type=prior_type,
+                covariance_type=covar_type)
+
+            # Check a warning message arrive if we don't do fit
+            assert_raise_message(NotFittedError,
+                                 "This BayesianGaussianMixture instance"
+                                 " is not fitted yet. Call 'fit' with "
+                                 "appropriate arguments before using "
+                                 "this method.", bgmm.predict, X)
+
+            bgmm.fit(X)
+            Y_pred = bgmm.predict(X)
+            Y_pred_proba = bgmm.predict_proba(X).argmax(axis=1)
+            assert_array_equal(Y_pred, Y_pred_proba)
+            assert_greater_equal(adjusted_rand_score(Y, Y_pred), .95)
diff --git a/sklearn/mixture/tests/test_gaussian_mixture.py b/sklearn/mixture/tests/test_gaussian_mixture.py
--- a/sklearn/mixture/tests/test_gaussian_mixture.py
+++ b/sklearn/mixture/tests/test_gaussian_mixture.py
@@ -3,6 +3,7 @@
 # License: BSD 3 clause
 
 import sys
+import copy
 import warnings
 
 import numpy as np
@@ -569,6 +570,26 @@ def test_gaussian_mixture_predict_predict_proba():
         assert_greater(adjusted_rand_score(Y, Y_pred), .95)
 
 
+def test_gaussian_mixture_fit_predict():
+    rng = np.random.RandomState(0)
+    rand_data = RandomData(rng)
+    for covar_type in COVARIANCE_TYPE:
+        X = rand_data.X[covar_type]
+        Y = rand_data.Y
+        g = GaussianMixture(n_components=rand_data.n_components,
+                            random_state=rng, weights_init=rand_data.weights,
+                            means_init=rand_data.means,
+                            precisions_init=rand_data.precisions[covar_type],
+                            covariance_type=covar_type)
+
+        # check if fit_predict(X) is equivalent to fit(X).predict(X)
+        f = copy.deepcopy(g)
+        Y_pred1 = f.fit(X).predict(X)
+        Y_pred2 = g.fit_predict(X)
+        assert_array_equal(Y_pred1, Y_pred2)
+        assert_greater(adjusted_rand_score(Y, Y_pred2), .95)
+
+
 def test_gaussian_mixture_fit():
     # recover the ground truth
     rng = np.random.RandomState(0)

```


## Code snippets

### 1 - sklearn/mixture/gaussian_mixture.py:

Start line: 435, End line: 581

```python
class GaussianMixture(BaseMixture):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Read more in the :ref:`User Guide <gmm>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'},
            defaults to 'full'.
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weights_init : array-like, shape (n_components, ), optional
        The user-provided initial weights, defaults to None.
        If it None, weights are initialized using the `init_params` method.

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None,
        If it None, means are initialized using the `init_params` method.

    precisions_init : array-like, optional.
        The user-provided initial precisions (inverse of the covariance
        matrices), defaults to None.
        If it None, precisions are initialized using the 'init_params' method.
        The shape depends on 'covariance_type'::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    See Also
    --------
    BayesianGaussianMixture : Gaussian mixture model fit with a variational
        inference.
    """
```
### 2 - sklearn/mixture/gmm.py:

Start line: 133, End line: 257

```python
class _GMMBase(BaseEstimator):
    """Gaussian Mixture Model.

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Initializes parameters such that every mixture component has zero
    mean and identity covariance.

    Read more in the :ref:`User Guide <gmm>`.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. The best results is kept.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars. Defaults to 'wmc'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars. Defaults to 'wmc'.

    verbose : int, default: 0
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the change and time needed for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components, n_features)             if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------

    DPGMM : Infinite gaussian mixture model, using the Dirichlet
        process, fit with a variational algorithm


    VBGMM : Finite gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn import mixture
    >>> np.random.seed(1)
    >>> g = mixture.GMM(n_components=2)
    >>> # Generate random observations with two modes centered on 0
    >>> # and 10 to use for training.
    >>> obs = np.concatenate((np.random.randn(100, 1),
    ...                       10 + np.random.randn(300, 1)))
    >>> g.fit(obs)  # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, tol=0.001, verbose=0)
    >>> np.round(g.weights_, 2)
    array([0.75, 0.25])
    >>> np.round(g.means_, 2)
    array([[10.05],
           [ 0.06]])
    >>> np.round(g.covars_, 2) # doctest: +SKIP
    array([[[ 1.02]],
           [[ 0.96]]])
    >>> g.predict([[0], [2], [9], [10]]) # doctest: +ELLIPSIS
    array([1, 1, 0, 0]...)
    >>> np.round(g.score([[0], [2], [9], [10]]), 2)
    array([-2.19, -4.58, -1.75, -1.21])
    >>> # Refit the model on new data (initial parameters remain the
    >>> # same), this time with an even split between the two modes.
    >>> g.fit(20 * [[0]] + 20 * [[10]])  # doctest: +NORMALIZE_WHITESPACE
    GMM(covariance_type='diag', init_params='wmc', min_covar=0.001,
            n_components=2, n_init=1, n_iter=100, params='wmc',
            random_state=None, tol=0.001, verbose=0)
    >>> np.round(g.weights_, 2)
    array([0.5, 0.5])

    """
```
### 3 - sklearn/mixture/bayesian_mixture.py:

Start line: 66, End line: 307

```python
class BayesianGaussianMixture(BaseMixture):
    """Variational Bayesian estimation of a Gaussian mixture.

    This class allows to infer an approximate posterior distribution over the
    parameters of a Gaussian mixture distribution. The effective number of
    components can be inferred from the data.

    This class implements two types of prior for the weights distribution: a
    finite mixture model with Dirichlet distribution and an infinite mixture
    model with the Dirichlet Process. In practice Dirichlet Process inference
    algorithm is approximated and uses a truncated distribution with a fixed
    maximum number of components (called the Stick-breaking representation).
    The number of components actually used almost always depends on the data.

    .. versionadded:: 0.18

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components. Depending on the data and the value
        of the `weight_concentration_prior` the model can decide to not use
        all the components by setting some component `weights_` to values very
        close to zero. The number of effective components is therefore smaller
        than n_components.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        String describing the type of covariance parameters to use.
        Must be one of::

            'full' (each component has its own general covariance matrix),
            'tied' (all components share the same general covariance matrix),
            'diag' (each component has its own diagonal covariance matrix),
            'spherical' (each component has its own single variance).

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain on the likelihood (of the training data with
        respect to the model) is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The result with the highest
        lower bound value on the likelihood is kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        covariances.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of::

            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).

    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``.

    mean_precision_prior : float | None, optional.
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed. Smaller
        values concentrate the means of each clusters around `mean_prior`.
        The value of the parameter must be greater than 0.
        If it is None, it's set to 1.

    mean_prior : array-like, shape (n_features,), optional
        The prior on the mean distribution (Gaussian).
        If it is None, it's set to the mean of X.

    degrees_of_freedom_prior : float | None, optional.
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart). If it is None, it's set to `n_features`.

    covariance_prior : float or array-like, optional
        The prior on the covariance distribution (Wishart).
        If it is None, the emiprical covariance prior is initialized using the
        covariance of X. The shape depends on `covariance_type`::

                (n_features, n_features) if 'full',
                (n_features, n_features) if 'tied',
                (n_features)             if 'diag',
                float                    if 'spherical'

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    warm_start : bool, default to False.
        If 'warm_start' is True, the solution of the last fitting is used as
        initialization for the next call of fit(). This can speed up
        convergence when fit is called several times on similar problems.
        See :term:`the Glossary <warm_start>`.

    verbose : int, default to 0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.

    verbose_interval : int, default to 10.
        Number of iteration done before the next print.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_ : array-like
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    precisions_cholesky_ : array-like
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on ``covariance_type``::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : int
        Number of step used by the best fit of inference to reach the
        convergence.

    lower_bound_ : float
        Lower bound value on the likelihood (of the training data with
        respect to the model) of the best fit of inference.

    weight_concentration_prior_ : tuple or float
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). The type depends on
        ``weight_concentration_prior_type``::

            (float, float) if 'dirichlet_process' (Beta parameters),
            float          if 'dirichlet_distribution' (Dirichlet parameters).

        The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        simplex.

    weight_concentration_ : array-like, shape (n_components,)
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet).

    mean_precision_prior : float
        The precision prior on the mean distribution (Gaussian).
        Controls the extend to where means can be placed.
        Smaller values concentrate the means of each clusters around
        `mean_prior`.

    mean_precision_ : array-like, shape (n_components,)
        The precision of each components on the mean distribution (Gaussian).

    means_prior_ : array-like, shape (n_features,)
        The prior on the mean distribution (Gaussian).

    degrees_of_freedom_prior_ : float
        The prior of the number of degrees of freedom on the covariance
        distributions (Wishart).

    degrees_of_freedom_ : array-like, shape (n_components,)
        The number of degrees of freedom of each components in the model.

    covariance_prior_ : float or array-like
        The prior on the covariance distribution (Wishart).
        The shape depends on `covariance_type`::

            (n_features, n_features) if 'full',
            (n_features, n_features) if 'tied',
            (n_features)             if 'diag',
            float                    if 'spherical'

    See Also
    --------
    GaussianMixture : Finite Gaussian mixture fit with EM.

    References
    ----------

    .. [1] `Bishop, Christopher M. (2006). "Pattern recognition and machine
       learning". Vol. 4 No. 4. New York: Springer.
       <http://www.springer.com/kr/book/9780387310732>`_

    .. [2] `Hagai Attias. (2000). "A Variational Bayesian Framework for
       Graphical Models". In Advances in Neural Information Processing
       Systems 12.
       <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2841&rep=rep1&type=pdf>`_

    .. [3] `Blei, David M. and Michael I. Jordan. (2006). "Variational
       inference for Dirichlet process mixtures". Bayesian analysis 1.1
       <http://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/BleiJordan2005.pdf>`_
    """
```
### 4 - sklearn/mixture/base.py:

Start line: 67, End line: 86

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    """Base class for mixture models.

    This abstract class specifies an interface for all mixture classes and
    provides basic common methods for mixture models.
    """

    def __init__(self, n_components, tol, reg_covar,
                 max_iter, n_init, init_params, random_state, warm_start,
                 verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
```
### 5 - sklearn/mixture/__init__.py:

Start line: 1, End line: 23

```python
"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gmm import sample_gaussian, log_multivariate_normal_density
from .gmm import GMM, distribute_covar_matrix_to_match_covariance_type
from .gmm import _validate_covars
from .dpgmm import DPGMM, VBGMM

from .gaussian_mixture import GaussianMixture
from .bayesian_mixture import BayesianGaussianMixture


__all__ = ['DPGMM',
           'GMM',
           'VBGMM',
           '_validate_covars',
           'distribute_covar_matrix_to_match_covariance_type',
           'log_multivariate_normal_density',
           'sample_gaussian',
           'GaussianMixture',
           'BayesianGaussianMixture']
```
### 6 - sklearn/mixture/dpgmm.py:

Start line: 128, End line: 219

```python
class _DPGMMBase(_GMMBase):
    """Variational Inference for the Infinite Gaussian Mixture Model.

    DPGMM stands for Dirichlet Process Gaussian Mixture Model, and it
    is an infinite mixture model with the Dirichlet Process as a prior
    distribution on the number of clusters. In practice the
    approximate inference algorithm uses a truncated distribution with
    a fixed maximum number of components, but almost always the number
    of components actually used depends on the data.

    Stick-breaking Representation of a Gaussian mixture model
    probability distribution. This class allows for easy and efficient
    inference of an approximate posterior distribution over the
    parameters of a Gaussian mixture model with a variable number of
    components (smaller than the truncation parameter n_components).

    Initialization is with normally-distributed means and identity
    covariance, for proper convergence.

    Read more in the :ref:`User Guide <dpgmm>`.

    Parameters
    ----------
    n_components : int, default 1
        Number of mixture components.

    covariance_type : string, default 'diag'
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    alpha : float, default 1
        Real number representing the concentration parameter of
        the dirichlet process. Intuitively, the Dirichlet Process
        is as likely to start a new cluster for a point as it is
        to add that point to a cluster with alpha elements. A
        higher alpha means more clusters, as the expected number
        of clusters is ``alpha*log(N)``.

    tol : float, default 1e-3
        Convergence threshold.

    n_iter : int, default 10
        Maximum number of iterations to perform before convergence.

    params : string, default 'wmc'
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.

    init_params : string, default 'wmc'
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    verbose : int, default 0
        Controls output verbosity.

    Attributes
    ----------
    covariance_type : string
        String describing the type of covariance parameters used by
        the DP-GMM.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_components : int
        Number of mixture components.

    weights_ : array, shape (`n_components`,)
        Mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    precs_ : array
        Precision (inverse covariance) parameters for each mixture
        component.  The shape depends on `covariance_type`::

            (`n_components`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)                  if 'tied',
            (`n_components`, `n_features`)                if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------
    GMM : Finite Gaussian mixture model fit with EM

    VBGMM : Finite Gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.
    """
```
### 7 - sklearn/mixture/dpgmm.py:

Start line: 1, End line: 44

```python
"""Bayesian Gaussian Mixture Models and
Dirichlet Process Gaussian Mixture Models"""
from __future__ import print_function

import numpy as np
from scipy.special import digamma as _digamma, gammaln as _gammaln
from scipy import linalg
from scipy.linalg import pinvh
from scipy.spatial.distance import cdist

from ..externals.six.moves import xrange
from ..utils import check_random_state, check_array, deprecated
from ..utils.fixes import logsumexp
from ..utils.extmath import squared_norm, stable_cumsum
from ..utils.validation import check_is_fitted
from .. import cluster
from .gmm import _GMMBase


@deprecated("The function digamma is deprecated in 0.18 and "
            "will be removed in 0.20. Use scipy.special.digamma instead.")
def digamma(x):
    return _digamma(x + np.finfo(np.float32).eps)


@deprecated("The function gammaln is deprecated in 0.18 and "
            "will be removed in 0.20. Use scipy.special.gammaln instead.")
def gammaln(x):
    return _gammaln(x + np.finfo(np.float32).eps)
```
### 8 - sklearn/mixture/gmm.py:

Start line: 1, End line: 72

```python
"""
Gaussian Mixture Models.

This implementation corresponds to frequentist (non-Bayesian) formulation
of Gaussian Mixture Models.
"""
from time import time

import numpy as np
from scipy import linalg

from ..base import BaseEstimator
from ..utils import check_random_state, check_array, deprecated
from ..utils.fixes import logsumexp
from ..utils.validation import check_is_fitted
from .. import cluster

from sklearn.externals.six.moves import zip

EPS = np.finfo(float).eps

@deprecated("The function log_multivariate_normal_density is deprecated in 0.18"
            " and will be removed in 0.20.")
def log_multivariate_normal_density(X, means, covars, covariance_type='diag'):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'

    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    log_multivariate_normal_density_dict = {
        'spherical': _log_multivariate_normal_density_spherical,
        'tied': _log_multivariate_normal_density_tied,
        'diag': _log_multivariate_normal_density_diag,
        'full': _log_multivariate_normal_density_full}
    return log_multivariate_normal_density_dict[covariance_type](
        X, means, covars)
```
### 9 - examples/mixture/plot_gmm_selection.py:

Start line: 1, End line: 77

```python
"""
================================
Gaussian Mixture Model Selection
================================

This example shows that model selection can be performed with
Gaussian Mixture Models using information-theoretic criteria (BIC).
Model selection concerns both the covariance type
and the number of components in the model.
In that case, AIC also provides the right result (not shown to save time),
but BIC is better suited if the problem is to identify the right model.
Unlike Bayesian procedures, such inferences are prior-free.

In that case, the model with 2 components and full covariance
(which corresponds to the true generative model) is selected.
"""

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(2, 1, 2)
Y_ = clf.predict(X)
```
### 10 - sklearn/mixture/gmm.py:

Start line: 490, End line: 581

```python
class _GMMBase(BaseEstimator):

    def _fit(self, X, y=None, do_prediction=False):
        # ... other code

        for init in range(self.n_init):
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state).fit(X).cluster_centers_
                if self.verbose > 1:
                    print('\tMeans have been initialized.')

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)
                if self.verbose > 1:
                    print('\tWeights have been initialized.')

            if 'c' in self.init_params or not hasattr(self, 'covars_'):
                cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
                if not cv.shape:
                    cv.shape = (1, 1)
                self.covars_ = \
                    distribute_covar_matrix_to_match_covariance_type(
                        cv, self.covariance_type, self.n_components)
                if self.verbose > 1:
                    print('\tCovariance matrices have been initialized.')

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            for i in range(self.n_iter):
                if self.verbose > 0:
                    print('\tEM iteration ' + str(i + 1))
                    start_iter_time = time()
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self.score_samples(X)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if self.verbose > 1:
                        print('\t\tChange: ' + str(change))
                    if change < self.tol:
                        self.converged_ = True
                        if self.verbose > 0:
                            print('\t\tEM algorithm converged.')
                        break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)
                if self.verbose > 1:
                    print('\t\tEM iteration ' + str(i + 1) + ' took {0:.5f}s'.format(
                        time() - start_iter_time))

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}
                    if self.verbose > 1:
                        print('\tBetter parameters were found.')

            if self.verbose > 1:
                print('\tInitialization ' + str(init + 1) + ' took {0:.5f}s'.format(
                    time() - start_init_time))

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:  # self.n_iter == 0 occurs when using GMM within HMM
            # Need to make sure that there are responsibilities to output
            # Output zeros because it was just a quick initialization
            responsibilities = np.zeros((X.shape[0], self.n_components))

        return responsibilities
```
### 22 - sklearn/mixture/base.py:

Start line: 1, End line: 38

```python
"""Base class for mixture models."""

from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np

from .. import cluster
from ..base import BaseEstimator
from ..base import DensityMixin
from ..externals import six
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state
from ..utils.fixes import logsumexp


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : string
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))
```
### 31 - sklearn/mixture/base.py:

Start line: 264, End line: 288

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    @abstractmethod
    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass
```
### 49 - sklearn/mixture/base.py:

Start line: 343, End line: 361

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)
```
### 66 - sklearn/mixture/base.py:

Start line: 124, End line: 170

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    @abstractmethod
    def _check_parameters(self, X):
        """Check initial parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)
        """
        pass

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self._initialize(X, resp)

    @abstractmethod
    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        pass
```
### 68 - sklearn/mixture/base.py:

Start line: 451, End line: 484

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose == 1:
            print("Initialization %d" % n_init)
        elif self.verbose >= 2:
            print("Initialization %d" % n_init)
            self._init_prev_time = time()
            self._iter_prev_time = self._init_prev_time
```
### 73 - sklearn/mixture/base.py:

Start line: 245, End line: 262

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
```
### 82 - sklearn/mixture/base.py:

Start line: 88, End line: 122

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _check_initial_parameters(self, X):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        """
        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        # Check all the parameters values of the derived class
        self._check_parameters(X)
```
### 90 - sklearn/mixture/base.py:

Start line: 172, End line: 243

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fit the model `n_init` times and set the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a `ConvergenceWarning` is raised.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        X = _check_X(X, self.n_components, ensure_min_samples=2)
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = self.lower_bound_

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                self.lower_bound_ = self._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = self.lower_bound_ - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(self.lower_bound_)

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self
```
### 124 - sklearn/mixture/base.py:

Start line: 412, End line: 449

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        pass
```
### 137 - sklearn/mixture/base.py:

Start line: 290, End line: 307

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])

        return logsumexp(self._estimate_weighted_log_prob(X), axis=1)
```
### 152 - sklearn/mixture/base.py:

Start line: 363, End line: 410

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample

        y : array, shape (nsamples,)
            Component labels

        """
        self._check_is_fitted()

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components))

        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == 'full':
            X = np.vstack([
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp)])
        elif self.covariance_type == "tied":
            X = np.vstack([
                rng.multivariate_normal(mean, self.covariances_, int(sample))
                for (mean, sample) in zip(
                    self.means_, n_samples_comp)])
        else:
            X = np.vstack([
                mean + rng.randn(sample, n_features) * np.sqrt(covariance)
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp)])

        y = np.concatenate([j * np.ones(sample, dtype=int)
                           for j, sample in enumerate(n_samples_comp)])

        return (X, y)
```
### 155 - sklearn/mixture/base.py:

Start line: 486, End line: 504

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def _print_verbose_msg_iter_end(self, n_iter, diff_ll):
        """Print verbose message on initialization."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose == 1:
                print("  Iteration %d" % n_iter)
            elif self.verbose >= 2:
                cur_time = time()
                print("  Iteration %d\t time lapse %.5fs\t ll change %.5f" % (
                    n_iter, cur_time - self._iter_prev_time, diff_ll))
                self._iter_prev_time = cur_time

    def _print_verbose_msg_init_end(self, ll):
        """Print verbose message on the end of iteration."""
        if self.verbose == 1:
            print("Initialization converged: %s" % self.converged_)
        elif self.verbose >= 2:
            print("Initialization converged: %s\t time lapse %.5fs\t ll %.5f" %
                  (self.converged_, time() - self._init_prev_time, ll))
```
### 166 - sklearn/mixture/base.py:

Start line: 309, End line: 341

```python
class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_likelihood : float
            Log likelihood of the Gaussian mixture given X.
        """
        return self.score_samples(X).mean()

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.means_.shape[1])
        return self._estimate_weighted_log_prob(X).argmax(axis=1)
```

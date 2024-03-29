# scikit-learn__scikit-learn-10306

| **scikit-learn/scikit-learn** | `b90661d6a46aa3619d3eec94d5281f5888add501` |
| ---- | ---- |
| **No of patches** | 9 |
| **All found context length** | - |
| **Any found context length** | 26544 |
| **Avg pos** | 8.444444444444445 |
| **Min pos** | 76 |
| **Max pos** | 76 |
| **Top file pos** | 4 |
| **Missing snippets** | 17 |
| **Missing patch files** | 6 |


## Expected patch

```diff
diff --git a/sklearn/cluster/affinity_propagation_.py b/sklearn/cluster/affinity_propagation_.py
--- a/sklearn/cluster/affinity_propagation_.py
+++ b/sklearn/cluster/affinity_propagation_.py
@@ -390,5 +390,5 @@ def predict(self, X):
         else:
             warnings.warn("This model does not have any cluster centers "
                           "because affinity propagation did not converge. "
-                          "Labeling every sample as '-1'.")
+                          "Labeling every sample as '-1'.", ConvergenceWarning)
             return np.array([-1] * X.shape[0])
diff --git a/sklearn/cluster/birch.py b/sklearn/cluster/birch.py
--- a/sklearn/cluster/birch.py
+++ b/sklearn/cluster/birch.py
@@ -15,7 +15,7 @@
 from ..utils import check_array
 from ..utils.extmath import row_norms, safe_sparse_dot
 from ..utils.validation import check_is_fitted
-from ..exceptions import NotFittedError
+from ..exceptions import NotFittedError, ConvergenceWarning
 from .hierarchical import AgglomerativeClustering
 
 
@@ -626,7 +626,7 @@ def _global_clustering(self, X=None):
                 warnings.warn(
                     "Number of subclusters found (%d) by Birch is less "
                     "than (%d). Decrease the threshold."
-                    % (len(centroids), self.n_clusters))
+                    % (len(centroids), self.n_clusters), ConvergenceWarning)
         else:
             # The global clustering step that clusters the subclusters of
             # the leaves. It assumes the centroids of the subclusters as
diff --git a/sklearn/cross_decomposition/pls_.py b/sklearn/cross_decomposition/pls_.py
--- a/sklearn/cross_decomposition/pls_.py
+++ b/sklearn/cross_decomposition/pls_.py
@@ -16,6 +16,7 @@
 from ..utils import check_array, check_consistent_length
 from ..utils.extmath import svd_flip
 from ..utils.validation import check_is_fitted, FLOAT_DTYPES
+from ..exceptions import ConvergenceWarning
 from ..externals import six
 
 __all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']
@@ -74,7 +75,8 @@ def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
         if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
             break
         if ite == max_iter:
-            warnings.warn('Maximum number of iterations reached')
+            warnings.warn('Maximum number of iterations reached',
+                          ConvergenceWarning)
             break
         x_weights_old = x_weights
         ite += 1
diff --git a/sklearn/decomposition/fastica_.py b/sklearn/decomposition/fastica_.py
--- a/sklearn/decomposition/fastica_.py
+++ b/sklearn/decomposition/fastica_.py
@@ -15,6 +15,7 @@
 from scipy import linalg
 
 from ..base import BaseEstimator, TransformerMixin
+from ..exceptions import ConvergenceWarning
 from ..externals import six
 from ..externals.six import moves
 from ..externals.six import string_types
@@ -116,7 +117,8 @@ def _ica_par(X, tol, g, fun_args, max_iter, w_init):
             break
     else:
         warnings.warn('FastICA did not converge. Consider increasing '
-                      'tolerance or the maximum number of iterations.')
+                      'tolerance or the maximum number of iterations.',
+                      ConvergenceWarning)
 
     return W, ii + 1
 
diff --git a/sklearn/gaussian_process/gpc.py b/sklearn/gaussian_process/gpc.py
--- a/sklearn/gaussian_process/gpc.py
+++ b/sklearn/gaussian_process/gpc.py
@@ -19,6 +19,7 @@
 from sklearn.utils import check_random_state
 from sklearn.preprocessing import LabelEncoder
 from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
+from sklearn.exceptions import ConvergenceWarning
 
 
 # Values required for approximating the logistic sigmoid by
@@ -428,7 +429,8 @@ def _constrained_optimization(self, obj_func, initial_theta, bounds):
                 fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
             if convergence_dict["warnflag"] != 0:
                 warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
-                              " state: %s" % convergence_dict)
+                              " state: %s" % convergence_dict,
+                              ConvergenceWarning)
         elif callable(self.optimizer):
             theta_opt, func_min = \
                 self.optimizer(obj_func, initial_theta, bounds=bounds)
diff --git a/sklearn/gaussian_process/gpr.py b/sklearn/gaussian_process/gpr.py
--- a/sklearn/gaussian_process/gpr.py
+++ b/sklearn/gaussian_process/gpr.py
@@ -16,6 +16,7 @@
 from sklearn.utils import check_random_state
 from sklearn.utils.validation import check_X_y, check_array
 from sklearn.utils.deprecation import deprecated
+from sklearn.exceptions import ConvergenceWarning
 
 
 class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
@@ -461,7 +462,8 @@ def _constrained_optimization(self, obj_func, initial_theta, bounds):
                 fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
             if convergence_dict["warnflag"] != 0:
                 warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
-                              " state: %s" % convergence_dict)
+                              " state: %s" % convergence_dict,
+                              ConvergenceWarning)
         elif callable(self.optimizer):
             theta_opt, func_min = \
                 self.optimizer(obj_func, initial_theta, bounds=bounds)
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -29,7 +29,7 @@
 from ..utils.fixes import logsumexp
 from ..utils.optimize import newton_cg
 from ..utils.validation import check_X_y
-from ..exceptions import NotFittedError
+from ..exceptions import NotFittedError, ConvergenceWarning
 from ..utils.multiclass import check_classification_targets
 from ..externals.joblib import Parallel, delayed
 from ..model_selection import check_cv
@@ -716,7 +716,7 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                     iprint=(verbose > 0) - 1, pgtol=tol)
             if info["warnflag"] == 1 and verbose > 0:
                 warnings.warn("lbfgs failed to converge. Increase the number "
-                              "of iterations.")
+                              "of iterations.", ConvergenceWarning)
             try:
                 n_iter_i = info['nit'] - 1
             except:
diff --git a/sklearn/linear_model/ransac.py b/sklearn/linear_model/ransac.py
--- a/sklearn/linear_model/ransac.py
+++ b/sklearn/linear_model/ransac.py
@@ -13,6 +13,7 @@
 from ..utils.validation import check_is_fitted
 from .base import LinearRegression
 from ..utils.validation import has_fit_parameter
+from ..exceptions import ConvergenceWarning
 
 _EPSILON = np.spacing(1)
 
@@ -453,7 +454,7 @@ def fit(self, X, y, sample_weight=None):
                               " early due to skipping more iterations than"
                               " `max_skips`. See estimator attributes for"
                               " diagnostics (n_skips*).",
-                              UserWarning)
+                              ConvergenceWarning)
 
         # estimate final model using all inliers
         base_estimator.fit(X_inlier_best, y_inlier_best)
diff --git a/sklearn/linear_model/ridge.py b/sklearn/linear_model/ridge.py
--- a/sklearn/linear_model/ridge.py
+++ b/sklearn/linear_model/ridge.py
@@ -31,6 +31,7 @@
 from ..model_selection import GridSearchCV
 from ..externals import six
 from ..metrics.scorer import check_scoring
+from ..exceptions import ConvergenceWarning
 
 
 def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=1e-3, verbose=0):
@@ -73,7 +74,7 @@ def _mv(x):
 
         if max_iter is None and info > 0 and verbose:
             warnings.warn("sparse_cg did not converge after %d iterations." %
-                          info)
+                          info, ConvergenceWarning)
 
     return coefs
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/cluster/affinity_propagation_.py | 393 | 393 | - | 14 | -
| sklearn/cluster/birch.py | 18 | 18 | - | 4 | -
| sklearn/cluster/birch.py | 629 | 629 | - | 4 | -
| sklearn/cross_decomposition/pls_.py | 19 | 19 | - | - | -
| sklearn/cross_decomposition/pls_.py | 77 | 77 | - | - | -
| sklearn/decomposition/fastica_.py | 18 | 18 | - | 26 | -
| sklearn/decomposition/fastica_.py | 119 | 119 | 76 | 26 | 26544
| sklearn/gaussian_process/gpc.py | 22 | 22 | - | - | -
| sklearn/gaussian_process/gpc.py | 431 | 431 | - | - | -
| sklearn/gaussian_process/gpr.py | 19 | 19 | - | - | -
| sklearn/gaussian_process/gpr.py | 464 | 464 | - | - | -
| sklearn/linear_model/logistic.py | 32 | 32 | - | - | -
| sklearn/linear_model/logistic.py | 719 | 719 | - | - | -
| sklearn/linear_model/ransac.py | 16 | 16 | - | - | -
| sklearn/linear_model/ransac.py | 456 | 456 | - | - | -
| sklearn/linear_model/ridge.py | 34 | 34 | - | - | -
| sklearn/linear_model/ridge.py | 76 | 76 | - | - | -


## Problem Statement

```
Some UserWarnings should be ConvergenceWarnings
Some warnings raised during testing show that we do not use `ConvergenceWarning` when it is appropriate in some cases. For example (from [here](https://github.com/scikit-learn/scikit-learn/issues/10158#issuecomment-345453334)):

\`\`\`python
/home/lesteve/dev/alt-scikit-learn/sklearn/decomposition/fastica_.py:118: UserWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
/home/lesteve/dev/alt-scikit-learn/sklearn/cluster/birch.py:629: UserWarning: Number of subclusters found (2) by Birch is less than (3). Decrease the threshold.
\`\`\`

These should be changed, at least. For bonus points, the contributor could look for other warning messages that mention "converge".

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/exceptions.py | 40 | 71| 216 | 216 | 1186 | 
| 2 | 1 sklearn/exceptions.py | 74 | 96| 186 | 402 | 1186 | 
| 3 | 1 sklearn/exceptions.py | 99 | 128| 341 | 743 | 1186 | 
| 4 | 2 sklearn/utils/testing.py | 117 | 159| 299 | 1042 | 7974 | 
| 5 | 2 sklearn/exceptions.py | 131 | 157| 175 | 1217 | 7974 | 
| 6 | 3 sklearn/utils/estimator_checks.py | 1576 | 1618| 449 | 1666 | 25893 | 
| 7 | 3 sklearn/utils/testing.py | 162 | 228| 522 | 2188 | 25893 | 
| 8 | 3 sklearn/utils/testing.py | 231 | 248| 164 | 2352 | 25893 | 
| 9 | 3 sklearn/utils/estimator_checks.py | 1145 | 1199| 567 | 2919 | 25893 | 
| 10 | 3 sklearn/utils/estimator_checks.py | 1253 | 1360| 1123 | 4042 | 25893 | 
| 11 | **4 sklearn/cluster/birch.py** | 297 | 321| 270 | 4312 | 31176 | 
| 12 | 4 sklearn/utils/estimator_checks.py | 1414 | 1443| 324 | 4636 | 31176 | 
| 13 | 5 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 312 | 4948 | 43317 | 
| 14 | 5 sklearn/utils/estimator_checks.py | 1008 | 1076| 601 | 5549 | 43317 | 
| 15 | 5 sklearn/utils/estimator_checks.py | 1446 | 1480| 336 | 5885 | 43317 | 
| 16 | 6 sklearn/__init__.py | 97 | 148| 510 | 6395 | 44538 | 
| 17 | 7 sklearn/utils/validation.py | 1 | 31| 141 | 6536 | 51801 | 
| 18 | 7 sklearn/utils/estimator_checks.py | 661 | 696| 341 | 6877 | 51801 | 
| 19 | 7 sklearn/utils/estimator_checks.py | 1621 | 1638| 174 | 7051 | 51801 | 
| 20 | 7 sklearn/utils/estimator_checks.py | 1511 | 1549| 442 | 7493 | 51801 | 
| 21 | 7 sklearn/utils/estimator_checks.py | 489 | 524| 328 | 7821 | 51801 | 
| 22 | 8 sklearn/linear_model/coordinate_descent.py | 8 | 29| 154 | 7975 | 72203 | 
| 23 | 8 sklearn/utils/estimator_checks.py | 792 | 814| 248 | 8223 | 72203 | 
| 24 | 8 sklearn/utils/estimator_checks.py | 1483 | 1508| 277 | 8500 | 72203 | 
| 25 | 8 sklearn/utils/estimator_checks.py | 1 | 72| 625 | 9125 | 72203 | 
| 26 | 8 sklearn/utils/testing.py | 281 | 338| 449 | 9574 | 72203 | 
| 27 | 8 sklearn/utils/estimator_checks.py | 141 | 161| 203 | 9777 | 72203 | 
| 28 | 8 sklearn/utils/estimator_checks.py | 1641 | 1672| 340 | 10117 | 72203 | 
| 29 | 9 sklearn/cluster/k_means_.py | 1162 | 1225| 659 | 10776 | 85928 | 
| 30 | 9 sklearn/utils/estimator_checks.py | 203 | 237| 305 | 11081 | 85928 | 
| 31 | 9 sklearn/utils/estimator_checks.py | 779 | 789| 142 | 11223 | 85928 | 
| 32 | 9 sklearn/utils/testing.py | 711 | 745| 240 | 11463 | 85928 | 
| 33 | 10 sklearn/decomposition/nmf.py | 481 | 506| 247 | 11710 | 97611 | 
| 34 | 11 sklearn/learning_curve.py | 1 | 25| 145 | 11855 | 101019 | 
| 35 | **11 sklearn/cluster/birch.py** | 540 | 553| 140 | 11995 | 101019 | 
| 36 | 11 sklearn/utils/estimator_checks.py | 894 | 928| 372 | 12367 | 101019 | 
| 37 | 12 benchmarks/bench_rcv1_logreg_convergence.py | 67 | 92| 201 | 12568 | 102967 | 
| 38 | 12 sklearn/utils/estimator_checks.py | 1675 | 1710| 352 | 12920 | 102967 | 
| 39 | 12 sklearn/utils/estimator_checks.py | 473 | 486| 166 | 13086 | 102967 | 
| 40 | 13 sklearn/cluster/hierarchical.py | 10 | 27| 126 | 13212 | 111057 | 
| 41 | 13 sklearn/utils/estimator_checks.py | 1782 | 1802| 200 | 13412 | 111057 | 
| 42 | **14 sklearn/cluster/affinity_propagation_.py** | 121 | 205| 796 | 14208 | 114236 | 
| 43 | 14 sklearn/utils/estimator_checks.py | 1765 | 1779| 211 | 14419 | 114236 | 
| 44 | 15 sklearn/utils/deprecation.py | 107 | 135| 244 | 14663 | 115166 | 
| 45 | 15 sklearn/utils/estimator_checks.py | 571 | 619| 441 | 15104 | 115166 | 
| 46 | 15 sklearn/utils/estimator_checks.py | 112 | 138| 258 | 15362 | 115166 | 
| 47 | **15 sklearn/cluster/affinity_propagation_.py** | 207 | 231| 278 | 15640 | 115166 | 
| 48 | 15 sklearn/utils/estimator_checks.py | 727 | 758| 297 | 15937 | 115166 | 
| 49 | 15 sklearn/utils/estimator_checks.py | 1119 | 1142| 185 | 16122 | 115166 | 
| 50 | 15 benchmarks/bench_rcv1_logreg_convergence.py | 95 | 121| 218 | 16340 | 115166 | 
| 51 | 15 sklearn/utils/estimator_checks.py | 1219 | 1250| 258 | 16598 | 115166 | 
| 52 | 15 sklearn/utils/estimator_checks.py | 959 | 982| 269 | 16867 | 115166 | 
| 53 | 16 examples/cluster/plot_linkage_comparison.py | 1 | 80| 576 | 17443 | 116301 | 
| 54 | 16 sklearn/utils/estimator_checks.py | 761 | 776| 139 | 17582 | 116301 | 
| 55 | 16 sklearn/utils/estimator_checks.py | 1552 | 1573| 214 | 17796 | 116301 | 
| 56 | **16 sklearn/cluster/birch.py** | 451 | 500| 447 | 18243 | 116301 | 
| 57 | 17 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 18754 | 116834 | 
| 58 | 18 benchmarks/bench_plot_nmf.py | 106 | 149| 401 | 19155 | 120747 | 
| 59 | 18 sklearn/utils/estimator_checks.py | 450 | 470| 253 | 19408 | 120747 | 
| 60 | 19 sklearn/svm/classes.py | 1 | 245| 104 | 19512 | 131298 | 
| 61 | **19 sklearn/cluster/birch.py** | 324 | 426| 990 | 20502 | 131298 | 
| 62 | 20 sklearn/cross_validation.py | 1 | 59| 389 | 20891 | 148790 | 
| 63 | 20 examples/cluster/plot_linkage_comparison.py | 82 | 150| 559 | 21450 | 148790 | 
| 64 | 20 sklearn/utils/estimator_checks.py | 1742 | 1762| 233 | 21683 | 148790 | 
| 65 | 20 sklearn/cluster/k_means_.py | 1 | 37| 207 | 21890 | 148790 | 
| 66 | 21 sklearn/metrics/scorer.py | 469 | 533| 744 | 22634 | 153664 | 
| 67 | 22 examples/cluster/plot_cluster_comparison.py | 88 | 184| 833 | 23467 | 155195 | 
| 68 | 22 sklearn/utils/estimator_checks.py | 622 | 644| 221 | 23688 | 155195 | 
| 69 | 23 sklearn/metrics/pairwise.py | 1 | 30| 131 | 23819 | 166727 | 
| 70 | 23 sklearn/utils/validation.py | 477 | 550| 839 | 24658 | 166727 | 
| 71 | 23 sklearn/utils/estimator_checks.py | 291 | 344| 520 | 25178 | 166727 | 
| 72 | 24 sklearn/utils/optimize.py | 1 | 24| 139 | 25317 | 168218 | 
| 73 | 25 examples/covariance/plot_outlier_detection.py | 77 | 130| 632 | 25949 | 169437 | 
| 74 | 25 sklearn/utils/estimator_checks.py | 1202 | 1216| 146 | 26095 | 169437 | 
| 75 | 25 sklearn/utils/estimator_checks.py | 699 | 724| 234 | 26329 | 169437 | 
| **-> 76 <-** | **26 sklearn/decomposition/fastica_.py** | 98 | 121| 215 | 26544 | 174137 | 
| 77 | 26 sklearn/utils/estimator_checks.py | 1920 | 1943| 308 | 26852 | 174137 | 


## Missing Patch Files

 * 1: sklearn/cluster/affinity_propagation_.py
 * 2: sklearn/cluster/birch.py
 * 3: sklearn/cross_decomposition/pls_.py
 * 4: sklearn/decomposition/fastica_.py
 * 5: sklearn/gaussian_process/gpc.py
 * 6: sklearn/gaussian_process/gpr.py
 * 7: sklearn/linear_model/logistic.py
 * 8: sklearn/linear_model/ransac.py
 * 9: sklearn/linear_model/ridge.py

### Hint

```
Could I give this a go?
@patrick1011 please go ahead!
```

## Patch

```diff
diff --git a/sklearn/cluster/affinity_propagation_.py b/sklearn/cluster/affinity_propagation_.py
--- a/sklearn/cluster/affinity_propagation_.py
+++ b/sklearn/cluster/affinity_propagation_.py
@@ -390,5 +390,5 @@ def predict(self, X):
         else:
             warnings.warn("This model does not have any cluster centers "
                           "because affinity propagation did not converge. "
-                          "Labeling every sample as '-1'.")
+                          "Labeling every sample as '-1'.", ConvergenceWarning)
             return np.array([-1] * X.shape[0])
diff --git a/sklearn/cluster/birch.py b/sklearn/cluster/birch.py
--- a/sklearn/cluster/birch.py
+++ b/sklearn/cluster/birch.py
@@ -15,7 +15,7 @@
 from ..utils import check_array
 from ..utils.extmath import row_norms, safe_sparse_dot
 from ..utils.validation import check_is_fitted
-from ..exceptions import NotFittedError
+from ..exceptions import NotFittedError, ConvergenceWarning
 from .hierarchical import AgglomerativeClustering
 
 
@@ -626,7 +626,7 @@ def _global_clustering(self, X=None):
                 warnings.warn(
                     "Number of subclusters found (%d) by Birch is less "
                     "than (%d). Decrease the threshold."
-                    % (len(centroids), self.n_clusters))
+                    % (len(centroids), self.n_clusters), ConvergenceWarning)
         else:
             # The global clustering step that clusters the subclusters of
             # the leaves. It assumes the centroids of the subclusters as
diff --git a/sklearn/cross_decomposition/pls_.py b/sklearn/cross_decomposition/pls_.py
--- a/sklearn/cross_decomposition/pls_.py
+++ b/sklearn/cross_decomposition/pls_.py
@@ -16,6 +16,7 @@
 from ..utils import check_array, check_consistent_length
 from ..utils.extmath import svd_flip
 from ..utils.validation import check_is_fitted, FLOAT_DTYPES
+from ..exceptions import ConvergenceWarning
 from ..externals import six
 
 __all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']
@@ -74,7 +75,8 @@ def _nipals_twoblocks_inner_loop(X, Y, mode="A", max_iter=500, tol=1e-06,
         if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
             break
         if ite == max_iter:
-            warnings.warn('Maximum number of iterations reached')
+            warnings.warn('Maximum number of iterations reached',
+                          ConvergenceWarning)
             break
         x_weights_old = x_weights
         ite += 1
diff --git a/sklearn/decomposition/fastica_.py b/sklearn/decomposition/fastica_.py
--- a/sklearn/decomposition/fastica_.py
+++ b/sklearn/decomposition/fastica_.py
@@ -15,6 +15,7 @@
 from scipy import linalg
 
 from ..base import BaseEstimator, TransformerMixin
+from ..exceptions import ConvergenceWarning
 from ..externals import six
 from ..externals.six import moves
 from ..externals.six import string_types
@@ -116,7 +117,8 @@ def _ica_par(X, tol, g, fun_args, max_iter, w_init):
             break
     else:
         warnings.warn('FastICA did not converge. Consider increasing '
-                      'tolerance or the maximum number of iterations.')
+                      'tolerance or the maximum number of iterations.',
+                      ConvergenceWarning)
 
     return W, ii + 1
 
diff --git a/sklearn/gaussian_process/gpc.py b/sklearn/gaussian_process/gpc.py
--- a/sklearn/gaussian_process/gpc.py
+++ b/sklearn/gaussian_process/gpc.py
@@ -19,6 +19,7 @@
 from sklearn.utils import check_random_state
 from sklearn.preprocessing import LabelEncoder
 from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
+from sklearn.exceptions import ConvergenceWarning
 
 
 # Values required for approximating the logistic sigmoid by
@@ -428,7 +429,8 @@ def _constrained_optimization(self, obj_func, initial_theta, bounds):
                 fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
             if convergence_dict["warnflag"] != 0:
                 warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
-                              " state: %s" % convergence_dict)
+                              " state: %s" % convergence_dict,
+                              ConvergenceWarning)
         elif callable(self.optimizer):
             theta_opt, func_min = \
                 self.optimizer(obj_func, initial_theta, bounds=bounds)
diff --git a/sklearn/gaussian_process/gpr.py b/sklearn/gaussian_process/gpr.py
--- a/sklearn/gaussian_process/gpr.py
+++ b/sklearn/gaussian_process/gpr.py
@@ -16,6 +16,7 @@
 from sklearn.utils import check_random_state
 from sklearn.utils.validation import check_X_y, check_array
 from sklearn.utils.deprecation import deprecated
+from sklearn.exceptions import ConvergenceWarning
 
 
 class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
@@ -461,7 +462,8 @@ def _constrained_optimization(self, obj_func, initial_theta, bounds):
                 fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
             if convergence_dict["warnflag"] != 0:
                 warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
-                              " state: %s" % convergence_dict)
+                              " state: %s" % convergence_dict,
+                              ConvergenceWarning)
         elif callable(self.optimizer):
             theta_opt, func_min = \
                 self.optimizer(obj_func, initial_theta, bounds=bounds)
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -29,7 +29,7 @@
 from ..utils.fixes import logsumexp
 from ..utils.optimize import newton_cg
 from ..utils.validation import check_X_y
-from ..exceptions import NotFittedError
+from ..exceptions import NotFittedError, ConvergenceWarning
 from ..utils.multiclass import check_classification_targets
 from ..externals.joblib import Parallel, delayed
 from ..model_selection import check_cv
@@ -716,7 +716,7 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                     iprint=(verbose > 0) - 1, pgtol=tol)
             if info["warnflag"] == 1 and verbose > 0:
                 warnings.warn("lbfgs failed to converge. Increase the number "
-                              "of iterations.")
+                              "of iterations.", ConvergenceWarning)
             try:
                 n_iter_i = info['nit'] - 1
             except:
diff --git a/sklearn/linear_model/ransac.py b/sklearn/linear_model/ransac.py
--- a/sklearn/linear_model/ransac.py
+++ b/sklearn/linear_model/ransac.py
@@ -13,6 +13,7 @@
 from ..utils.validation import check_is_fitted
 from .base import LinearRegression
 from ..utils.validation import has_fit_parameter
+from ..exceptions import ConvergenceWarning
 
 _EPSILON = np.spacing(1)
 
@@ -453,7 +454,7 @@ def fit(self, X, y, sample_weight=None):
                               " early due to skipping more iterations than"
                               " `max_skips`. See estimator attributes for"
                               " diagnostics (n_skips*).",
-                              UserWarning)
+                              ConvergenceWarning)
 
         # estimate final model using all inliers
         base_estimator.fit(X_inlier_best, y_inlier_best)
diff --git a/sklearn/linear_model/ridge.py b/sklearn/linear_model/ridge.py
--- a/sklearn/linear_model/ridge.py
+++ b/sklearn/linear_model/ridge.py
@@ -31,6 +31,7 @@
 from ..model_selection import GridSearchCV
 from ..externals import six
 from ..metrics.scorer import check_scoring
+from ..exceptions import ConvergenceWarning
 
 
 def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=1e-3, verbose=0):
@@ -73,7 +74,7 @@ def _mv(x):
 
         if max_iter is None and info > 0 and verbose:
             warnings.warn("sparse_cg did not converge after %d iterations." %
-                          info)
+                          info, ConvergenceWarning)
 
     return coefs
 

```

## Test Patch

```diff
diff --git a/sklearn/cluster/tests/test_affinity_propagation.py b/sklearn/cluster/tests/test_affinity_propagation.py
--- a/sklearn/cluster/tests/test_affinity_propagation.py
+++ b/sklearn/cluster/tests/test_affinity_propagation.py
@@ -133,12 +133,14 @@ def test_affinity_propagation_predict_non_convergence():
     X = np.array([[0, 0], [1, 1], [-2, -2]])
 
     # Force non-convergence by allowing only a single iteration
-    af = AffinityPropagation(preference=-10, max_iter=1).fit(X)
+    af = assert_warns(ConvergenceWarning,
+                      AffinityPropagation(preference=-10, max_iter=1).fit, X)
 
     # At prediction time, consider new samples as noise since there are no
     # clusters
-    assert_array_equal(np.array([-1, -1, -1]),
-                       af.predict(np.array([[2, 2], [3, 3], [4, 4]])))
+    to_predict = np.array([[2, 2], [3, 3], [4, 4]])
+    y = assert_warns(ConvergenceWarning, af.predict, to_predict)
+    assert_array_equal(np.array([-1, -1, -1]), y)
 
 
 def test_equal_similarities_and_preferences():
diff --git a/sklearn/cluster/tests/test_birch.py b/sklearn/cluster/tests/test_birch.py
--- a/sklearn/cluster/tests/test_birch.py
+++ b/sklearn/cluster/tests/test_birch.py
@@ -9,6 +9,7 @@
 from sklearn.cluster.birch import Birch
 from sklearn.cluster.hierarchical import AgglomerativeClustering
 from sklearn.datasets import make_blobs
+from sklearn.exceptions import ConvergenceWarning
 from sklearn.linear_model import ElasticNet
 from sklearn.metrics import pairwise_distances_argmin, v_measure_score
 
@@ -93,7 +94,7 @@ def test_n_clusters():
 
     # Test that a small number of clusters raises a warning.
     brc4 = Birch(threshold=10000.)
-    assert_warns(UserWarning, brc4.fit, X)
+    assert_warns(ConvergenceWarning, brc4.fit, X)
 
 
 def test_sparse_X():
diff --git a/sklearn/cross_decomposition/tests/test_pls.py b/sklearn/cross_decomposition/tests/test_pls.py
--- a/sklearn/cross_decomposition/tests/test_pls.py
+++ b/sklearn/cross_decomposition/tests/test_pls.py
@@ -3,11 +3,12 @@
 
 from sklearn.utils.testing import (assert_equal, assert_array_almost_equal,
                                    assert_array_equal, assert_true,
-                                   assert_raise_message)
+                                   assert_raise_message, assert_warns)
 from sklearn.datasets import load_linnerud
 from sklearn.cross_decomposition import pls_, CCA
 from sklearn.preprocessing import StandardScaler
 from sklearn.utils import check_random_state
+from sklearn.exceptions import ConvergenceWarning
 
 
 def test_pls():
@@ -260,6 +261,15 @@ def check_ortho(M, err_msg):
     check_ortho(pls_ca.y_scores_, "y scores are not orthogonal")
 
 
+def test_convergence_fail():
+    d = load_linnerud()
+    X = d.data
+    Y = d.target
+    pls_bynipals = pls_.PLSCanonical(n_components=X.shape[1],
+                                     max_iter=2, tol=1e-10)
+    assert_warns(ConvergenceWarning, pls_bynipals.fit, X, Y)
+
+
 def test_PLSSVD():
     # Let's check the PLSSVD doesn't return all possible component but just
     # the specified number
diff --git a/sklearn/decomposition/tests/test_fastica.py b/sklearn/decomposition/tests/test_fastica.py
--- a/sklearn/decomposition/tests/test_fastica.py
+++ b/sklearn/decomposition/tests/test_fastica.py
@@ -18,6 +18,7 @@
 from sklearn.decomposition import FastICA, fastica, PCA
 from sklearn.decomposition.fastica_ import _gs_decorrelation
 from sklearn.externals.six import moves
+from sklearn.exceptions import ConvergenceWarning
 
 
 def center_and_norm(x, axis=-1):
@@ -141,6 +142,31 @@ def test_fastica_nowhiten():
     assert_true(hasattr(ica, 'mixing_'))
 
 
+def test_fastica_convergence_fail():
+    # Test the FastICA algorithm on very simple data
+    # (see test_non_square_fastica).
+    # Ensure a ConvergenceWarning raised if the tolerance is sufficiently low.
+    rng = np.random.RandomState(0)
+
+    n_samples = 1000
+    # Generate two sources:
+    t = np.linspace(0, 100, n_samples)
+    s1 = np.sin(t)
+    s2 = np.ceil(np.sin(np.pi * t))
+    s = np.c_[s1, s2].T
+    center_and_norm(s)
+    s1, s2 = s
+
+    # Mixing matrix
+    mixing = rng.randn(6, 2)
+    m = np.dot(mixing, s)
+
+    # Do fastICA with tolerance 0. to ensure failing convergence
+    ica = FastICA(algorithm="parallel", n_components=2, random_state=rng,
+                  max_iter=2, tol=0.)
+    assert_warns(ConvergenceWarning, ica.fit, m.T)
+
+
 def test_non_square_fastica(add_noise=False):
     # Test the FastICA algorithm on very simple data.
     rng = np.random.RandomState(0)
diff --git a/sklearn/linear_model/tests/test_logistic.py b/sklearn/linear_model/tests/test_logistic.py
--- a/sklearn/linear_model/tests/test_logistic.py
+++ b/sklearn/linear_model/tests/test_logistic.py
@@ -312,6 +312,15 @@ def test_consistency_path():
                                   err_msg="with solver = %s" % solver)
 
 
+def test_logistic_regression_path_convergence_fail():
+    rng = np.random.RandomState(0)
+    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
+    y = [1] * 100 + [-1] * 100
+    Cs = [1e3]
+    assert_warns(ConvergenceWarning, logistic_regression_path,
+                 X, y, Cs=Cs, tol=0., max_iter=1, random_state=0, verbose=1)
+
+
 def test_liblinear_dual_random_state():
     # random_state is relevant for liblinear solver only if dual=True
     X, y = make_classification(n_samples=20, random_state=0)
diff --git a/sklearn/linear_model/tests/test_ransac.py b/sklearn/linear_model/tests/test_ransac.py
--- a/sklearn/linear_model/tests/test_ransac.py
+++ b/sklearn/linear_model/tests/test_ransac.py
@@ -13,6 +13,7 @@
 from sklearn.utils.testing import assert_raises
 from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso
 from sklearn.linear_model.ransac import _dynamic_max_trials
+from sklearn.exceptions import ConvergenceWarning
 
 
 # Generate coordinates of line
@@ -230,7 +231,7 @@ def is_data_valid(X, y):
                                        max_skips=3,
                                        max_trials=5)
 
-    assert_warns(UserWarning, ransac_estimator.fit, X, y)
+    assert_warns(ConvergenceWarning, ransac_estimator.fit, X, y)
     assert_equal(ransac_estimator.n_skips_no_inliers_, 0)
     assert_equal(ransac_estimator.n_skips_invalid_data_, 4)
     assert_equal(ransac_estimator.n_skips_invalid_model_, 0)
diff --git a/sklearn/linear_model/tests/test_ridge.py b/sklearn/linear_model/tests/test_ridge.py
--- a/sklearn/linear_model/tests/test_ridge.py
+++ b/sklearn/linear_model/tests/test_ridge.py
@@ -14,6 +14,8 @@
 from sklearn.utils.testing import ignore_warnings
 from sklearn.utils.testing import assert_warns
 
+from sklearn.exceptions import ConvergenceWarning
+
 from sklearn import datasets
 from sklearn.metrics import mean_squared_error
 from sklearn.metrics import make_scorer
@@ -137,6 +139,16 @@ def test_ridge_regression_sample_weights():
                 assert_array_almost_equal(coefs, coefs2)
 
 
+def test_ridge_regression_convergence_fail():
+    rng = np.random.RandomState(0)
+    y = rng.randn(5)
+    X = rng.randn(5, 10)
+
+    assert_warns(ConvergenceWarning, ridge_regression,
+                 X, y, alpha=1.0, solver="sparse_cg",
+                 tol=0., max_iter=None, verbose=1)
+
+
 def test_ridge_sample_weights():
     # TODO: loop over sparse data as well
 

```


## Code snippets

### 1 - sklearn/exceptions.py:

Start line: 40, End line: 71

```python
class ChangedBehaviorWarning(UserWarning):
    """Warning class used to notify the user of any change in the behavior.

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class DataConversionWarning(UserWarning):
    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """
```
### 2 - sklearn/exceptions.py:

Start line: 74, End line: 96

```python
class DataDimensionalityWarning(UserWarning):
    """Custom warning to notify potential issues with data dimensionality.

    For example, in random projection, this warning is raised when the
    number of components, which quantifies the dimensionality of the target
    projection space, is higher than the number of features, which quantifies
    the dimensionality of the original source space, to imply that the
    dimensionality of the problem will not be reduced.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


class EfficiencyWarning(UserWarning):
    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18
    """
```
### 3 - sklearn/exceptions.py:

Start line: 99, End line: 128

```python
class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import FitFailedWarning
    >>> import warnings
    >>> warnings.simplefilter('always', FitFailedWarning)
    >>> gs = GridSearchCV(LinearSVC(), {'C': [-1, -2]}, error_score=0, cv=2)
    >>> X, y = [[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1]
    >>> with warnings.catch_warnings(record=True) as w:
    ...     try:
    ...         gs.fit(X, y)   # This will raise a ValueError since C is < 0
    ...     except ValueError:
    ...         pass
    ...     print(repr(w[-1].message))
    ... # doctest: +NORMALIZE_WHITESPACE
    FitFailedWarning('Estimator fit failed. The score on this train-test
    partition for these parameters will be set to 0.000000.
    Details: \\nValueError: Penalty term must be positive; got (C=-2)\\n',)

    .. versionchanged:: 0.18
       Moved from sklearn.cross_validation.
    """
```
### 4 - sklearn/utils/testing.py:

Start line: 117, End line: 159

```python
def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`

    Returns
    -------

    result : the return value of `func`

    """
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError("%s did not give warning: %s( is %s)"
                                 % (func.__name__, warning_class, w))
    return result
```
### 5 - sklearn/exceptions.py:

Start line: 131, End line: 157

```python
class NonBLASDotWarning(EfficiencyWarning):
    """Warning used when the dot operation does not use BLAS.

    This warning is used to notify the user that BLAS was not used for dot
    operation and hence the efficiency may be affected.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation, extends EfficiencyWarning.
    """


class SkipTestWarning(UserWarning):
    """Warning class used to notify the user of a test that was skipped.

    For example, one of the estimator checks requires a pandas import.
    If the pandas package cannot be imported, the test will be skipped rather
    than register as a failure.
    """


class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid

    .. versionchanged:: 0.18
       Moved from sklearn.base.
    """
```
### 6 - sklearn/utils/estimator_checks.py:

Start line: 1576, End line: 1618

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_classifiers(name, classifier_orig):
    if name == "NuSVC":
        # the sparse version has a parameter that doesn't do anything
        raise SkipTest("Not testing NuSVC class weight as it is ignored.")
    if name.endswith("NB"):
        # NaiveBayes classifiers have a somewhat different interface.
        # FIXME SOON!
        raise SkipTest

    for n_centers in [2, 3]:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0)

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _is_pairwise(classifier_orig):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(
            class_weight=class_weight)
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
        #      0.88 (Issue #9111)
        assert_greater(np.mean(y_pred == 0), 0.87)
```
### 7 - sklearn/utils/testing.py:

Start line: 162, End line: 228

```python
def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.

    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.

    message : str | callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.

    func : callable
        Callable object to trigger warnings.

    *args : the positional arguments to `func`.

    **kw : the keyword arguments to `func`.

    Returns
    -------
    result : the return value of `func`

    """
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Let's not catch the numpy internal DeprecationWarnings
            warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError("No warning raised for %s with class "
                                 "%s"
                                 % (func.__name__, warning_class))

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, 'args') else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:
                check_in_message = lambda msg: message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError("Did not receive the message you expected "
                                 "('%s') for <%s>, got: '%s'"
                                 % (message, func.__name__, msg))

    return result
```
### 8 - sklearn/utils/testing.py:

Start line: 231, End line: 248

```python
# To remove when we support numpy 1.7
def assert_no_warnings(func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        if len(w) > 0:
            raise AssertionError("Got warnings when calling %s: [%s]"
                                 % (func.__name__,
                                    ', '.join(str(warning) for warning in w)))
    return result
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 1145, End line: 1199

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig):
    clusterer = clone(clusterer_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
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
    assert_equal(pred.shape, (n_samples,))
    assert_greater(adjusted_rand_score(pred, y), 0.4)
    # fit another time with ``fit_predict`` and compare results
    if name == 'SpectralClustering':
        # there is no way to make Spectral clustering deterministic :(
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert_in(pred.dtype, [np.dtype('int32'), np.dtype('int64')])
    assert_in(pred2.dtype, [np.dtype('int32'), np.dtype('int64')])

    # Add noise to X to test the possible values of the labels
    rng = np.random.RandomState(7)
    X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(labels_sorted[0],
                                                labels_sorted[-1] + 1))

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert_true(labels_sorted[0] in [0, -1])
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert_greater_equal(n_clusters - 1, labels_sorted[-1])
    # else labels should be less than max(labels_) which is necessarily true
```
### 10 - sklearn/utils/estimator_checks.py:

Start line: 1253, End line: 1360

```python
@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            X -= X.min()
        X = pairwise_estimator_convert_X(X, classifier_orig)
        set_random_state(classifier)
        # raises error on malformed input for fit
        with assert_raises(ValueError, msg="The classifer {} does not"
                           " raise an error when incorrect/malformed input "
                           "data for fit is passed. The number of training "
                           "examples is not the same as the number of labels."
                           " Perhaps use check_X_y in fit.".format(name)):
            classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert_true(hasattr(classifier, "classes_"))
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        if name not in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        if _is_pairwise(classifier):
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when shape of X"
                               "in predict is not equal to (n_test_samples,"
                               "n_training_samples)".format(name)):
                classifier.predict(X.reshape(-1, 1))
        else:
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when the number of features "
                               "in predict is different from the number of"
                               " features in fit.".format(name)):
                classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    assert_equal(decision.shape, (n_samples,))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                if (n_classes == 3 and
                        # 1on1 of LibSVM works differently
                        not isinstance(classifier, BaseLibSVM)):
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if _is_pairwise(classifier):
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the  "
                                       "shape of X in decision_function is "
                                       "not equal to (n_test_samples, "
                                       "n_training_samples) in fit."
                                       .format(name)):
                        classifier.decision_function(X.reshape(-1, 1))
                else:
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the number "
                                       "of features in decision_function is "
                                       "different from the number of features"
                                       " in fit.".format(name)):
                        classifier.decision_function(X.T)
            except NotImplementedError:
                pass
        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert_equal(y_prob.shape, (n_samples, n_classes))
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
            # raises error on malformed input for predict_proba
            if _is_pairwise(classifier_orig):
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the shape of X"
                                   "in predict_proba is not equal to "
                                   "(n_test_samples, n_training_samples)."
                                   .format(name)):
                    classifier.predict_proba(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the number of "
                                   "features in predict_proba is different "
                                   "from the number of features in fit."
                                   .format(name)):
                    classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))
```
### 11 - sklearn/cluster/birch.py:

Start line: 297, End line: 321

```python
class _CFSubcluster(object):

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,
             self.centroid_, self.sq_norm_) = \
                new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False

    @property
    def radius(self):
        """Return radius of the subcluster"""
        dot_product = -2 * np.dot(self.linear_sum_, self.centroid_)
        return sqrt(
            ((self.squared_sum_ + dot_product) / self.n_samples_) +
            self.sq_norm_)
```
### 35 - sklearn/cluster/birch.py:

Start line: 540, End line: 553

```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):

    def _check_fit(self, X):
        is_fitted = hasattr(self, 'subcluster_centers_')

        # Called by partial_fit, before fitting.
        has_partial_fit = hasattr(self, 'partial_fit_')

        # Should raise an error if one does not fit before predicting.
        if not (is_fitted or has_partial_fit):
            raise NotFittedError("Fit training data before predicting")

        if is_fitted and X.shape[1] != self.subcluster_centers_.shape[1]:
            raise ValueError(
                "Training data and predicted data do "
                "not have same number of features.")
```
### 42 - sklearn/cluster/affinity_propagation_.py:

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
### 47 - sklearn/cluster/affinity_propagation_.py:

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
### 56 - sklearn/cluster/birch.py:

Start line: 451, End line: 500

```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):

    def _fit(self, X):
        X = check_array(X, accept_sparse='csr', copy=self.copy)
        threshold = self.threshold
        branching_factor = self.branching_factor

        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")
        n_samples, n_features = X.shape

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        partial_fit = getattr(self, 'partial_fit_')
        has_root = getattr(self, 'root_', None)
        if getattr(self, 'fit_') or (partial_fit and not has_root):
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _CFNode(threshold, branching_factor, is_leaf=True,
                                 n_features=n_features)

            # To enable getting back subclusters.
            self.dummy_leaf_ = _CFNode(threshold, branching_factor,
                                       is_leaf=True, n_features=n_features)
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample in iter_func(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold, branching_factor,
                                     is_leaf=False,
                                     n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        centroids = np.concatenate([
            leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids

        self._global_clustering(X)
        return self
```
### 61 - sklearn/cluster/birch.py:

Start line: 324, End line: 426

```python
class Birch(BaseEstimator, TransformerMixin, ClusterMixin):
    """Implements the Birch clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    Parameters
    ----------
    threshold : float, default 0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default 50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model, default 3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - `sklearn.cluster` Estimator : If a model is provided, the model is
          fit treating the subclusters as new samples and the initial data is
          mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default True
        Whether or not to compute labels for each fit.

    copy : bool, default True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray,
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray,
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray, shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,
    ... compute_labels=True)
    >>> brc.fit(X)
    Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,
       threshold=0.5)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.
    """
```
### 76 - sklearn/decomposition/fastica_.py:

Start line: 98, End line: 121

```python
def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in moves.xrange(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_
                                - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge. Consider increasing '
                      'tolerance or the maximum number of iterations.')

    return W, ii + 1
```

# scikit-learn__scikit-learn-10306

* repo: scikit-learn/scikit-learn
* base_commit: `b90661d6a46aa3619d3eec94d5281f5888add501`

## Problem statement

Some UserWarnings should be ConvergenceWarnings
Some warnings raised during testing show that we do not use `ConvergenceWarning` when it is appropriate in some cases. For example (from [here](https://github.com/scikit-learn/scikit-learn/issues/10158#issuecomment-345453334)):

```python
/home/lesteve/dev/alt-scikit-learn/sklearn/decomposition/fastica_.py:118: UserWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
/home/lesteve/dev/alt-scikit-learn/sklearn/cluster/birch.py:629: UserWarning: Number of subclusters found (2) by Birch is less than (3). Decrease the threshold.
```

These should be changed, at least. For bonus points, the contributor could look for other warning messages that mention "converge".


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

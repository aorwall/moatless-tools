# scikit-learn__scikit-learn-13302

* repo: scikit-learn/scikit-learn
* base_commit: `4de404d46d24805ff48ad255ec3169a5155986f0`

## Problem statement

[WIP] EHN: Ridge with solver SAG/SAGA does not cast to float64
closes #11642 

build upon #11155 

TODO:

- [ ] Merge #11155 to reduce the diff.
- [ ] Ensure that the casting rule is clear between base classes, classes and functions. I suspect that we have some copy which are not useful.



## Patch

```diff
diff --git a/sklearn/linear_model/ridge.py b/sklearn/linear_model/ridge.py
--- a/sklearn/linear_model/ridge.py
+++ b/sklearn/linear_model/ridge.py
@@ -226,9 +226,17 @@ def _solve_svd(X, y, alpha):
     return np.dot(Vt.T, d_UT_y).T
 
 
+def _get_valid_accept_sparse(is_X_sparse, solver):
+    if is_X_sparse and solver in ['auto', 'sag', 'saga']:
+        return 'csr'
+    else:
+        return ['csr', 'csc', 'coo']
+
+
 def ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                      max_iter=None, tol=1e-3, verbose=0, random_state=None,
-                     return_n_iter=False, return_intercept=False):
+                     return_n_iter=False, return_intercept=False,
+                     check_input=True):
     """Solve the ridge equation by the method of normal equations.
 
     Read more in the :ref:`User Guide <ridge_regression>`.
@@ -332,6 +340,11 @@ def ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
 
         .. versionadded:: 0.17
 
+    check_input : boolean, default True
+        If False, the input arrays X and y will not be checked.
+
+        .. versionadded:: 0.21
+
     Returns
     -------
     coef : array, shape = [n_features] or [n_targets, n_features]
@@ -360,13 +373,14 @@ def ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                              return_n_iter=return_n_iter,
                              return_intercept=return_intercept,
                              X_scale=None,
-                             X_offset=None)
+                             X_offset=None,
+                             check_input=check_input)
 
 
 def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                       max_iter=None, tol=1e-3, verbose=0, random_state=None,
                       return_n_iter=False, return_intercept=False,
-                      X_scale=None, X_offset=None):
+                      X_scale=None, X_offset=None, check_input=True):
 
     has_sw = sample_weight is not None
 
@@ -388,17 +402,12 @@ def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                          "intercept. Please change solver to 'sag' or set "
                          "return_intercept=False.")
 
-    _dtype = [np.float64, np.float32]
-
-    # SAG needs X and y columns to be C-contiguous and np.float64
-    if solver in ['sag', 'saga']:
-        X = check_array(X, accept_sparse=['csr'],
-                        dtype=np.float64, order='C')
-        y = check_array(y, dtype=np.float64, ensure_2d=False, order='F')
-    else:
-        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
-                        dtype=_dtype)
-        y = check_array(y, dtype=X.dtype, ensure_2d=False)
+    if check_input:
+        _dtype = [np.float64, np.float32]
+        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X), solver)
+        X = check_array(X, accept_sparse=_accept_sparse, dtype=_dtype,
+                        order="C")
+        y = check_array(y, dtype=X.dtype, ensure_2d=False, order="C")
     check_consistent_length(X, y)
 
     n_samples, n_features = X.shape
@@ -417,8 +426,6 @@ def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
         raise ValueError("Number of samples in X and y does not correspond:"
                          " %d != %d" % (n_samples, n_samples_))
 
-
-
     if has_sw:
         if np.atleast_1d(sample_weight).ndim > 1:
             raise ValueError("Sample weights must be 1D array or scalar")
@@ -438,7 +445,6 @@ def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
     if alpha.size == 1 and n_targets > 1:
         alpha = np.repeat(alpha, n_targets)
 
-
     n_iter = None
     if solver == 'sparse_cg':
         coef = _solve_sparse_cg(X, y, alpha,
@@ -461,7 +467,6 @@ def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
             except linalg.LinAlgError:
                 # use SVD solver if matrix is singular
                 solver = 'svd'
-
         else:
             try:
                 coef = _solve_cholesky(X, y, alpha)
@@ -473,11 +478,12 @@ def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
         # precompute max_squared_sum for all targets
         max_squared_sum = row_norms(X, squared=True).max()
 
-        coef = np.empty((y.shape[1], n_features))
+        coef = np.empty((y.shape[1], n_features), dtype=X.dtype)
         n_iter = np.empty(y.shape[1], dtype=np.int32)
-        intercept = np.zeros((y.shape[1], ))
+        intercept = np.zeros((y.shape[1], ), dtype=X.dtype)
         for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
-            init = {'coef': np.zeros((n_features + int(return_intercept), 1))}
+            init = {'coef': np.zeros((n_features + int(return_intercept), 1),
+                                     dtype=X.dtype)}
             coef_, n_iter_, _ = sag_solver(
                 X, target.ravel(), sample_weight, 'squared', alpha_i, 0,
                 max_iter, tol, verbose, random_state, False, max_squared_sum,
@@ -530,13 +536,13 @@ def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
 
     def fit(self, X, y, sample_weight=None):
 
-        if self.solver in ('sag', 'saga'):
-            _dtype = np.float64
-        else:
-            # all other solvers work at both float precision levels
-            _dtype = [np.float64, np.float32]
-
-        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=_dtype,
+        # all other solvers work at both float precision levels
+        _dtype = [np.float64, np.float32]
+        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X),
+                                                  self.solver)
+        X, y = check_X_y(X, y,
+                         accept_sparse=_accept_sparse,
+                         dtype=_dtype,
                          multi_output=True, y_numeric=True)
 
         if ((sample_weight is not None) and
@@ -555,7 +561,7 @@ def fit(self, X, y, sample_weight=None):
                 X, y, alpha=self.alpha, sample_weight=sample_weight,
                 max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                 random_state=self.random_state, return_n_iter=True,
-                return_intercept=True)
+                return_intercept=True, check_input=False)
             # add the offset which was subtracted by _preprocess_data
             self.intercept_ += y_offset
         else:
@@ -570,8 +576,7 @@ def fit(self, X, y, sample_weight=None):
                 X, y, alpha=self.alpha, sample_weight=sample_weight,
                 max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                 random_state=self.random_state, return_n_iter=True,
-                return_intercept=False, **params)
-
+                return_intercept=False, check_input=False, **params)
             self._set_intercept(X_offset, y_offset, X_scale)
 
         return self
@@ -893,8 +898,9 @@ def fit(self, X, y, sample_weight=None):
         -------
         self : returns an instance of self.
         """
-        check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
-                  multi_output=True)
+        _accept_sparse = _get_valid_accept_sparse(sparse.issparse(X),
+                                                  self.solver)
+        check_X_y(X, y, accept_sparse=_accept_sparse, multi_output=True)
 
         self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
         Y = self._label_binarizer.fit_transform(y)
@@ -1077,10 +1083,13 @@ def fit(self, X, y, sample_weight=None):
         -------
         self : object
         """
-        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
+        X, y = check_X_y(X, y,
+                         accept_sparse=['csr', 'csc', 'coo'],
+                         dtype=[np.float64, np.float32],
                          multi_output=True, y_numeric=True)
         if sample_weight is not None and not isinstance(sample_weight, float):
-            sample_weight = check_array(sample_weight, ensure_2d=False)
+            sample_weight = check_array(sample_weight, ensure_2d=False,
+                                        dtype=X.dtype)
         n_samples, n_features = X.shape
 
         X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_ridge.py b/sklearn/linear_model/tests/test_ridge.py
--- a/sklearn/linear_model/tests/test_ridge.py
+++ b/sklearn/linear_model/tests/test_ridge.py
@@ -1,3 +1,4 @@
+import os
 import numpy as np
 import scipy.sparse as sp
 from scipy import linalg
@@ -6,6 +7,7 @@
 import pytest
 
 from sklearn.utils.testing import assert_almost_equal
+from sklearn.utils.testing import assert_allclose
 from sklearn.utils.testing import assert_array_almost_equal
 from sklearn.utils.testing import assert_allclose
 from sklearn.utils.testing import assert_equal
@@ -38,7 +40,7 @@
 from sklearn.model_selection import GridSearchCV
 from sklearn.model_selection import KFold
 
-from sklearn.utils import check_random_state
+from sklearn.utils import check_random_state, _IS_32BIT
 from sklearn.datasets import make_multilabel_classification
 
 diabetes = datasets.load_diabetes()
@@ -934,7 +936,9 @@ def test_ridge_classifier_no_support_multilabel():
     assert_raises(ValueError, RidgeClassifier().fit, X, y)
 
 
-def test_dtype_match():
+@pytest.mark.parametrize(
+    "solver", ["svd", "sparse_cg", "cholesky", "lsqr", "sag", "saga"])
+def test_dtype_match(solver):
     rng = np.random.RandomState(0)
     alpha = 1.0
 
@@ -944,25 +948,22 @@ def test_dtype_match():
     X_32 = X_64.astype(np.float32)
     y_32 = y_64.astype(np.float32)
 
-    solvers = ["svd", "sparse_cg", "cholesky", "lsqr"]
-    for solver in solvers:
-
-        # Check type consistency 32bits
-        ridge_32 = Ridge(alpha=alpha, solver=solver)
-        ridge_32.fit(X_32, y_32)
-        coef_32 = ridge_32.coef_
+    # Check type consistency 32bits
+    ridge_32 = Ridge(alpha=alpha, solver=solver, max_iter=500, tol=1e-10,)
+    ridge_32.fit(X_32, y_32)
+    coef_32 = ridge_32.coef_
 
-        # Check type consistency 64 bits
-        ridge_64 = Ridge(alpha=alpha, solver=solver)
-        ridge_64.fit(X_64, y_64)
-        coef_64 = ridge_64.coef_
+    # Check type consistency 64 bits
+    ridge_64 = Ridge(alpha=alpha, solver=solver, max_iter=500, tol=1e-10,)
+    ridge_64.fit(X_64, y_64)
+    coef_64 = ridge_64.coef_
 
-        # Do the actual checks at once for easier debug
-        assert coef_32.dtype == X_32.dtype
-        assert coef_64.dtype == X_64.dtype
-        assert ridge_32.predict(X_32).dtype == X_32.dtype
-        assert ridge_64.predict(X_64).dtype == X_64.dtype
-        assert_almost_equal(ridge_32.coef_, ridge_64.coef_, decimal=5)
+    # Do the actual checks at once for easier debug
+    assert coef_32.dtype == X_32.dtype
+    assert coef_64.dtype == X_64.dtype
+    assert ridge_32.predict(X_32).dtype == X_32.dtype
+    assert ridge_64.predict(X_64).dtype == X_64.dtype
+    assert_allclose(ridge_32.coef_, ridge_64.coef_, rtol=1e-4)
 
 
 def test_dtype_match_cholesky():
@@ -993,3 +994,32 @@ def test_dtype_match_cholesky():
     assert ridge_32.predict(X_32).dtype == X_32.dtype
     assert ridge_64.predict(X_64).dtype == X_64.dtype
     assert_almost_equal(ridge_32.coef_, ridge_64.coef_, decimal=5)
+
+
+@pytest.mark.parametrize(
+    'solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
+def test_ridge_regression_dtype_stability(solver):
+    random_state = np.random.RandomState(0)
+    n_samples, n_features = 6, 5
+    X = random_state.randn(n_samples, n_features)
+    coef = random_state.randn(n_features)
+    y = np.dot(X, coef) + 0.01 * rng.randn(n_samples)
+    alpha = 1.0
+    rtol = 1e-2 if os.name == 'nt' and _IS_32BIT else 1e-5
+
+    results = dict()
+    for current_dtype in (np.float32, np.float64):
+        results[current_dtype] = ridge_regression(X.astype(current_dtype),
+                                                  y.astype(current_dtype),
+                                                  alpha=alpha,
+                                                  solver=solver,
+                                                  random_state=random_state,
+                                                  sample_weight=None,
+                                                  max_iter=500,
+                                                  tol=1e-10,
+                                                  return_n_iter=False,
+                                                  return_intercept=False)
+
+    assert results[np.float32].dtype == np.float32
+    assert results[np.float64].dtype == np.float64
+    assert_allclose(results[np.float32], results[np.float64], rtol=rtol)

```

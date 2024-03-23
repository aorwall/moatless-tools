# scikit-learn__scikit-learn-10452

* repo: scikit-learn/scikit-learn
* base_commit: `3e5469eda719956c076ae8e685ec1183bfd98569`

## Problem statement

Polynomial Features for sparse data
I'm not sure if that came up before but PolynomialFeatures doesn't support sparse data, which is not great. Should be easy but I haven't checked ;)


## Patch

```diff
diff --git a/sklearn/preprocessing/data.py b/sklearn/preprocessing/data.py
--- a/sklearn/preprocessing/data.py
+++ b/sklearn/preprocessing/data.py
@@ -1325,7 +1325,7 @@ def fit(self, X, y=None):
         -------
         self : instance
         """
-        n_samples, n_features = check_array(X).shape
+        n_samples, n_features = check_array(X, accept_sparse=True).shape
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
@@ -1338,31 +1338,42 @@ def transform(self, X):
 
         Parameters
         ----------
-        X : array-like, shape [n_samples, n_features]
+        X : array-like or sparse matrix, shape [n_samples, n_features]
             The data to transform, row by row.
+            Sparse input should preferably be in CSC format.
 
         Returns
         -------
-        XP : np.ndarray shape [n_samples, NP]
+        XP : np.ndarray or CSC sparse matrix, shape [n_samples, NP]
             The matrix of features, where NP is the number of polynomial
             features generated from the combination of inputs.
         """
         check_is_fitted(self, ['n_input_features_', 'n_output_features_'])
 
-        X = check_array(X, dtype=FLOAT_DTYPES)
+        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse='csc')
         n_samples, n_features = X.shape
 
         if n_features != self.n_input_features_:
             raise ValueError("X shape does not match training shape")
 
-        # allocate output data
-        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
-
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
-        for i, c in enumerate(combinations):
-            XP[:, i] = X[:, c].prod(1)
+        if sparse.isspmatrix(X):
+            columns = []
+            for comb in combinations:
+                if comb:
+                    out_col = 1
+                    for col_idx in comb:
+                        out_col = X[:, col_idx].multiply(out_col)
+                    columns.append(out_col)
+                else:
+                    columns.append(sparse.csc_matrix(np.ones((X.shape[0], 1))))
+            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
+        else:
+            XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
+            for i, comb in enumerate(combinations):
+                XP[:, i] = X[:, comb].prod(1)
 
         return XP
 

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_data.py b/sklearn/preprocessing/tests/test_data.py
--- a/sklearn/preprocessing/tests/test_data.py
+++ b/sklearn/preprocessing/tests/test_data.py
@@ -7,10 +7,12 @@
 
 import warnings
 import re
+
 import numpy as np
 import numpy.linalg as la
 from scipy import sparse, stats
 from distutils.version import LooseVersion
+import pytest
 
 from sklearn.utils import gen_batches
 
@@ -155,6 +157,28 @@ def test_polynomial_feature_names():
                        feature_names)
 
 
+@pytest.mark.parametrize(['deg', 'include_bias', 'interaction_only', 'dtype'],
+                         [(1, True, False, int),
+                          (2, True, False, int),
+                          (2, True, False, np.float32),
+                          (2, True, False, np.float64),
+                          (3, False, False, np.float64),
+                          (3, False, True, np.float64)])
+def test_polynomial_features_sparse_X(deg, include_bias, interaction_only,
+                                      dtype):
+    rng = np.random.RandomState(0)
+    X = rng.randint(0, 2, (100, 2))
+    X_sparse = sparse.csr_matrix(X)
+
+    est = PolynomialFeatures(deg, include_bias=include_bias)
+    Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
+    Xt_dense = est.fit_transform(X.astype(dtype))
+
+    assert isinstance(Xt_sparse, sparse.csc_matrix)
+    assert Xt_sparse.dtype == Xt_dense.dtype
+    assert_array_almost_equal(Xt_sparse.A, Xt_dense)
+
+
 def test_standard_scaler_1d():
     # Test scaling of dataset along single axis
     for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

```

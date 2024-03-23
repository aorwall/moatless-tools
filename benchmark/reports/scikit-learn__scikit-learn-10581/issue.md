# scikit-learn__scikit-learn-10581

* repo: scikit-learn/scikit-learn
* base_commit: `b27e285ea39450550fc8c81f308a91a660c03a56`

## Problem statement

ElasticNet overwrites X even with copy_X=True
The `fit` function of an `ElasticNet`, called with `check_input=False`, overwrites X, even when `copy_X=True`:
```python
import numpy as np
from sklearn.linear_model import ElasticNet


rng = np.random.RandomState(0)
n_samples, n_features = 20, 2
X = rng.randn(n_samples, n_features).copy(order='F')
beta = rng.randn(n_features)
y = 2 + np.dot(X, beta) + rng.randn(n_samples)

X_copy = X.copy()
enet = ElasticNet(fit_intercept=True, normalize=False, copy_X=True)
enet.fit(X, y, check_input=False)

print("X unchanged = ", np.all(X == X_copy))
```
ElasticNet overwrites X even with copy_X=True
The `fit` function of an `ElasticNet`, called with `check_input=False`, overwrites X, even when `copy_X=True`:
```python
import numpy as np
from sklearn.linear_model import ElasticNet


rng = np.random.RandomState(0)
n_samples, n_features = 20, 2
X = rng.randn(n_samples, n_features).copy(order='F')
beta = rng.randn(n_features)
y = 2 + np.dot(X, beta) + rng.randn(n_samples)

X_copy = X.copy()
enet = ElasticNet(fit_intercept=True, normalize=False, copy_X=True)
enet.fit(X, y, check_input=False)

print("X unchanged = ", np.all(X == X_copy))
```
[MRG] FIX #10540 ElasticNet overwrites X even with copy_X=True
Made changes as suggested by @gxyd.
please review and suggest changes @jnothman @gxyd 


## Patch

```diff
diff --git a/sklearn/linear_model/coordinate_descent.py b/sklearn/linear_model/coordinate_descent.py
--- a/sklearn/linear_model/coordinate_descent.py
+++ b/sklearn/linear_model/coordinate_descent.py
@@ -700,19 +700,23 @@ def fit(self, X, y, check_input=True):
             raise ValueError('precompute should be one of True, False or'
                              ' array-like. Got %r' % self.precompute)
 
+        # Remember if X is copied
+        X_copied = False
         # We expect X and y to be float64 or float32 Fortran ordered arrays
         # when bypassing checks
         if check_input:
+            X_copied = self.copy_X and self.fit_intercept
             X, y = check_X_y(X, y, accept_sparse='csc',
                              order='F', dtype=[np.float64, np.float32],
-                             copy=self.copy_X and self.fit_intercept,
-                             multi_output=True, y_numeric=True)
+                             copy=X_copied, multi_output=True, y_numeric=True)
             y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                             ensure_2d=False)
 
+        # Ensure copying happens only once, don't do it again if done above
+        should_copy = self.copy_X and not X_copied
         X, y, X_offset, y_offset, X_scale, precompute, Xy = \
             _pre_fit(X, y, None, self.precompute, self.normalize,
-                     self.fit_intercept, copy=False)
+                     self.fit_intercept, copy=should_copy)
         if y.ndim == 1:
             y = y[:, np.newaxis]
         if Xy is not None and Xy.ndim == 1:

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_coordinate_descent.py b/sklearn/linear_model/tests/test_coordinate_descent.py
--- a/sklearn/linear_model/tests/test_coordinate_descent.py
+++ b/sklearn/linear_model/tests/test_coordinate_descent.py
@@ -3,6 +3,7 @@
 # License: BSD 3 clause
 
 import numpy as np
+import pytest
 from scipy import interpolate, sparse
 from copy import deepcopy
 
@@ -669,6 +670,30 @@ def test_check_input_false():
     assert_raises(ValueError, clf.fit, X, y, check_input=False)
 
 
+@pytest.mark.parametrize("check_input", [True, False])
+def test_enet_copy_X_True(check_input):
+    X, y, _, _ = build_dataset()
+    X = X.copy(order='F')
+
+    original_X = X.copy()
+    enet = ElasticNet(copy_X=True)
+    enet.fit(X, y, check_input=check_input)
+
+    assert_array_equal(original_X, X)
+
+
+def test_enet_copy_X_False_check_input_False():
+    X, y, _, _ = build_dataset()
+    X = X.copy(order='F')
+
+    original_X = X.copy()
+    enet = ElasticNet(copy_X=False)
+    enet.fit(X, y, check_input=False)
+
+    # No copying, X is overwritten
+    assert_true(np.any(np.not_equal(original_X, X)))
+
+
 def test_overrided_gram_matrix():
     X, y, _, _ = build_dataset(n_samples=20, n_features=10)
     Gram = X.T.dot(X)

```

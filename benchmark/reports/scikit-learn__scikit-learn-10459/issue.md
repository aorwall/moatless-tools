# scikit-learn__scikit-learn-10459

* repo: scikit-learn/scikit-learn
* base_commit: `2e85c8608c93ad0e3290414c4e5e650b87d44b27`

## Problem statement

[RFC] Dissociate NaN and Inf when considering force_all_finite in check_array
Due to changes proposed in #10404, it seems that `check_array` as currently a main limitation. `force_all_finite` will force both `NaN` and `inf`to be rejected. If preprocessing methods (whenever this is possible) should let pass `NaN`, this argument is not enough permissive.

Before to implement anything, I think it could be good to have some feedback on the way to go. I see the following solutions:

1. `force_all_finite` could still accept a bool to preserve the behaviour. Additionally, it could accept an `str` to filter only `inf`.
2. #7892 proposes to have an additional argument `allow_nan`. @amueller was worried that it makes `check_array` to complex.
3. make a private function `_assert_finite_or_nan` (similarly to [this proposal](https://github.com/scikit-learn/scikit-learn/pull/10437/files#diff-5ebddebc20987b6125fffc893f5abc4cR2379) removing the numpy version checking) in the `data.py` which can be shared between the preprocessing methods.

They are the solutions that I have in mind for the moment but anything else is welcomed.
@jnothman @agramfort @amueller @lesteve @ogrisel @GaelVaroquaux I would be grateful for any insight.


## Patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -31,7 +31,7 @@
 warnings.simplefilter('ignore', NonBLASDotWarning)
 
 
-def _assert_all_finite(X):
+def _assert_all_finite(X, allow_nan=False):
     """Like assert_all_finite, but only for ndarray."""
     if _get_config()['assume_finite']:
         return
@@ -39,20 +39,27 @@ def _assert_all_finite(X):
     # First try an O(n) time, O(1) space solution for the common case that
     # everything is finite; fall back to O(n) space np.isfinite to prevent
     # false positives from overflow in sum method.
-    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
-            and not np.isfinite(X).all()):
-        raise ValueError("Input contains NaN, infinity"
-                         " or a value too large for %r." % X.dtype)
-
-
-def assert_all_finite(X):
+    is_float = X.dtype.kind in 'fc'
+    if is_float and np.isfinite(X.sum()):
+        pass
+    elif is_float:
+        msg_err = "Input contains {} or a value too large for {!r}."
+        if (allow_nan and np.isinf(X).any() or
+                not allow_nan and not np.isfinite(X).all()):
+            type_err = 'infinity' if allow_nan else 'NaN, infinity'
+            raise ValueError(msg_err.format(type_err, X.dtype))
+
+
+def assert_all_finite(X, allow_nan=False):
     """Throw a ValueError if X contains NaN or infinity.
 
     Parameters
     ----------
     X : array or sparse matrix
+
+    allow_nan : bool
     """
-    _assert_all_finite(X.data if sp.issparse(X) else X)
+    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)
 
 
 def as_float_array(X, copy=True, force_all_finite=True):
@@ -70,8 +77,17 @@ def as_float_array(X, copy=True, force_all_finite=True):
         If True, a copy of X will be created. If False, a copy may still be
         returned if X's dtype is not a floating point type.
 
-    force_all_finite : boolean (default=True)
-        Whether to raise an error on np.inf and np.nan in X.
+    force_all_finite : boolean or 'allow-nan', (default=True)
+        Whether to raise an error on np.inf and np.nan in X. The possibilities
+        are:
+
+        - True: Force all values of X to be finite.
+        - False: accept both np.inf and np.nan in X.
+        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
+          infinite.
+
+        .. versionadded:: 0.20
+           ``force_all_finite`` accepts the string ``'allow-nan'``.
 
     Returns
     -------
@@ -256,8 +272,17 @@ def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
         Whether a forced copy will be triggered. If copy=False, a copy might
         be triggered by a conversion.
 
-    force_all_finite : boolean
-        Whether to raise an error on np.inf and np.nan in X.
+    force_all_finite : boolean or 'allow-nan', (default=True)
+        Whether to raise an error on np.inf and np.nan in X. The possibilities
+        are:
+
+        - True: Force all values of X to be finite.
+        - False: accept both np.inf and np.nan in X.
+        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
+          infinite.
+
+        .. versionadded:: 0.20
+           ``force_all_finite`` accepts the string ``'allow-nan'``.
 
     Returns
     -------
@@ -304,7 +329,9 @@ def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
             warnings.warn("Can't check %s sparse matrix for nan or inf."
                           % spmatrix.format)
         else:
-            _assert_all_finite(spmatrix.data)
+            _assert_all_finite(spmatrix.data,
+                               allow_nan=force_all_finite == 'allow-nan')
+
     return spmatrix
 
 
@@ -359,8 +386,17 @@ def check_array(array, accept_sparse=False, dtype="numeric", order=None,
         Whether a forced copy will be triggered. If copy=False, a copy might
         be triggered by a conversion.
 
-    force_all_finite : boolean (default=True)
-        Whether to raise an error on np.inf and np.nan in X.
+    force_all_finite : boolean or 'allow-nan', (default=True)
+        Whether to raise an error on np.inf and np.nan in X. The possibilities
+        are:
+
+        - True: Force all values of X to be finite.
+        - False: accept both np.inf and np.nan in X.
+        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
+          infinite.
+
+        .. versionadded:: 0.20
+           ``force_all_finite`` accepts the string ``'allow-nan'``.
 
     ensure_2d : boolean (default=True)
         Whether to raise a value error if X is not 2d.
@@ -425,6 +461,10 @@ def check_array(array, accept_sparse=False, dtype="numeric", order=None,
             # list of accepted types.
             dtype = dtype[0]
 
+    if force_all_finite not in (True, False, 'allow-nan'):
+        raise ValueError('force_all_finite should be a bool or "allow-nan"'
+                         '. Got {!r} instead'.format(force_all_finite))
+
     if estimator is not None:
         if isinstance(estimator, six.string_types):
             estimator_name = estimator
@@ -483,7 +523,8 @@ def check_array(array, accept_sparse=False, dtype="numeric", order=None,
             raise ValueError("Found array with dim %d. %s expected <= 2."
                              % (array.ndim, estimator_name))
         if force_all_finite:
-            _assert_all_finite(array)
+            _assert_all_finite(array,
+                               allow_nan=force_all_finite == 'allow-nan')
 
     shape_repr = _shape_repr(array.shape)
     if ensure_min_samples > 0:
@@ -555,9 +596,18 @@ def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
         Whether a forced copy will be triggered. If copy=False, a copy might
         be triggered by a conversion.
 
-    force_all_finite : boolean (default=True)
+    force_all_finite : boolean or 'allow-nan', (default=True)
         Whether to raise an error on np.inf and np.nan in X. This parameter
         does not influence whether y can have np.inf or np.nan values.
+        The possibilities are:
+
+        - True: Force all values of X to be finite.
+        - False: accept both np.inf and np.nan in X.
+        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
+          infinite.
+
+        .. versionadded:: 0.20
+           ``force_all_finite`` accepts the string ``'allow-nan'``.
 
     ensure_2d : boolean (default=True)
         Whether to make X at least 2d.

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -6,8 +6,8 @@
 from tempfile import NamedTemporaryFile
 from itertools import product
 
+import pytest
 import numpy as np
-from numpy.testing import assert_array_equal
 import scipy.sparse as sp
 
 from sklearn.utils.testing import assert_true, assert_false, assert_equal
@@ -18,6 +18,8 @@
 from sklearn.utils.testing import assert_warns
 from sklearn.utils.testing import ignore_warnings
 from sklearn.utils.testing import SkipTest
+from sklearn.utils.testing import assert_array_equal
+from sklearn.utils.testing import assert_allclose_dense_sparse
 from sklearn.utils import as_float_array, check_array, check_symmetric
 from sklearn.utils import check_X_y
 from sklearn.utils.mocking import MockDataFrame
@@ -88,6 +90,17 @@ def test_as_float_array():
         assert_false(np.isnan(M).any())
 
 
+@pytest.mark.parametrize(
+    "X",
+    [(np.random.random((10, 2))),
+     (sp.rand(10, 2).tocsr())])
+def test_as_float_array_nan(X):
+    X[5, 0] = np.nan
+    X[6, 1] = np.nan
+    X_converted = as_float_array(X, force_all_finite='allow-nan')
+    assert_allclose_dense_sparse(X_converted, X)
+
+
 def test_np_matrix():
     # Confirm that input validation code does not return np.matrix
     X = np.arange(12).reshape(3, 4)
@@ -132,6 +145,43 @@ def test_ordering():
     assert_false(X.data.flags['C_CONTIGUOUS'])
 
 
+@pytest.mark.parametrize(
+    "value, force_all_finite",
+    [(np.inf, False), (np.nan, 'allow-nan'), (np.nan, False)]
+)
+@pytest.mark.parametrize(
+    "retype",
+    [np.asarray, sp.csr_matrix]
+)
+def test_check_array_force_all_finite_valid(value, force_all_finite, retype):
+    X = retype(np.arange(4).reshape(2, 2).astype(np.float))
+    X[0, 0] = value
+    X_checked = check_array(X, force_all_finite=force_all_finite,
+                            accept_sparse=True)
+    assert_allclose_dense_sparse(X, X_checked)
+
+
+@pytest.mark.parametrize(
+    "value, force_all_finite, match_msg",
+    [(np.inf, True, 'Input contains NaN, infinity'),
+     (np.inf, 'allow-nan', 'Input contains infinity'),
+     (np.nan, True, 'Input contains NaN, infinity'),
+     (np.nan, 'allow-inf', 'force_all_finite should be a bool or "allow-nan"'),
+     (np.nan, 1, 'force_all_finite should be a bool or "allow-nan"')]
+)
+@pytest.mark.parametrize(
+    "retype",
+    [np.asarray, sp.csr_matrix]
+)
+def test_check_array_force_all_finiteinvalid(value, force_all_finite,
+                                             match_msg, retype):
+    X = retype(np.arange(4).reshape(2, 2).astype(np.float))
+    X[0, 0] = value
+    with pytest.raises(ValueError, message=match_msg):
+        check_array(X, force_all_finite=force_all_finite,
+                    accept_sparse=True)
+
+
 @ignore_warnings
 def test_check_array():
     # accept_sparse == None
@@ -153,16 +203,6 @@ def test_check_array():
     X_ndim = np.arange(8).reshape(2, 2, 2)
     assert_raises(ValueError, check_array, X_ndim)
     check_array(X_ndim, allow_nd=True)  # doesn't raise
-    # force_all_finite
-    X_inf = np.arange(4).reshape(2, 2).astype(np.float)
-    X_inf[0, 0] = np.inf
-    assert_raises(ValueError, check_array, X_inf)
-    check_array(X_inf, force_all_finite=False)  # no raise
-    # nan check
-    X_nan = np.arange(4).reshape(2, 2).astype(np.float)
-    X_nan[0, 0] = np.nan
-    assert_raises(ValueError, check_array, X_nan)
-    check_array(X_inf, force_all_finite=False)  # no raise
 
     # dtype and order enforcement.
     X_C = np.arange(4).reshape(2, 2).copy("C")

```

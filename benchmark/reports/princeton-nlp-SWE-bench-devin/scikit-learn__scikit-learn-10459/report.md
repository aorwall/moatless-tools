# scikit-learn__scikit-learn-10459

| **scikit-learn/scikit-learn** | `2e85c8608c93ad0e3290414c4e5e650b87d44b27` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 24855 |
| **Any found context length** | 203 |
| **Avg pos** | 130.0 |
| **Min pos** | 1 |
| **Max pos** | 64 |
| **Top file pos** | 1 |
| **Missing snippets** | 9 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/validation.py | 34 | 34 | 1 | 1 | 203
| sklearn/utils/validation.py | 42 | 55 | 1 | 1 | 203
| sklearn/utils/validation.py | 73 | 74 | 10 | 1 | 4168
| sklearn/utils/validation.py | 259 | 260 | 21 | 1 | 7558
| sklearn/utils/validation.py | 307 | 307 | 21 | 1 | 7558
| sklearn/utils/validation.py | 362 | 363 | 6 | 1 | 2988
| sklearn/utils/validation.py | 428 | 428 | 2 | 1 | 607
| sklearn/utils/validation.py | 486 | 486 | 4 | 1 | 2033
| sklearn/utils/validation.py | 558 | 558 | 64 | 1 | 24855


## Problem Statement

```
[RFC] Dissociate NaN and Inf when considering force_all_finite in check_array
Due to changes proposed in #10404, it seems that `check_array` as currently a main limitation. `force_all_finite` will force both `NaN` and `inf`to be rejected. If preprocessing methods (whenever this is possible) should let pass `NaN`, this argument is not enough permissive.

Before to implement anything, I think it could be good to have some feedback on the way to go. I see the following solutions:

1. `force_all_finite` could still accept a bool to preserve the behaviour. Additionally, it could accept an `str` to filter only `inf`.
2. #7892 proposes to have an additional argument `allow_nan`. @amueller was worried that it makes `check_array` to complex.
3. make a private function `_assert_finite_or_nan` (similarly to [this proposal](https://github.com/scikit-learn/scikit-learn/pull/10437/files#diff-5ebddebc20987b6125fffc893f5abc4cR2379) removing the numpy version checking) in the `data.py` which can be shared between the preprocessing methods.

They are the solutions that I have in mind for the moment but anything else is welcomed.
@jnothman @agramfort @amueller @lesteve @ogrisel @GaelVaroquaux I would be grateful for any insight.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/utils/validation.py** | 34 | 55| 203 | 203 | 6731 | 
| **-> 2 <-** | **1 sklearn/utils/validation.py** | 395 | 435| 404 | 607 | 6731 | 
| 3 | 2 sklearn/utils/estimator_checks.py | 1008 | 1076| 601 | 1208 | 24648 | 
| **-> 4 <-** | **2 sklearn/utils/validation.py** | 437 | 509| 825 | 2033 | 24648 | 
| 5 | 2 sklearn/utils/estimator_checks.py | 141 | 161| 203 | 2236 | 24648 | 
| **-> 6 <-** | **2 sklearn/utils/validation.py** | 318 | 394| 752 | 2988 | 24648 | 
| 7 | 3 sklearn/utils/testing.py | 231 | 248| 164 | 3152 | 31436 | 
| 8 | **3 sklearn/utils/validation.py** | 604 | 618| 205 | 3357 | 31436 | 
| 9 | 4 sklearn/utils/fixes.py | 1 | 71| 443 | 3800 | 33832 | 
| **-> 10 <-** | **4 sklearn/utils/validation.py** | 58 | 95| 368 | 4168 | 33832 | 
| 11 | 4 sklearn/utils/estimator_checks.py | 661 | 696| 341 | 4509 | 33832 | 
| 12 | 4 sklearn/utils/estimator_checks.py | 489 | 524| 328 | 4837 | 33832 | 
| 13 | **4 sklearn/utils/validation.py** | 190 | 205| 127 | 4964 | 33832 | 
| 14 | 5 sklearn/preprocessing/imputation.py | 4 | 32| 154 | 5118 | 36812 | 
| 15 | 5 sklearn/utils/fixes.py | 74 | 150| 753 | 5871 | 36812 | 
| 16 | 5 sklearn/utils/estimator_checks.py | 1765 | 1779| 211 | 6082 | 36812 | 
| 17 | 5 sklearn/preprocessing/imputation.py | 35 | 60| 240 | 6322 | 36812 | 
| 18 | 6 sklearn/feature_selection/univariate_selection.py | 1 | 32| 197 | 6519 | 42915 | 
| 19 | 6 sklearn/utils/estimator_checks.py | 792 | 814| 248 | 6767 | 42915 | 
| 20 | 7 sklearn/preprocessing/data.py | 68 | 83| 129 | 6896 | 68809 | 
| **-> 21 <-** | **7 sklearn/utils/validation.py** | 234 | 315| 662 | 7558 | 68809 | 
| 22 | 7 sklearn/utils/estimator_checks.py | 1742 | 1762| 233 | 7791 | 68809 | 
| 23 | 7 sklearn/preprocessing/data.py | 10 | 65| 335 | 8126 | 68809 | 
| 24 | 8 examples/preprocessing/plot_all_scaling.py | 219 | 320| 895 | 9021 | 71911 | 
| 25 | 8 sklearn/utils/estimator_checks.py | 406 | 447| 427 | 9448 | 71911 | 
| 26 | 8 sklearn/utils/estimator_checks.py | 1576 | 1618| 449 | 9897 | 71911 | 
| 27 | 8 sklearn/preprocessing/data.py | 2386 | 2403| 192 | 10089 | 71911 | 
| 28 | 8 sklearn/utils/estimator_checks.py | 347 | 403| 300 | 10389 | 71911 | 
| 29 | 8 sklearn/utils/testing.py | 388 | 426| 348 | 10737 | 71911 | 
| 30 | 9 sklearn/externals/joblib/numpy_pickle.py | 197 | 246| 360 | 11097 | 76762 | 
| 31 | 9 sklearn/utils/estimator_checks.py | 1782 | 1802| 200 | 11297 | 76762 | 
| 32 | 9 sklearn/utils/estimator_checks.py | 1446 | 1480| 336 | 11633 | 76762 | 
| 33 | 10 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 951 | 12584 | 81146 | 
| 34 | 11 sklearn/decomposition/nmf.py | 189 | 216| 295 | 12879 | 92827 | 
| 35 | 11 sklearn/utils/fixes.py | 153 | 257| 776 | 13655 | 92827 | 
| 36 | 11 sklearn/utils/estimator_checks.py | 75 | 109| 264 | 13919 | 92827 | 
| 37 | 12 conftest.py | 1 | 15| 133 | 14052 | 92961 | 
| 38 | 13 sklearn/mixture/gaussian_mixture.py | 77 | 95| 180 | 14232 | 99209 | 
| 39 | 14 sklearn/utils/mocking.py | 1 | 40| 246 | 14478 | 99837 | 
| 40 | 15 sklearn/preprocessing/label.py | 9 | 36| 151 | 14629 | 106357 | 
| 41 | 15 sklearn/decomposition/nmf.py | 1 | 53| 320 | 14949 | 106357 | 
| 42 | 15 sklearn/preprocessing/data.py | 1568 | 1609| 386 | 15335 | 106357 | 
| 43 | 16 sklearn/random_projection.py | 135 | 153| 157 | 15492 | 111330 | 
| 44 | 17 sklearn/decomposition/fastica_.py | 266 | 347| 788 | 16280 | 116030 | 
| 45 | **17 sklearn/utils/validation.py** | 698 | 750| 469 | 16749 | 116030 | 
| 46 | **17 sklearn/utils/validation.py** | 208 | 231| 163 | 16912 | 116030 | 
| 47 | **17 sklearn/utils/validation.py** | 1 | 31| 141 | 17053 | 116030 | 
| 48 | 18 sklearn/utils/multiclass.py | 108 | 155| 403 | 17456 | 119896 | 
| 49 | 19 sklearn/utils/extmath.py | 1 | 38| 211 | 17667 | 126132 | 
| 50 | 20 sklearn/mixture/bayesian_mixture.py | 412 | 451| 419 | 18086 | 133391 | 
| 51 | 21 sklearn/feature_selection/rfe.py | 9 | 21| 119 | 18205 | 137380 | 
| 52 | 22 sklearn/mixture/base.py | 41 | 64| 208 | 18413 | 140945 | 
| 53 | 23 sklearn/feature_selection/variance_threshold.py | 4 | 46| 331 | 18744 | 141540 | 
| 54 | 23 sklearn/decomposition/fastica_.py | 124 | 146| 228 | 18972 | 141540 | 
| 55 | 24 sklearn/linear_model/least_angle.py | 839 | 863| 183 | 19155 | 155347 | 
| 56 | 25 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 19666 | 155880 | 
| 57 | 25 sklearn/utils/estimator_checks.py | 1253 | 1360| 1123 | 20789 | 155880 | 
| 58 | 26 doc/conf.py | 1 | 113| 843 | 21632 | 158291 | 
| 59 | 27 sklearn/covariance/robust_covariance.py | 409 | 505| 1054 | 22686 | 165150 | 
| 60 | 27 sklearn/utils/estimator_checks.py | 727 | 758| 297 | 22983 | 165150 | 
| 61 | 28 sklearn/model_selection/_split.py | 1671 | 1712| 373 | 23356 | 183151 | 
| 62 | 28 sklearn/mixture/gaussian_mixture.py | 98 | 137| 318 | 23674 | 183151 | 
| 63 | 29 benchmarks/bench_plot_nmf.py | 1 | 47| 298 | 23972 | 187064 | 
| **-> 64 <-** | **29 sklearn/utils/validation.py** | 512 | 603| 883 | 24855 | 187064 | 
| 65 | 30 sklearn/utils/__init__.py | 260 | 274| 152 | 25007 | 190899 | 
| 66 | 31 sklearn/utils/_scipy_sparse_lsqr_backport.py | 349 | 491| 1496 | 26503 | 196058 | 
| 67 | 31 sklearn/mixture/bayesian_mixture.py | 330 | 354| 236 | 26739 | 196058 | 
| 68 | 31 sklearn/model_selection/_split.py | 2008 | 2059| 393 | 27132 | 196058 | 
| 69 | 31 sklearn/utils/estimator_checks.py | 527 | 568| 334 | 27466 | 196058 | 
| 70 | 31 sklearn/utils/estimator_checks.py | 761 | 776| 139 | 27605 | 196058 | 


### Hint

```
Unsurprisingly, @raghavrv's PR was forgotten in recent discussion of this. I think we want `force_all_finite='allow-nan'` or =`'-nan'` or similar  
Note: solving #10438 depends on this decision.
Oops! Had not noticed that @glemaitre had started this issue, so was tinkering around with the "allow_nan" and "allow_inf" arguments for check_array() based on a discussion with @jnothman a few months ago. In any case, [here](https://github.com/scikit-learn/scikit-learn/compare/master...ashimb9:checkarray?expand=1) is what I have so far, which might or might not be useful when the discussion here concludes.
Oh sorry about that. I just have ping you on the issue as well.
I will submit what I did yesterday and you can review with what you have in head.
```

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


## Code snippets

### 1 - sklearn/utils/validation.py:

Start line: 34, End line: 55

```python
def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    """
    _assert_all_finite(X.data if sp.issparse(X) else X)
```
### 2 - sklearn/utils/validation.py:

Start line: 395, End line: 435

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""
    # ... other code
```
### 3 - sklearn/utils/estimator_checks.py:

Start line: 1008, End line: 1076

```python
@ignore_warnings(category=DeprecationWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    rnd = np.random.RandomState(0)
    X_train_finite = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                                  estimator_orig)
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    error_string_fit = "Estimator doesn't check for NaN and inf in fit."
    error_string_predict = ("Estimator doesn't check for NaN and inf in"
                            " predict.")
    error_string_transform = ("Estimator doesn't check for NaN and inf in"
                              " transform.")
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            try:
                estimator.fit(X_train, y)
            except ValueError as e:
                if 'inf' not in repr(e) and 'NaN' not in repr(e):
                    print(error_string_fit, estimator, e)
                    traceback.print_exc(file=sys.stdout)
                    raise e
            except Exception as exc:
                print(error_string_fit, estimator, exc)
                traceback.print_exc(file=sys.stdout)
                raise exc
            else:
                raise AssertionError(error_string_fit, estimator)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                try:
                    estimator.predict(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_predict, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_predict, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_predict, estimator)

            # transform
            if hasattr(estimator, "transform"):
                try:
                    estimator.transform(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_transform, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_transform, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_transform, estimator)
```
### 4 - sklearn/utils/validation.py:

Start line: 437, End line: 509

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    # ... other code

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy,
                                      force_all_finite)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.array(array, dtype=dtype, order=order, copy=copy)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happend, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # To ensure that array flags are maintained
            array = np.array(array, dtype=dtype, order=order, copy=copy)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        warnings.warn(msg, DataConversionWarning)
    return array
```
### 5 - sklearn/utils/estimator_checks.py:

Start line: 141, End line: 161

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_no_nan(name, estimator_orig):
    # Checks that the Estimator targets are not NaN.
    estimator = clone(estimator_orig)
    rng = np.random.RandomState(888)
    X = rng.randn(10, 5)
    y = np.ones(10) * np.inf
    y = multioutput_estimator_convert_y_2d(estimator, y)

    errmsg = "Input contains NaN, infinity or a value too large for " \
             "dtype('float64')."
    try:
        estimator.fit(X, y)
    except ValueError as e:
        if str(e) != errmsg:
            raise ValueError("Estimator {0} raised error as expected, but "
                             "does not match expected error message"
                             .format(name))
    else:
        raise ValueError("Estimator {0} should have raised error on fitting "
                         "array y with NaN value.".format(name))
```
### 6 - sklearn/utils/validation.py:

Start line: 318, End line: 394

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    """
    # ... other code
```
### 7 - sklearn/utils/testing.py:

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
### 8 - sklearn/utils/validation.py:

Start line: 604, End line: 618

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
                    ensure_2d, allow_nd, ensure_min_samples,
                    ensure_min_features, warn_on_dtype, estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
                        dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)

    check_consistent_length(X, y)

    return X, y
```
### 9 - sklearn/utils/fixes.py:

Start line: 1, End line: 71

```python
"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"""

import warnings
import os
import errno

import numpy as np
import scipy.sparse as sp
import scipy

try:
    from inspect import signature
except ImportError:
    from ..externals.funcsigs import signature


def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)


euler_gamma = getattr(np, 'euler_gamma',
                      0.577215664901532860606512090082402431)

np_version = _parse_version(np.__version__)
sp_version = _parse_version(scipy.__version__)


# Remove when minimum required NumPy >= 1.10
try:
    if (not np.allclose(np.divide(.4, 1, casting="unsafe"),
                        np.divide(.4, 1, casting="unsafe", dtype=np.float64))
            or not np.allclose(np.divide(.4, 1), .4)):
        raise TypeError('Divide not working with dtype: '
                        'https://github.com/numpy/numpy/issues/3484')
    divide = np.divide

except TypeError:
    # Compat for old versions of np.divide that do not provide support for
    # the dtype args
    def divide(x1, x2, out=None, dtype=None):
        out_orig = out
        if out is None:
            out = np.asarray(x1, dtype=dtype)
            if out is x1:
                out = x1.copy()
        else:
            if out is not x1:
                out[:] = x1
        if dtype is not None and out.dtype != dtype:
            out = out.astype(dtype)
        out /= x2
        if out_orig is None and np.isscalar(x1):
            out = np.asscalar(out)
        return out
```
### 10 - sklearn/utils/validation.py:

Start line: 58, End line: 95

```python
def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats.

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, np.matrix) or (not isinstance(X, np.ndarray)
                                    and not sp.issparse(X)):
        return check_array(X, ['csr', 'csc', 'coo'], dtype=np.float64,
                           copy=copy, force_all_finite=force_all_finite,
                           ensure_2d=False)
    elif sp.issparse(X) and X.dtype in [np.float32, np.float64]:
        return X.copy() if copy else X
    elif X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        if X.dtype.kind in 'uib' and X.dtype.itemsize <= 4:
            return_dtype = np.float32
        else:
            return_dtype = np.float64
        return X.astype(return_dtype)
```
### 13 - sklearn/utils/validation.py:

Start line: 190, End line: 205

```python
def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])
```
### 21 - sklearn/utils/validation.py:

Start line: 234, End line: 315

```python
def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean
        Whether to raise an error on np.inf and np.nan in X.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]

    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data)
    return spmatrix


def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))
```
### 45 - sklearn/utils/validation.py:

Start line: 698, End line: 750

```python
def check_symmetric(array, tol=1E-10, raise_warning=True,
                    raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : nd-array or sparse matrix
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.
    tol : float
        Absolute tolerance for equivalence of arrays. Default = 1E-10.
    raise_warning : boolean (default=True)
        If True then raise a warning if conversion is required.
    raise_exception : boolean (default=False)
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : ndarray or sparse matrix
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError("array must be 2-dimensional and square. "
                         "shape = {0}".format(array.shape))

    if sp.issparse(array):
        diff = array - array.T
        # only csr, csc, and coo have `data` attribute
        if diff.format not in ['csr', 'csc', 'coo']:
            diff = diff.tocsr()
        symmetric = np.all(abs(diff.data) < tol)
    else:
        symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn("Array is not symmetric, and will be converted "
                          "to symmetric by average with its transpose.")
        if sp.issparse(array):
            conversion = 'to' + array.format
            array = getattr(0.5 * (array + array.T), conversion)()
        else:
            array = 0.5 * (array + array.T)

    return array
```
### 46 - sklearn/utils/validation.py:

Start line: 208, End line: 231

```python
def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if sp.issparse(X):
            result.append(X.tocsr())
        elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    check_consistent_length(*result)
    return result
```
### 47 - sklearn/utils/validation.py:

Start line: 1, End line: 31

```python
"""Utilities for input validation"""

import warnings
import numbers

import numpy as np
import scipy.sparse as sp
from numpy.core.numeric import ComplexWarning

from ..externals import six
from ..utils.fixes import signature
from .. import get_config as _get_config
from ..exceptions import NonBLASDotWarning
from ..exceptions import NotFittedError
from ..exceptions import DataConversionWarning
from ..externals.joblib import Memory


FLOAT_DTYPES = (np.float64, np.float32, np.float16)

# Silenced by default to reduce verbosity. Turn on at runtime for
# performance profiling.
warnings.simplefilter('ignore', NonBLASDotWarning)
```
### 64 - sklearn/utils/validation.py:

Start line: 512, End line: 603

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2d and sparse y.  If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    # ... other code
```

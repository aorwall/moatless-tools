# scikit-learn__scikit-learn-13302

| **scikit-learn/scikit-learn** | `4de404d46d24805ff48ad255ec3169a5155986f0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 3763 |
| **Avg pos** | 145.0 |
| **Min pos** | 4 |
| **Max pos** | 62 |
| **Top file pos** | 2 |
| **Missing snippets** | 13 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/ridge.py | 229 | 229 | 13 | 2 | 7902
| sklearn/linear_model/ridge.py | 335 | 335 | 13 | 2 | 7902
| sklearn/linear_model/ridge.py | 363 | 369 | - | 2 | -
| sklearn/linear_model/ridge.py | 391 | 401 | 4 | 2 | 3763
| sklearn/linear_model/ridge.py | 420 | 421 | 4 | 2 | 3763
| sklearn/linear_model/ridge.py | 441 | 441 | 4 | 2 | 3763
| sklearn/linear_model/ridge.py | 464 | 464 | 7 | 2 | 5226
| sklearn/linear_model/ridge.py | 476 | 480 | 7 | 2 | 5226
| sklearn/linear_model/ridge.py | 533 | 539 | 5 | 2 | 4383
| sklearn/linear_model/ridge.py | 558 | 558 | 5 | 2 | 4383
| sklearn/linear_model/ridge.py | 573 | 574 | 5 | 2 | 4383
| sklearn/linear_model/ridge.py | 896 | 897 | 62 | 2 | 32282
| sklearn/linear_model/ridge.py | 1080 | 1083 | 16 | 2 | 9651


## Problem Statement

```
[WIP] EHN: Ridge with solver SAG/SAGA does not cast to float64
closes #11642 

build upon #11155 

TODO:

- [ ] Merge #11155 to reduce the diff.
- [ ] Ensure that the casting rule is clear between base classes, classes and functions. I suspect that we have some copy which are not useful.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/linear_model/sag.py | 91 | 240| 1513 | 1513 | 3171 | 
| 2 | 1 sklearn/linear_model/sag.py | 241 | 317| 785 | 2298 | 3171 | 
| 3 | 1 sklearn/linear_model/sag.py | 1 | 352| 755 | 3053 | 3171 | 
| **-> 4 <-** | **2 sklearn/linear_model/ridge.py** | 366 | 442| 710 | 3763 | 16634 | 
| **-> 5 <-** | **2 sklearn/linear_model/ridge.py** | 517 | 577| 620 | 4383 | 16634 | 
| 6 | **2 sklearn/linear_model/ridge.py** | 1 | 33| 181 | 4564 | 16634 | 
| **-> 7 <-** | **2 sklearn/linear_model/ridge.py** | 443 | 514| 662 | 5226 | 16634 | 
| 8 | **2 sklearn/linear_model/ridge.py** | 998 | 1016| 192 | 5418 | 16634 | 
| 9 | **2 sklearn/linear_model/ridge.py** | 36 | 112| 635 | 6053 | 16634 | 
| 10 | **2 sklearn/linear_model/ridge.py** | 866 | 873| 118 | 6171 | 16634 | 
| 11 | 2 sklearn/linear_model/sag.py | 318 | 353| 307 | 6478 | 16634 | 
| 12 | **2 sklearn/linear_model/ridge.py** | 1038 | 1052| 209 | 6687 | 16634 | 
| **-> 13 <-** | **2 sklearn/linear_model/ridge.py** | 229 | 363| 1215 | 7902 | 16634 | 
| 14 | **2 sklearn/linear_model/ridge.py** | 1018 | 1036| 255 | 8157 | 16634 | 
| 15 | 3 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 772 | 8929 | 17555 | 
| **-> 16 <-** | **3 sklearn/linear_model/ridge.py** | 1062 | 1140| 722 | 9651 | 17555 | 
| 17 | **3 sklearn/linear_model/ridge.py** | 1142 | 1170| 276 | 9927 | 17555 | 
| 18 | **3 sklearn/linear_model/ridge.py** | 1054 | 1060| 123 | 10050 | 17555 | 
| 19 | **3 sklearn/linear_model/ridge.py** | 714 | 741| 231 | 10281 | 17555 | 
| 20 | 4 benchmarks/bench_plot_nmf.py | 105 | 148| 401 | 10682 | 21451 | 
| 21 | 4 benchmarks/bench_plot_nmf.py | 151 | 192| 476 | 11158 | 21451 | 
| 22 | **4 sklearn/linear_model/ridge.py** | 924 | 973| 425 | 11583 | 21451 | 
| 23 | **4 sklearn/linear_model/ridge.py** | 744 | 864| 1138 | 12721 | 21451 | 
| 24 | 5 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 14035 | 22792 | 
| 25 | 6 sklearn/linear_model/bayes.py | 1 | 142| 1162 | 15197 | 28444 | 
| 26 | **6 sklearn/linear_model/ridge.py** | 987 | 996| 133 | 15330 | 28444 | 
| 27 | 7 benchmarks/bench_saga.py | 6 | 104| 837 | 16167 | 30910 | 
| 28 | **7 sklearn/linear_model/ridge.py** | 975 | 985| 133 | 16300 | 30910 | 
| 29 | **7 sklearn/linear_model/ridge.py** | 1173 | 1230| 453 | 16753 | 30910 | 
| 30 | 8 examples/linear_model/plot_huber_vs_ridge.py | 1 | 66| 592 | 17345 | 31524 | 
| 31 | 8 benchmarks/bench_saga.py | 287 | 306| 182 | 17527 | 31524 | 
| 32 | **8 sklearn/linear_model/ridge.py** | 133 | 154| 238 | 17765 | 31524 | 
| 33 | **8 sklearn/linear_model/ridge.py** | 580 | 713| 1307 | 19072 | 31524 | 
| 34 | 9 sklearn/utils/estimator_checks.py | 2079 | 2099| 227 | 19299 | 53500 | 
| 35 | 9 sklearn/linear_model/bayes.py | 144 | 158| 182 | 19481 | 53500 | 
| 36 | 10 sklearn/linear_model/ransac.py | 1 | 19| 111 | 19592 | 57565 | 
| 37 | 10 sklearn/linear_model/bayes.py | 160 | 249| 799 | 20391 | 57565 | 
| 38 | 11 sklearn/cluster/hierarchical.py | 10 | 26| 126 | 20517 | 66573 | 
| 39 | **11 sklearn/linear_model/ridge.py** | 157 | 215| 517 | 21034 | 66573 | 
| 40 | 11 benchmarks/bench_plot_nmf.py | 49 | 104| 590 | 21624 | 66573 | 
| 41 | 11 sklearn/utils/estimator_checks.py | 1 | 62| 444 | 22068 | 66573 | 
| 42 | 11 sklearn/linear_model/bayes.py | 309 | 330| 234 | 22302 | 66573 | 
| 43 | **11 sklearn/linear_model/ridge.py** | 1451 | 1499| 414 | 22716 | 66573 | 
| 44 | 11 sklearn/utils/estimator_checks.py | 1907 | 1951| 472 | 23188 | 66573 | 
| 45 | 11 sklearn/linear_model/bayes.py | 251 | 275| 247 | 23435 | 66573 | 
| 46 | **11 sklearn/linear_model/ridge.py** | 218 | 226| 131 | 23566 | 66573 | 
| 47 | **11 sklearn/linear_model/ridge.py** | 115 | 130| 169 | 23735 | 66573 | 
| 48 | 12 examples/linear_model/plot_ridge_path.py | 1 | 67| 434 | 24169 | 67036 | 
| 49 | 12 benchmarks/bench_saga.py | 192 | 284| 761 | 24930 | 67036 | 
| 50 | 13 examples/linear_model/plot_ridge_coeffs.py | 1 | 90| 590 | 25520 | 67648 | 
| 51 | 13 sklearn/utils/estimator_checks.py | 1416 | 1431| 203 | 25723 | 67648 | 
| 52 | 14 sklearn/utils/optimize.py | 55 | 111| 425 | 26148 | 69139 | 
| 53 | 15 examples/linear_model/plot_ols_ridge_variance.py | 1 | 68| 484 | 26632 | 69631 | 
| 54 | 16 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 361 | 26993 | 83559 | 
| 55 | 17 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 27366 | 85504 | 
| 56 | 17 sklearn/utils/estimator_checks.py | 1433 | 1532| 940 | 28306 | 85504 | 
| 57 | 18 sklearn/decomposition/nmf.py | 500 | 525| 247 | 28553 | 97411 | 
| 58 | 19 sklearn/neighbors/base.py | 1 | 58| 434 | 28987 | 105143 | 
| 59 | 19 sklearn/decomposition/nmf.py | 623 | 702| 716 | 29703 | 105143 | 
| 60 | 20 sklearn/linear_model/least_angle.py | 475 | 632| 1591 | 31294 | 121641 | 
| 61 | 20 benchmarks/bench_saga.py | 107 | 189| 637 | 31931 | 121641 | 
| **-> 62 <-** | **20 sklearn/linear_model/ridge.py** | 875 | 921| 351 | 32282 | 121641 | 
| 63 | 21 sklearn/linear_model/base.py | 177 | 191| 155 | 32437 | 126402 | 
| 64 | 21 sklearn/utils/estimator_checks.py | 2142 | 2218| 630 | 33067 | 126402 | 
| 65 | 21 benchmarks/bench_plot_nmf.py | 230 | 277| 530 | 33597 | 126402 | 
| 66 | 21 sklearn/utils/estimator_checks.py | 630 | 663| 312 | 33909 | 126402 | 
| 67 | 21 sklearn/decomposition/nmf.py | 792 | 1071| 603 | 34512 | 126402 | 
| 68 | 21 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 35044 | 126402 | 
| 69 | 21 sklearn/utils/estimator_checks.py | 819 | 855| 352 | 35396 | 126402 | 
| 70 | 21 sklearn/utils/estimator_checks.py | 1838 | 1880| 469 | 35865 | 126402 | 
| 71 | 22 sklearn/linear_model/logistic.py | 1204 | 1442| 2732 | 38597 | 148445 | 
| 72 | 22 examples/linear_model/plot_bayesian_ridge.py | 100 | 115| 149 | 38746 | 148445 | 
| 73 | 22 benchmarks/bench_plot_nmf.py | 195 | 228| 317 | 39063 | 148445 | 
| 74 | 23 sklearn/covariance/graph_lasso_.py | 202 | 280| 891 | 39954 | 157499 | 
| 75 | 23 sklearn/utils/estimator_checks.py | 592 | 627| 472 | 40426 | 157499 | 
| 76 | 23 sklearn/utils/estimator_checks.py | 1161 | 1229| 601 | 41027 | 157499 | 
| 77 | 23 sklearn/utils/estimator_checks.py | 1115 | 1135| 236 | 41263 | 157499 | 
| 78 | 24 sklearn/kernel_ridge.py | 1 | 107| 971 | 42234 | 159180 | 
| 79 | 25 benchmarks/bench_covertype.py | 99 | 109| 151 | 42385 | 161071 | 
| 80 | 25 sklearn/kernel_ridge.py | 108 | 129| 186 | 42571 | 161071 | 
| 81 | 26 sklearn/svm/base.py | 170 | 223| 596 | 43167 | 169287 | 
| 82 | 26 sklearn/decomposition/nmf.py | 528 | 620| 844 | 44011 | 169287 | 
| 83 | 26 sklearn/utils/estimator_checks.py | 79 | 112| 261 | 44272 | 169287 | 
| 84 | 27 sklearn/linear_model/__init__.py | 1 | 82| 674 | 44946 | 169961 | 
| 85 | 27 sklearn/utils/estimator_checks.py | 140 | 160| 202 | 45148 | 169961 | 
| 86 | 27 sklearn/utils/estimator_checks.py | 1974 | 2008| 363 | 45511 | 169961 | 
| 87 | 28 sklearn/utils/extmath.py | 660 | 689| 258 | 45769 | 176277 | 
| 88 | 29 sklearn/utils/validation.py | 481 | 550| 810 | 46579 | 184494 | 
| 89 | 29 sklearn/utils/estimator_checks.py | 499 | 547| 491 | 47070 | 184494 | 
| 90 | 29 sklearn/decomposition/nmf.py | 391 | 419| 286 | 47356 | 184494 | 
| 91 | 30 sklearn/cluster/optics_.py | 695 | 705| 131 | 47487 | 192818 | 


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


## Code snippets

### 1 - sklearn/linear_model/sag.py:

Start line: 91, End line: 240

```python
def sag_solver(X, y, sample_weight=None, loss='log', alpha=1., beta=0.,
               max_iter=1000, tol=0.001, verbose=0, random_state=None,
               check_input=True, max_squared_sum=None,
               warm_start_mem=None,
               is_saga=False):
    """SAG solver for Ridge and LogisticRegression

    SAG stands for Stochastic Average Gradient: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a constant learning rate.

    IMPORTANT NOTE: 'sag' solver converges faster on columns that are on the
    same scale. You can normalize the data by using
    sklearn.preprocessing.StandardScaler on your data before passing it to the
    fit method.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values for the features. It will
    fit the data according to squared loss or log loss.

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using the squared euclidean norm L2.

    .. versionadded:: 0.17

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data

    y : numpy array, shape (n_samples,)
        Target values. With loss='multinomial', y must be label encoded
        (see preprocessing.LabelEncoder).

    sample_weight : array-like, shape (n_samples,), optional
        Weights applied to individual samples (1. for unweighted).

    loss : 'log' | 'squared' | 'multinomial'
        Loss function that will be optimized:
        -'log' is the binary logistic loss, as used in LogisticRegression.
        -'squared' is the squared loss, as used in Ridge.
        -'multinomial' is the multinomial logistic loss, as used in
         LogisticRegression.

        .. versionadded:: 0.18
           *loss='multinomial'*

    alpha : float, optional
        L2 regularization term in the objective function
        ``(0.5 * alpha * || W ||_F^2)``. Defaults to 1.

    beta : float, optional
        L1 regularization term in the objective function
        ``(beta * || W ||_1)``. Only applied if ``is_saga`` is set to True.
        Defaults to 0.

    max_iter : int, optional
        The max number of passes over the training data if the stopping
        criteria is not reached. Defaults to 1000.

    tol : double, optional
        The stopping criteria for the weights. The iterations will stop when
        max(change in weights) / max(weights) < tol. Defaults to .001

    verbose : integer, optional
        The verbosity level.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    check_input : bool, default True
        If False, the input arrays X and y will not be checked.

    max_squared_sum : float, default None
        Maximum squared sum of X over samples. If None, it will be computed,
        going through all the samples. The value should be precomputed
        to speed up cross validation.

    warm_start_mem : dict, optional
        The initialization parameters used for warm starting. Warm starting is
        currently used in LogisticRegression but not in Ridge.
        It contains:
            - 'coef': the weight vector, with the intercept in last line
                if the intercept is fitted.
            - 'gradient_memory': the scalar gradient for all seen samples.
            - 'sum_gradient': the sum of gradient over all seen samples,
                for each feature.
            - 'intercept_sum_gradient': the sum of gradient over all seen
                samples, for the intercept.
            - 'seen': array of boolean describing the seen samples.
            - 'num_seen': the number of seen samples.

    is_saga : boolean, optional
        Whether to use the SAGA algorithm or the SAG algorithm. SAGA behaves
        better in the first epochs, and allow for l1 regularisation.

    Returns
    -------
    coef_ : array, shape (n_features)
        Weight vector.

    n_iter_ : int
        The number of full pass on all samples.

    warm_start_mem : dict
        Contains a 'coef' key with the fitted result, and possibly the
        fitted intercept at the end of the array. Contains also other keys
        used for warm starting.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(n_samples, n_features)
    >>> y = rng.randn(n_samples)
    >>> clf = linear_model.Ridge(solver='sag')
    >>> clf.fit(X, y)
    ... #doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='sag', tol=0.001)

    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.LogisticRegression(
    ...     solver='sag', multi_class='multinomial')
    >>> clf.fit(X, y)
    ... #doctest: +NORMALIZE_WHITESPACE
    LogisticRegression(C=1.0, class_weight=None, dual=False,
        fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100,
        multi_class='multinomial', n_jobs=None, penalty='l2',
        random_state=None, solver='sag', tol=0.0001, verbose=0,
        warm_start=False)

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202

    See also
    --------
    Ridge, SGDRegressor, ElasticNet, Lasso, SVR, and
    LogisticRegression, SGDClassifier, LinearSVC, Perceptron
    """
    # ... other code
```
### 2 - sklearn/linear_model/sag.py:

Start line: 241, End line: 317

```python
def sag_solver(X, y, sample_weight=None, loss='log', alpha=1., beta=0.,
               max_iter=1000, tol=0.001, verbose=0, random_state=None,
               check_input=True, max_squared_sum=None,
               warm_start_mem=None,
               is_saga=False):
    if warm_start_mem is None:
        warm_start_mem = {}
    # Ridge default max_iter is None
    if max_iter is None:
        max_iter = 1000

    if check_input:
        _dtype = [np.float64, np.float32]
        X = check_array(X, dtype=_dtype, accept_sparse='csr', order='C')
        y = check_array(y, dtype=_dtype, ensure_2d=False, order='C')

    n_samples, n_features = X.shape[0], X.shape[1]
    # As in SGD, the alpha is scaled by n_samples.
    alpha_scaled = float(alpha) / n_samples
    beta_scaled = float(beta) / n_samples

    # if loss == 'multinomial', y should be label encoded.
    n_classes = int(y.max()) + 1 if loss == 'multinomial' else 1

    # initialization
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=X.dtype, order='C')

    if 'coef' in warm_start_mem.keys():
        coef_init = warm_start_mem['coef']
    else:
        # assume fit_intercept is False
        coef_init = np.zeros((n_features, n_classes), dtype=X.dtype,
                             order='C')

    # coef_init contains possibly the intercept_init at the end.
    # Note that Ridge centers the data before fitting, so fit_intercept=False.
    fit_intercept = coef_init.shape[0] == (n_features + 1)
    if fit_intercept:
        intercept_init = coef_init[-1, :]
        coef_init = coef_init[:-1, :]
    else:
        intercept_init = np.zeros(n_classes, dtype=X.dtype)

    if 'intercept_sum_gradient' in warm_start_mem.keys():
        intercept_sum_gradient = warm_start_mem['intercept_sum_gradient']
    else:
        intercept_sum_gradient = np.zeros(n_classes, dtype=X.dtype)

    if 'gradient_memory' in warm_start_mem.keys():
        gradient_memory_init = warm_start_mem['gradient_memory']
    else:
        gradient_memory_init = np.zeros((n_samples, n_classes),
                                        dtype=X.dtype, order='C')
    if 'sum_gradient' in warm_start_mem.keys():
        sum_gradient_init = warm_start_mem['sum_gradient']
    else:
        sum_gradient_init = np.zeros((n_features, n_classes),
                                     dtype=X.dtype, order='C')

    if 'seen' in warm_start_mem.keys():
        seen_init = warm_start_mem['seen']
    else:
        seen_init = np.zeros(n_samples, dtype=np.int32, order='C')

    if 'num_seen' in warm_start_mem.keys():
        num_seen_init = warm_start_mem['num_seen']
    else:
        num_seen_init = 0

    dataset, intercept_decay = make_dataset(X, y, sample_weight, random_state)

    if max_squared_sum is None:
        max_squared_sum = row_norms(X, squared=True).max()
    step_size = get_auto_step_size(max_squared_sum, alpha_scaled, loss,
                                   fit_intercept, n_samples=n_samples,
                                   is_saga=is_saga)
    if step_size * alpha_scaled == 1:
        raise ZeroDivisionError("Current sag implementation does not handle "
                                "the case step_size * alpha_scaled == 1")

    sag = sag64 if X.dtype == np.float64 else sag32
    # ... other code
```
### 3 - sklearn/linear_model/sag.py:

Start line: 1, End line: 352

```python
"""Solvers for Ridge and LogisticRegression using SAG algorithm"""

import warnings

import numpy as np

from .base import make_dataset
from .sag_fast import sag32, sag64
from ..exceptions import ConvergenceWarning
from ..utils import check_array
from ..utils.extmath import row_norms


def get_auto_step_size(max_squared_sum, alpha_scaled, loss, fit_intercept,
                       n_samples=None,
                       is_saga=False):
    """Compute automatic step size for SAG solver

    The step size is set to 1 / (alpha_scaled + L + fit_intercept) where L is
    the max sum of squares for over all samples.

    Parameters
    ----------
    max_squared_sum : float
        Maximum squared sum of X over samples.

    alpha_scaled : float
        Constant that multiplies the regularization term, scaled by
        1. / n_samples, the number of samples.

    loss : string, in {"log", "squared"}
        The loss function used in SAG solver.

    fit_intercept : bool
        Specifies if a constant (a.k.a. bias or intercept) will be
        added to the decision function.

    n_samples : int, optional
        Number of rows in X. Useful if is_saga=True.

    is_saga : boolean, optional
        Whether to return step size for the SAGA algorithm or the SAG
        algorithm.

    Returns
    -------
    step_size : float
        Step size used in SAG solver.

    References
    ----------
    Schmidt, M., Roux, N. L., & Bach, F. (2013).
    Minimizing finite sums with the stochastic average gradient
    https://hal.inria.fr/hal-00860051/document

    Defazio, A., Bach F. & Lacoste-Julien S. (2014).
    SAGA: A Fast Incremental Gradient Method With Support
    for Non-Strongly Convex Composite Objectives
    https://arxiv.org/abs/1407.0202
    """
    if loss in ('log', 'multinomial'):
        L = (0.25 * (max_squared_sum + int(fit_intercept)) + alpha_scaled)
    elif loss == 'squared':
        # inverse Lipschitz constant for squared loss
        L = max_squared_sum + int(fit_intercept) + alpha_scaled
    else:
        raise ValueError("Unknown loss function for SAG solver, got %s "
                         "instead of 'log' or 'squared'" % loss)
    if is_saga:
        # SAGA theoretical step size is 1/3L or 1 / (2 * (L + mu n))
        # See Defazio et al. 2014
        mun = min(2 * n_samples * alpha_scaled, L)
        step = 1. / (2 * L + mun)
    else:
        # SAG theoretical step size is 1/16L but it is recommended to use 1 / L
        # see http://www.birs.ca//workshops//2014/14w5003/files/schmidt.pdf,
        # slide 65
        step = 1. / L
    return step


def sag_solver(X, y, sample_weight=None, loss='log', alpha=1., beta=0.,
               max_iter=1000, tol=0.001, verbose=0, random_state=None,
               check_input=True, max_squared_sum=None,
               warm_start_mem=None,
               is_saga=False):
    # ... other code
```
### 4 - sklearn/linear_model/ridge.py:

Start line: 366, End line: 442

```python
def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                      max_iter=None, tol=1e-3, verbose=0, random_state=None,
                      return_n_iter=False, return_intercept=False,
                      X_scale=None, X_offset=None):

    has_sw = sample_weight is not None

    if solver == 'auto':
        if return_intercept:
            # only sag supports fitting intercept directly
            solver = "sag"
        elif not sparse.issparse(X):
            solver = "cholesky"
        else:
            solver = "sparse_cg"

    if solver not in ('sparse_cg', 'cholesky', 'svd', 'lsqr', 'sag', 'saga'):
        raise ValueError("Known solvers are 'sparse_cg', 'cholesky', 'svd'"
                         " 'lsqr', 'sag' or 'saga'. Got %s." % solver)

    if return_intercept and solver != 'sag':
        raise ValueError("In Ridge, only 'sag' solver can directly fit the "
                         "intercept. Please change solver to 'sag' or set "
                         "return_intercept=False.")

    _dtype = [np.float64, np.float32]

    # SAG needs X and y columns to be C-contiguous and np.float64
    if solver in ['sag', 'saga']:
        X = check_array(X, accept_sparse=['csr'],
                        dtype=np.float64, order='C')
        y = check_array(y, dtype=np.float64, ensure_2d=False, order='F')
    else:
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=_dtype)
        y = check_array(y, dtype=X.dtype, ensure_2d=False)
    check_consistent_length(X, y)

    n_samples, n_features = X.shape

    if y.ndim > 2:
        raise ValueError("Target y has the wrong shape %s" % str(y.shape))

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    n_samples_, n_targets = y.shape

    if n_samples != n_samples_:
        raise ValueError("Number of samples in X and y does not correspond:"
                         " %d != %d" % (n_samples, n_samples_))



    if has_sw:
        if np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if solver not in ['sag', 'saga']:
            # SAG supports sample_weight directly. For other solvers,
            # we implement sample_weight via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

    # There should be either 1 or n_targets penalties
    alpha = np.asarray(alpha, dtype=X.dtype).ravel()
    if alpha.size not in [1, n_targets]:
        raise ValueError("Number of targets and number of penalties "
                         "do not correspond: %d != %d"
                         % (alpha.size, n_targets))

    if alpha.size == 1 and n_targets > 1:
        alpha = np.repeat(alpha, n_targets)


    n_iter = None
    # ... other code
```
### 5 - sklearn/linear_model/ridge.py:

Start line: 517, End line: 577

```python
class _BaseRidge(LinearModel, MultiOutputMixin, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):

        if self.solver in ('sag', 'saga'):
            _dtype = np.float64
        else:
            # all other solvers work at both float precision levels
            _dtype = [np.float64, np.float32]

        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=_dtype,
                         multi_output=True, y_numeric=True)

        if ((sample_weight is not None) and
                np.atleast_1d(sample_weight).ndim > 1):
            raise ValueError("Sample weights must be 1D array or scalar")

        # when X is sparse we only remove offset from y
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight, return_mean=True)

        # temporary fix for fitting the intercept with sparse data using 'sag'
        if (sparse.issparse(X) and self.fit_intercept and
           self.solver != 'sparse_cg'):
            self.coef_, self.n_iter_, self.intercept_ = _ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=True)
            # add the offset which was subtracted by _preprocess_data
            self.intercept_ += y_offset
        else:
            if sparse.issparse(X) and self.solver == 'sparse_cg':
                # required to fit intercept with sparse_cg solver
                params = {'X_offset': X_offset, 'X_scale': X_scale}
            else:
                # for dense matrices or when intercept is set to 0
                params = {}

            self.coef_, self.n_iter_ = _ridge_regression(
                X, y, alpha=self.alpha, sample_weight=sample_weight,
                max_iter=self.max_iter, tol=self.tol, solver=self.solver,
                random_state=self.random_state, return_n_iter=True,
                return_intercept=False, **params)

            self._set_intercept(X_offset, y_offset, X_scale)

        return self
```
### 6 - sklearn/linear_model/ridge.py:

Start line: 1, End line: 33

```python
"""
Ridge regression
"""


from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

from .base import LinearClassifierMixin, LinearModel, _rescale_data
from .sag import sag_solver
from ..base import RegressorMixin, MultiOutputMixin
from ..utils.extmath import safe_sparse_dot
from ..utils.extmath import row_norms
from ..utils import check_X_y
from ..utils import check_array
from ..utils import check_consistent_length
from ..utils import compute_sample_weight
from ..utils import column_or_1d
from ..preprocessing import LabelBinarizer
from ..model_selection import GridSearchCV
from ..metrics.scorer import check_scoring
from ..exceptions import ConvergenceWarning
```
### 7 - sklearn/linear_model/ridge.py:

Start line: 443, End line: 514

```python
def _ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                      max_iter=None, tol=1e-3, verbose=0, random_state=None,
                      return_n_iter=False, return_intercept=False,
                      X_scale=None, X_offset=None):
    # ... other code
    if solver == 'sparse_cg':
        coef = _solve_sparse_cg(X, y, alpha,
                                max_iter=max_iter,
                                tol=tol,
                                verbose=verbose,
                                X_offset=X_offset,
                                X_scale=X_scale)

    elif solver == 'lsqr':
        coef, n_iter = _solve_lsqr(X, y, alpha, max_iter, tol)

    elif solver == 'cholesky':
        if n_features > n_samples:
            K = safe_sparse_dot(X, X.T, dense_output=True)
            try:
                dual_coef = _solve_cholesky_kernel(K, y, alpha)

                coef = safe_sparse_dot(X.T, dual_coef, dense_output=True).T
            except linalg.LinAlgError:
                # use SVD solver if matrix is singular
                solver = 'svd'

        else:
            try:
                coef = _solve_cholesky(X, y, alpha)
            except linalg.LinAlgError:
                # use SVD solver if matrix is singular
                solver = 'svd'

    elif solver in ['sag', 'saga']:
        # precompute max_squared_sum for all targets
        max_squared_sum = row_norms(X, squared=True).max()

        coef = np.empty((y.shape[1], n_features))
        n_iter = np.empty(y.shape[1], dtype=np.int32)
        intercept = np.zeros((y.shape[1], ))
        for i, (alpha_i, target) in enumerate(zip(alpha, y.T)):
            init = {'coef': np.zeros((n_features + int(return_intercept), 1))}
            coef_, n_iter_, _ = sag_solver(
                X, target.ravel(), sample_weight, 'squared', alpha_i, 0,
                max_iter, tol, verbose, random_state, False, max_squared_sum,
                init,
                is_saga=solver == 'saga')
            if return_intercept:
                coef[i] = coef_[:-1]
                intercept[i] = coef_[-1]
            else:
                coef[i] = coef_
            n_iter[i] = n_iter_

        if intercept.shape[0] == 1:
            intercept = intercept[0]
        coef = np.asarray(coef)

    if solver == 'svd':
        if sparse.issparse(X):
            raise TypeError('SVD solver does not support sparse'
                            ' inputs currently')
        coef = _solve_svd(X, y, alpha)

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        coef = coef.ravel()

    if return_n_iter and return_intercept:
        return coef, n_iter, intercept
    elif return_intercept:
        return coef, intercept
    elif return_n_iter:
        return coef, n_iter
    else:
        return coef
```
### 8 - sklearn/linear_model/ridge.py:

Start line: 998, End line: 1016

```python
class _RidgeGCV(LinearModel):

    def _errors_and_values_helper(self, alpha, y, v, Q, QT_y):
        """Helper function to avoid code duplication between self._errors and
        self._values.

        Notes
        -----
        We don't construct matrix G, instead compute action on y & diagonal.
        """
        w = 1. / (v + alpha)
        constant_column = np.var(Q, 0) < 1.e-12
        # detect constant columns
        w[constant_column] = 0  # cancel the regularization for the intercept

        c = np.dot(Q, self._diag_dot(w, QT_y))
        G_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c
```
### 9 - sklearn/linear_model/ridge.py:

Start line: 36, End line: 112

```python
def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=1e-3, verbose=0,
                     X_offset=None, X_scale=None):

    def _get_rescaled_operator(X):

        X_offset_scale = X_offset / X_scale

        def matvec(b):
            return X.dot(b) - b.dot(X_offset_scale)

        def rmatvec(b):
            return X.T.dot(b) - X_offset_scale * np.sum(b)

        X1 = sparse.linalg.LinearOperator(shape=X.shape,
                                          matvec=matvec,
                                          rmatvec=rmatvec)
        return X1

    n_samples, n_features = X.shape

    if X_offset is None or X_scale is None:
        X1 = sp_linalg.aslinearoperator(X)
    else:
        X1 = _get_rescaled_operator(X)

    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)

    if n_features > n_samples:
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.matvec(X1.rmatvec(x)) + curr_alpha * x
            return _mv
    else:
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.rmatvec(X1.matvec(x)) + curr_alpha * x
            return _mv

    for i in range(y.shape[1]):
        y_column = y[:, i]

        mv = create_mv(alpha[i])
        if n_features > n_samples:
            # kernel ridge
            # w = X.T * inv(X X^t + alpha*Id) y
            C = sp_linalg.LinearOperator(
                (n_samples, n_samples), matvec=mv, dtype=X.dtype)
            # FIXME atol
            try:
                coef, info = sp_linalg.cg(C, y_column, tol=tol, atol='legacy')
            except TypeError:
                # old scipy
                coef, info = sp_linalg.cg(C, y_column, tol=tol)
            coefs[i] = X1.rmatvec(coef)
        else:
            # linear ridge
            # w = inv(X^t X + alpha*Id) * X.T y
            y_column = X1.rmatvec(y_column)
            C = sp_linalg.LinearOperator(
                (n_features, n_features), matvec=mv, dtype=X.dtype)
            # FIXME atol
            try:
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter,
                                              tol=tol, atol='legacy')
            except TypeError:
                # old scipy
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter,
                                              tol=tol)

        if info < 0:
            raise ValueError("Failed with error code %d" % info)

        if max_iter is None and info > 0 and verbose:
            warnings.warn("sparse_cg did not converge after %d iterations." %
                          info, ConvergenceWarning)

    return coefs
```
### 10 - sklearn/linear_model/ridge.py:

Start line: 866, End line: 873

```python
class RidgeClassifier(LinearClassifierMixin, _BaseRidge):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, class_weight=None,
                 solver="auto", random_state=None):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
            copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver,
            random_state=random_state)
        self.class_weight = class_weight
```
### 12 - sklearn/linear_model/ridge.py:

Start line: 1038, End line: 1052

```python
class _RidgeGCV(LinearModel):

    def _errors_and_values_svd_helper(self, alpha, y, v, U, UT_y):
        """Helper function to avoid code duplication between self._errors_svd
        and self._values_svd.
        """
        constant_column = np.var(U, 0) < 1.e-12
        # detect columns colinear to ones
        w = ((v + alpha) ** -1) - (alpha ** -1)
        w[constant_column] = - (alpha ** -1)
        # cancel the regularization for the intercept
        c = np.dot(U, self._diag_dot(w, UT_y)) + (alpha ** -1) * y
        G_diag = self._decomp_diag(w, U) + (alpha ** -1)
        if len(y.shape) != 1:
            # handle case where y is 2-d
            G_diag = G_diag[:, np.newaxis]
        return G_diag, c
```
### 13 - sklearn/linear_model/ridge.py:

Start line: 229, End line: 363

```python
def ridge_regression(X, y, alpha, sample_weight=None, solver='auto',
                     max_iter=None, tol=1e-3, verbose=0, random_state=None,
                     return_n_iter=False, return_intercept=False):
    """Solve the ridge equation by the method of normal equations.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    X : {array-like, sparse matrix, LinearOperator},
        shape = [n_samples, n_features]
        Training data

    y : array-like, shape = [n_samples] or [n_samples, n_targets]
        Target values

    alpha : {float, array-like},
        shape = [n_targets] if array-like
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.

    sample_weight : float or numpy array of shape [n_samples]
        Individual weights for each sample. If sample_weight is not None and
        solver='auto', the solver will be set to 'cholesky'.

        .. versionadded:: 0.17

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than
          'cholesky'.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution via a Cholesky decomposition of
          dot(X.T, X)

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.


        All last five solvers support both dense and sparse data. However, only
        'sag' and 'sparse_cg' supports sparse input when`fit_intercept` is
        True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        For the 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' and saga solver, the default value is
        1000.

    tol : float
        Precision of the solution.

    verbose : int
        Verbosity level. Setting verbose > 0 will display additional
        information depending on the solver used.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag'.

    return_n_iter : boolean, default False
        If True, the method also returns `n_iter`, the actual number of
        iteration performed by the solver.

        .. versionadded:: 0.17

    return_intercept : boolean, default False
        If True and if X is sparse, the method also returns the intercept,
        and the solver is automatically changed to 'sag'. This is only a
        temporary fix for fitting the intercept with sparse data. For dense
        data, use sklearn.linear_model._preprocess_data before your regression.

        .. versionadded:: 0.17

    Returns
    -------
    coef : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s).

    n_iter : int, optional
        The actual number of iteration performed by the solver.
        Only returned if `return_n_iter` is True.

    intercept : float or array, shape = [n_targets]
        The intercept of the model. Only returned if `return_intercept`
        is True and if X is a scipy sparse array.

    Notes
    -----
    This function won't compute the intercept.
    """

    return _ridge_regression(X, y, alpha,
                             sample_weight=sample_weight,
                             solver=solver,
                             max_iter=max_iter,
                             tol=tol,
                             verbose=verbose,
                             random_state=random_state,
                             return_n_iter=return_n_iter,
                             return_intercept=return_intercept,
                             X_scale=None,
                             X_offset=None)
```
### 14 - sklearn/linear_model/ridge.py:

Start line: 1018, End line: 1036

```python
class _RidgeGCV(LinearModel):

    def _errors(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return (c / G_diag) ** 2, c

    def _values(self, alpha, y, v, Q, QT_y):
        G_diag, c = self._errors_and_values_helper(alpha, y, v, Q, QT_y)
        return y - (c / G_diag), c

    def _pre_compute_svd(self, X, y, centered_kernel=True):
        if sparse.issparse(X):
            raise TypeError("SVD not supported for sparse matrices")
        if centered_kernel:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        # to emulate fit_intercept=True situation, add a column on ones
        # Note that by centering, the other columns are orthogonal to that one
        U, s, _ = linalg.svd(X, full_matrices=0)
        v = s ** 2
        UT_y = np.dot(U.T, y)
        return v, U, UT_y
```
### 16 - sklearn/linear_model/ridge.py:

Start line: 1062, End line: 1140

```python
class _RidgeGCV(LinearModel):

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'], dtype=np.float64,
                         multi_output=True, y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)
        n_samples, n_features = X.shape

        X, y, X_offset, y_offset, X_scale = LinearModel._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X,
            sample_weight=sample_weight)

        gcv_mode = self.gcv_mode
        with_sw = len(np.shape(sample_weight))

        if gcv_mode is None or gcv_mode == 'auto':
            if sparse.issparse(X) or n_features > n_samples or with_sw:
                gcv_mode = 'eigen'
            else:
                gcv_mode = 'svd'
        elif gcv_mode == "svd" and with_sw:
            # FIXME non-uniform sample weights not yet supported
            warnings.warn("non-uniform sample weights unsupported for svd, "
                          "forcing usage of eigen")
            gcv_mode = 'eigen'

        if gcv_mode == 'eigen':
            _pre_compute = self._pre_compute
            _errors = self._errors
            _values = self._values
        elif gcv_mode == 'svd':
            # assert n_samples >= n_features
            _pre_compute = self._pre_compute_svd
            _errors = self._errors_svd
            _values = self._values_svd
        else:
            raise ValueError('bad gcv_mode "%s"' % gcv_mode)

        if sample_weight is not None:
            X, y = _rescale_data(X, y, sample_weight)

        centered_kernel = not sparse.issparse(X) and self.fit_intercept

        v, Q, QT_y = _pre_compute(X, y, centered_kernel)
        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        cv_values = np.zeros((n_samples * n_y, len(self.alphas)))
        C = []

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        error = scorer is None

        if np.any(self.alphas < 0):
            raise ValueError("alphas cannot be negative. "
                             "Got {} containing some "
                             "negative value instead.".format(self.alphas))

        for i, alpha in enumerate(self.alphas):
            if error:
                out, c = _errors(float(alpha), y, v, Q, QT_y)
            else:
                out, c = _values(float(alpha), y, v, Q, QT_y)
            cv_values[:, i] = out.ravel()
            C.append(c)
        # ... other code
```
### 17 - sklearn/linear_model/ridge.py:

Start line: 1142, End line: 1170

```python
class _RidgeGCV(LinearModel):

    def fit(self, X, y, sample_weight=None):
        # ... other code

        if error:
            best = cv_values.mean(axis=0).argmin()
        else:
            # The scorer want an object that will make the predictions but
            # they are already computed efficiently by _RidgeGCV. This
            # identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.decision_function = lambda y_predict: y_predict
            identity_estimator.predict = lambda y_predict: y_predict

            out = [scorer(identity_estimator, y.ravel(), cv_values[:, i])
                   for i in range(len(self.alphas))]
            best = np.argmax(out)

        self.alpha_ = self.alphas[best]
        self.dual_coef_ = C[best]
        self.coef_ = safe_sparse_dot(self.dual_coef_.T, X)

        self._set_intercept(X_offset, y_offset, X_scale)

        if self.store_cv_values:
            if len(y.shape) == 1:
                cv_values_shape = n_samples, len(self.alphas)
            else:
                cv_values_shape = n_samples, n_y, len(self.alphas)
            self.cv_values_ = cv_values.reshape(cv_values_shape)

        return self
```
### 18 - sklearn/linear_model/ridge.py:

Start line: 1054, End line: 1060

```python
class _RidgeGCV(LinearModel):

    def _errors_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return (c / G_diag) ** 2, c

    def _values_svd(self, alpha, y, v, U, UT_y):
        G_diag, c = self._errors_and_values_svd_helper(alpha, y, v, U, UT_y)
        return y - (c / G_diag), c
```
### 19 - sklearn/linear_model/ridge.py:

Start line: 714, End line: 741

```python
class Ridge(_BaseRidge, RegressorMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=None, tol=1e-3, solver="auto",
                 random_state=None):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            normalize=normalize, copy_X=copy_X,
            max_iter=max_iter, tol=tol, solver=solver,
            random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values

        sample_weight : float or numpy array of shape [n_samples]
            Individual weights for each sample

        Returns
        -------
        self : returns an instance of self.
        """
        return super().fit(X, y, sample_weight=sample_weight)
```
### 22 - sklearn/linear_model/ridge.py:

Start line: 924, End line: 973

```python
class _RidgeGCV(LinearModel):
    """Ridge regression with built-in Generalized Cross-Validation

    It allows efficient Leave-One-Out cross-validation.

    This class is not intended to be used directly. Use RidgeCV instead.

    Notes
    -----

    We want to solve (K + alpha*Id)c = y,
    where K = X X^T is the kernel matrix.

    Let G = (K + alpha*Id)^-1.

    Dual solution: c = Gy
    Primal solution: w = X^T c

    Compute eigendecomposition K = Q V Q^T.
    Then G = Q (V + alpha*Id)^-1 Q^T,
    where (V + alpha*Id) is diagonal.
    It is thus inexpensive to inverse for many alphas.

    Let loov be the vector of prediction values for each example
    when the model was fitted with all examples but this example.

    loov = (KGY - diag(KG)Y) / diag(I-KG)

    Let looe be the vector of prediction errors for each example
    when the model was fitted with all examples but this example.

    looe = y - loov = c / diag(G)

    References
    ----------
    http://cbcl.mit.edu/publications/ps/MIT-CSAIL-TR-2007-025.pdf
    https://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf
    """

    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False,
                 scoring=None, copy_X=True,
                 gcv_mode=None, store_cv_values=False):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.copy_X = copy_X
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values
```
### 23 - sklearn/linear_model/ridge.py:

Start line: 744, End line: 864

```python
class RidgeClassifier(LinearClassifierMixin, _BaseRidge):
    """Classifier using Ridge regression.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alpha : float
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC.

    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations (e.g. data is expected to be
        already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        The default value is determined by scipy.sparse.linalg.

    tol : float
        Precision of the solution.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than
          'cholesky'.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its unbiased and more flexible version named SAGA. Both methods
          use an iterative procedure, and are often faster than other solvers
          when both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

          .. versionadded:: 0.17
             Stochastic Average Gradient descent solver.
          .. versionadded:: 0.19
           SAGA solver.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag'.

    Attributes
    ----------
    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        ``coef_`` is of shape (1, n_features) when the given problem is binary.

    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    n_iter_ : array or None, shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.linear_model import RidgeClassifier
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = RidgeClassifier().fit(X, y)
    >>> clf.score(X, y) # doctest: +ELLIPSIS
    0.9595...

    See also
    --------
    Ridge : Ridge regression
    RidgeClassifierCV :  Ridge classifier with built-in cross validation

    Notes
    -----
    For multi-class classification, n_class classifiers are trained in
    a one-versus-all approach. Concretely, this is implemented by taking
    advantage of the multi-variate response support in Ridge.
    """
```
### 26 - sklearn/linear_model/ridge.py:

Start line: 987, End line: 996

```python
class _RidgeGCV(LinearModel):

    def _decomp_diag(self, v_prime, Q):
        # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
        return (v_prime * Q ** 2).sum(axis=-1)

    def _diag_dot(self, D, B):
        # compute dot(diag(D), B)
        if len(B.shape) > 1:
            # handle case where B is > 1-d
            D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
        return D * B
```
### 28 - sklearn/linear_model/ridge.py:

Start line: 975, End line: 985

```python
class _RidgeGCV(LinearModel):

    def _pre_compute(self, X, y, centered_kernel=True):
        # even if X is very sparse, K is usually very dense
        K = safe_sparse_dot(X, X.T, dense_output=True)
        # the following emulates an additional constant regressor
        # corresponding to fit_intercept=True
        # but this is done only when the features have been centered
        if centered_kernel:
            K += np.ones_like(K)
        v, Q = linalg.eigh(K)
        QT_y = np.dot(Q.T, y)
        return v, Q, QT_y
```
### 29 - sklearn/linear_model/ridge.py:

Start line: 1173, End line: 1230

```python
class _BaseRidgeCV(LinearModel, MultiOutputMixin):
    def __init__(self, alphas=(0.1, 1.0, 10.0),
                 fit_intercept=True, normalize=False, scoring=None,
                 cv=None, gcv_mode=None,
                 store_cv_values=False):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_values = store_cv_values

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        sample_weight : float or array-like of shape [n_samples]
            Sample weight

        Returns
        -------
        self : object
        """
        if self.cv is None:
            estimator = _RidgeGCV(self.alphas,
                                  fit_intercept=self.fit_intercept,
                                  normalize=self.normalize,
                                  scoring=self.scoring,
                                  gcv_mode=self.gcv_mode,
                                  store_cv_values=self.store_cv_values)
            estimator.fit(X, y, sample_weight=sample_weight)
            self.alpha_ = estimator.alpha_
            if self.store_cv_values:
                self.cv_values_ = estimator.cv_values_
        else:
            if self.store_cv_values:
                raise ValueError("cv!=None and store_cv_values=True "
                                 " are incompatible")
            parameters = {'alpha': self.alphas}
            gs = GridSearchCV(Ridge(fit_intercept=self.fit_intercept,
                                    normalize=self.normalize),
                              parameters, cv=self.cv, scoring=self.scoring)
            gs.fit(X, y, sample_weight=sample_weight)
            estimator = gs.best_estimator_
            self.alpha_ = gs.best_estimator_.alpha

        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_

        return self
```
### 32 - sklearn/linear_model/ridge.py:

Start line: 133, End line: 154

```python
def _solve_cholesky(X, y, alpha):
    # w = inv(X^t X + alpha*Id) * X.T y
    n_samples, n_features = X.shape
    n_targets = y.shape[1]

    A = safe_sparse_dot(X.T, X, dense_output=True)
    Xy = safe_sparse_dot(X.T, y, dense_output=True)

    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    if one_alpha:
        A.flat[::n_features + 1] += alpha[0]
        return linalg.solve(A, Xy, sym_pos=True,
                            overwrite_a=True).T
    else:
        coefs = np.empty([n_targets, n_features], dtype=X.dtype)
        for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
            A.flat[::n_features + 1] += current_alpha
            coef[:] = linalg.solve(A, target, sym_pos=True,
                                   overwrite_a=False).ravel()
            A.flat[::n_features + 1] -= current_alpha
        return coefs
```
### 33 - sklearn/linear_model/ridge.py:

Start line: 580, End line: 713

```python
class Ridge(_BaseRidge, RegressorMixin):
    """Linear least squares with l2 regularization.

    Minimizes the objective function::

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alpha : {float, array-like}, shape (n_targets)
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``C^-1`` in other linear models such as
        LogisticRegression or LinearSVC. If an array is passed, penalties are
        assumed to be specific to the targets. Hence they must correspond in
        number.

    fit_intercept : boolean
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        Maximum number of iterations for conjugate gradient solver.
        For 'sparse_cg' and 'lsqr' solvers, the default value is determined
        by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.

    tol : float
        Precision of the solution.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}
        Solver to use in the computational routines:

        - 'auto' chooses the solver automatically based on the type of data.

        - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
          coefficients. More stable for singular matrices than
          'cholesky'.

        - 'cholesky' uses the standard scipy.linalg.solve function to
          obtain a closed-form solution.

        - 'sparse_cg' uses the conjugate gradient solver as found in
          scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
          more appropriate than 'cholesky' for large-scale data
          (possibility to set `tol` and `max_iter`).

        - 'lsqr' uses the dedicated regularized least-squares routine
          scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
          procedure.

        - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
          its improved, unbiased version named SAGA. Both methods also use an
          iterative procedure, and are often faster than other solvers when
          both n_samples and n_features are large. Note that 'sag' and
          'saga' fast convergence is only guaranteed on features with
          approximately the same scale. You can preprocess the data with a
          scaler from sklearn.preprocessing.

        All last five solvers support both dense and sparse data. However, only
        'sag' and 'sparse_cg' supports sparse input when `fit_intercept` is
        True.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag'.

        .. versionadded:: 0.17
           *random_state* to support Stochastic Average Gradient.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    intercept_ : float | array, shape = (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    n_iter_ : array or None, shape (n_targets,)
        Actual number of iterations for each target. Available only for
        sag and lsqr solvers. Other solvers will return None.

        .. versionadded:: 0.17

    See also
    --------
    RidgeClassifier : Ridge classifier
    RidgeCV : Ridge regression with built-in cross validation
    :class:`sklearn.kernel_ridge.KernelRidge` : Kernel ridge regression
        combines ridge regression with the kernel trick

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = Ridge(alpha=1.0)
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)

    """
```
### 39 - sklearn/linear_model/ridge.py:

Start line: 157, End line: 215

```python
def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    if copy:
        K = K.copy()

    alpha = np.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = np.sqrt(np.atleast_1d(sample_weight))
        y = y * sw[:, np.newaxis]
        K *= np.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[::n_samples + 1] += alpha[0]

        try:
            # Note: we must use overwrite_a=False in order to be able to
            #       use the fall-back solution below in case a LinAlgError
            #       is raised
            dual_coef = linalg.solve(K, y, sym_pos=True,
                                     overwrite_a=False)
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in solving dual problem. Using "
                          "least-squares solution instead.")
            dual_coef = linalg.lstsq(K, y)[0]

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[::n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[::n_samples + 1] += current_alpha

            dual_coef[:] = linalg.solve(K, target, sym_pos=True,
                                        overwrite_a=False).ravel()

            K.flat[::n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[np.newaxis, :]

        return dual_coefs.T
```
### 43 - sklearn/linear_model/ridge.py:

Start line: 1451, End line: 1499

```python
class RidgeClassifierCV(LinearClassifierMixin, _BaseRidgeCV):

    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True,
                 normalize=False, scoring=None, cv=None, class_weight=None,
                 store_cv_values=False):
        super().__init__(
            alphas=alphas, fit_intercept=fit_intercept, normalize=normalize,
            scoring=scoring, cv=cv, store_cv_values=store_cv_values)
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        """Fit the ridge classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : float or numpy array of shape (n_samples,)
            Sample weight.

        Returns
        -------
        self : object
        """
        check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                  multi_output=True)

        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            y = column_or_1d(y, warn=True)

        if self.class_weight:
            if sample_weight is None:
                sample_weight = 1.
            # modify the sample weights with the corresponding class weight
            sample_weight = (sample_weight *
                             compute_sample_weight(self.class_weight, y))

        _BaseRidgeCV.fit(self, X, Y, sample_weight=sample_weight)
        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_
```
### 46 - sklearn/linear_model/ridge.py:

Start line: 218, End line: 226

```python
def _solve_svd(X, y, alpha):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, np.newaxis]
    UTy = np.dot(U.T, y)
    d = np.zeros((s.size, alpha.size), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d_UT_y = d * UTy
    return np.dot(Vt.T, d_UT_y).T
```
### 47 - sklearn/linear_model/ridge.py:

Start line: 115, End line: 130

```python
def _solve_lsqr(X, y, alpha, max_iter=None, tol=1e-3):
    n_samples, n_features = X.shape
    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    n_iter = np.empty(y.shape[1], dtype=np.int32)

    # According to the lsqr documentation, alpha = damp^2.
    sqrt_alpha = np.sqrt(alpha)

    for i in range(y.shape[1]):
        y_column = y[:, i]
        info = sp_linalg.lsqr(X, y_column, damp=sqrt_alpha[i],
                              atol=tol, btol=tol, iter_lim=max_iter)
        coefs[i] = info[0]
        n_iter[i] = info[2]

    return coefs, n_iter
```
### 62 - sklearn/linear_model/ridge.py:

Start line: 875, End line: 921

```python
class RidgeClassifier(LinearClassifierMixin, _BaseRidge):

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples,n_features]
            Training data

        y : array-like, shape = [n_samples]
            Target values

        sample_weight : float or numpy array of shape (n_samples,)
            Sample weight.

            .. versionadded:: 0.17
               *sample_weight* support to Classifier.

        Returns
        -------
        self : returns an instance of self.
        """
        check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                  multi_output=True)

        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            y = column_or_1d(y, warn=True)
        else:
            # we don't (yet) support multi-label classification in Ridge
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        if self.class_weight:
            if sample_weight is None:
                sample_weight = 1.
            # modify the sample weights with the corresponding class weight
            sample_weight = (sample_weight *
                             compute_sample_weight(self.class_weight, y))

        super().fit(X, Y, sample_weight=sample_weight)
        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_
```

# scikit-learn__scikit-learn-25697

| **scikit-learn/scikit-learn** | `097c3683a73c5805a84e6eada71e4928cb35496e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 429 |
| **Avg pos** | 122.0 |
| **Min pos** | 1 |
| **Max pos** | 18 |
| **Top file pos** | 1 |
| **Missing snippets** | 21 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/linear_model/_bayes.py b/sklearn/linear_model/_bayes.py
--- a/sklearn/linear_model/_bayes.py
+++ b/sklearn/linear_model/_bayes.py
@@ -5,6 +5,7 @@
 # Authors: V. Michel, F. Pedregosa, A. Gramfort
 # License: BSD 3 clause
 
+import warnings
 from math import log
 from numbers import Integral, Real
 import numpy as np
@@ -15,7 +16,49 @@
 from ..utils.extmath import fast_logdet
 from scipy.linalg import pinvh
 from ..utils.validation import _check_sample_weight
-from ..utils._param_validation import Interval
+from ..utils._param_validation import Interval, Hidden, StrOptions
+
+
+# TODO(1.5) Remove
+def _deprecate_n_iter(n_iter, max_iter):
+    """Deprecates n_iter in favour of max_iter. Checks if the n_iter has been
+    used instead of max_iter and generates a deprecation warning if True.
+
+    Parameters
+    ----------
+    n_iter : int,
+        Value of n_iter attribute passed by the estimator.
+
+    max_iter : int, default=None
+        Value of max_iter attribute passed by the estimator.
+        If `None`, it corresponds to `max_iter=300`.
+
+    Returns
+    -------
+    max_iter : int,
+        Value of max_iter which shall further be used by the estimator.
+
+    Notes
+    -----
+    This function should be completely removed in 1.5.
+    """
+    if n_iter != "deprecated":
+        if max_iter is not None:
+            raise ValueError(
+                "Both `n_iter` and `max_iter` attributes were set. Attribute"
+                " `n_iter` was deprecated in version 1.3 and will be removed in"
+                " 1.5. To avoid this error, only set the `max_iter` attribute."
+            )
+        warnings.warn(
+            "'n_iter' was renamed to 'max_iter' in version 1.3 and "
+            "will be removed in 1.5",
+            FutureWarning,
+        )
+        max_iter = n_iter
+    elif max_iter is None:
+        max_iter = 300
+    return max_iter
+
 
 ###############################################################################
 # BayesianRidge regression
@@ -32,8 +75,12 @@ class BayesianRidge(RegressorMixin, LinearModel):
 
     Parameters
     ----------
-    n_iter : int, default=300
-        Maximum number of iterations. Should be greater than or equal to 1.
+    max_iter : int, default=None
+        Maximum number of iterations over the complete dataset before
+        stopping independently of any early stopping criterion. If `None`, it
+        corresponds to `max_iter=300`.
+
+        .. versionchanged:: 1.3
 
     tol : float, default=1e-3
         Stop the algorithm if w has converged.
@@ -83,6 +130,13 @@ class BayesianRidge(RegressorMixin, LinearModel):
     verbose : bool, default=False
         Verbose mode when fitting the model.
 
+    n_iter : int
+        Maximum number of iterations. Should be greater than or equal to 1.
+
+        .. deprecated:: 1.3
+           `n_iter` is deprecated in 1.3 and will be removed in 1.5. Use
+           `max_iter` instead.
+
     Attributes
     ----------
     coef_ : array-like of shape (n_features,)
@@ -90,7 +144,7 @@ class BayesianRidge(RegressorMixin, LinearModel):
 
     intercept_ : float
         Independent term in decision function. Set to 0.0 if
-        ``fit_intercept = False``.
+        `fit_intercept = False`.
 
     alpha_ : float
        Estimated precision of the noise.
@@ -162,7 +216,7 @@ class BayesianRidge(RegressorMixin, LinearModel):
     """
 
     _parameter_constraints: dict = {
-        "n_iter": [Interval(Integral, 1, None, closed="left")],
+        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
         "tol": [Interval(Real, 0, None, closed="neither")],
         "alpha_1": [Interval(Real, 0, None, closed="left")],
         "alpha_2": [Interval(Real, 0, None, closed="left")],
@@ -174,12 +228,16 @@ class BayesianRidge(RegressorMixin, LinearModel):
         "fit_intercept": ["boolean"],
         "copy_X": ["boolean"],
         "verbose": ["verbose"],
+        "n_iter": [
+            Interval(Integral, 1, None, closed="left"),
+            Hidden(StrOptions({"deprecated"})),
+        ],
     }
 
     def __init__(
         self,
         *,
-        n_iter=300,
+        max_iter=None,  # TODO(1.5): Set to 300
         tol=1.0e-3,
         alpha_1=1.0e-6,
         alpha_2=1.0e-6,
@@ -191,8 +249,9 @@ def __init__(
         fit_intercept=True,
         copy_X=True,
         verbose=False,
+        n_iter="deprecated",  # TODO(1.5): Remove
     ):
-        self.n_iter = n_iter
+        self.max_iter = max_iter
         self.tol = tol
         self.alpha_1 = alpha_1
         self.alpha_2 = alpha_2
@@ -204,6 +263,7 @@ def __init__(
         self.fit_intercept = fit_intercept
         self.copy_X = copy_X
         self.verbose = verbose
+        self.n_iter = n_iter
 
     def fit(self, X, y, sample_weight=None):
         """Fit the model.
@@ -228,6 +288,8 @@ def fit(self, X, y, sample_weight=None):
         """
         self._validate_params()
 
+        max_iter = _deprecate_n_iter(self.n_iter, self.max_iter)
+
         X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True)
 
         if sample_weight is not None:
@@ -274,7 +336,7 @@ def fit(self, X, y, sample_weight=None):
         eigen_vals_ = S**2
 
         # Convergence loop of the bayesian ridge regression
-        for iter_ in range(self.n_iter):
+        for iter_ in range(max_iter):
 
             # update posterior mean coef_ based on alpha_ and lambda_ and
             # compute corresponding rmse
@@ -430,8 +492,10 @@ class ARDRegression(RegressorMixin, LinearModel):
 
     Parameters
     ----------
-    n_iter : int, default=300
-        Maximum number of iterations.
+    max_iter : int, default=None
+        Maximum number of iterations. If `None`, it corresponds to `max_iter=300`.
+
+        .. versionchanged:: 1.3
 
     tol : float, default=1e-3
         Stop the algorithm if w has converged.
@@ -470,6 +534,13 @@ class ARDRegression(RegressorMixin, LinearModel):
     verbose : bool, default=False
         Verbose mode when fitting the model.
 
+    n_iter : int
+        Maximum number of iterations.
+
+        .. deprecated:: 1.3
+           `n_iter` is deprecated in 1.3 and will be removed in 1.5. Use
+           `max_iter` instead.
+
     Attributes
     ----------
     coef_ : array-like of shape (n_features,)
@@ -487,6 +558,11 @@ class ARDRegression(RegressorMixin, LinearModel):
     scores_ : float
         if computed, value of the objective function (to be maximized)
 
+    n_iter_ : int
+        The actual number of iterations to reach the stopping criterion.
+
+        .. versionadded:: 1.3
+
     intercept_ : float
         Independent term in decision function. Set to 0.0 if
         ``fit_intercept = False``.
@@ -542,7 +618,7 @@ class ARDRegression(RegressorMixin, LinearModel):
     """
 
     _parameter_constraints: dict = {
-        "n_iter": [Interval(Integral, 1, None, closed="left")],
+        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
         "tol": [Interval(Real, 0, None, closed="left")],
         "alpha_1": [Interval(Real, 0, None, closed="left")],
         "alpha_2": [Interval(Real, 0, None, closed="left")],
@@ -553,12 +629,16 @@ class ARDRegression(RegressorMixin, LinearModel):
         "fit_intercept": ["boolean"],
         "copy_X": ["boolean"],
         "verbose": ["verbose"],
+        "n_iter": [
+            Interval(Integral, 1, None, closed="left"),
+            Hidden(StrOptions({"deprecated"})),
+        ],
     }
 
     def __init__(
         self,
         *,
-        n_iter=300,
+        max_iter=None,  # TODO(1.5): Set to 300
         tol=1.0e-3,
         alpha_1=1.0e-6,
         alpha_2=1.0e-6,
@@ -569,8 +649,9 @@ def __init__(
         fit_intercept=True,
         copy_X=True,
         verbose=False,
+        n_iter="deprecated",  # TODO(1.5): Remove
     ):
-        self.n_iter = n_iter
+        self.max_iter = max_iter
         self.tol = tol
         self.fit_intercept = fit_intercept
         self.alpha_1 = alpha_1
@@ -581,6 +662,7 @@ def __init__(
         self.threshold_lambda = threshold_lambda
         self.copy_X = copy_X
         self.verbose = verbose
+        self.n_iter = n_iter
 
     def fit(self, X, y):
         """Fit the model according to the given training data and parameters.
@@ -603,6 +685,8 @@ def fit(self, X, y):
 
         self._validate_params()
 
+        max_iter = _deprecate_n_iter(self.n_iter, self.max_iter)
+
         X, y = self._validate_data(
             X, y, dtype=[np.float64, np.float32], y_numeric=True, ensure_min_samples=2
         )
@@ -648,7 +732,7 @@ def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
             else self._update_sigma_woodbury
         )
         # Iterative procedure of ARDRegression
-        for iter_ in range(self.n_iter):
+        for iter_ in range(max_iter):
             sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
             coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
 
@@ -688,6 +772,8 @@ def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
             if not keep_lambda.any():
                 break
 
+        self.n_iter_ = iter_ + 1
+
         if keep_lambda.any():
             # update sigma and mu using updated params from the last iteration
             sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/_bayes.py | 8 | 8 | 2 | 1 | 1652
| sklearn/linear_model/_bayes.py | 18 | 18 | 2 | 1 | 1652
| sklearn/linear_model/_bayes.py | 35 | 36 | 2 | 1 | 1652
| sklearn/linear_model/_bayes.py | 86 | 86 | 2 | 1 | 1652
| sklearn/linear_model/_bayes.py | 93 | 93 | 2 | 1 | 1652
| sklearn/linear_model/_bayes.py | 165 | 165 | 1 | 1 | 429
| sklearn/linear_model/_bayes.py | 177 | 177 | 1 | 1 | 429
| sklearn/linear_model/_bayes.py | 194 | 194 | 1 | 1 | 429
| sklearn/linear_model/_bayes.py | 207 | 207 | - | 1 | -
| sklearn/linear_model/_bayes.py | 231 | 231 | 13 | 1 | 8159
| sklearn/linear_model/_bayes.py | 277 | 277 | 3 | 1 | 2216
| sklearn/linear_model/_bayes.py | 433 | 434 | 5 | 1 | 4027
| sklearn/linear_model/_bayes.py | 473 | 473 | 5 | 1 | 4027
| sklearn/linear_model/_bayes.py | 490 | 490 | 5 | 1 | 4027
| sklearn/linear_model/_bayes.py | 545 | 545 | 9 | 1 | 5510
| sklearn/linear_model/_bayes.py | 556 | 556 | 9 | 1 | 5510
| sklearn/linear_model/_bayes.py | 572 | 572 | 9 | 1 | 5510
| sklearn/linear_model/_bayes.py | 584 | 584 | - | 1 | -
| sklearn/linear_model/_bayes.py | 606 | 606 | 15 | 1 | 9037
| sklearn/linear_model/_bayes.py | 651 | 651 | 18 | 1 | 16098
| sklearn/linear_model/_bayes.py | 691 | 691 | 18 | 1 | 16098


## Problem Statement

```
Deprecate `n_iter` in favor of `max_iter` for consistency
`BayesianRidge` and `ARDRegression` are exposing the parameter `n_iter` instead of `max_iter` as in other models. I think that we should deprecate `n_iter` and rename it `max_iter` to be consistent.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/linear_model/_bayes.py** | 164 | 206| 429 | 429 | 6410 | 
| **-> 2 <-** | **1 sklearn/linear_model/_bayes.py** | 1 | 162| 1223 | 1652 | 6410 | 
| **-> 3 <-** | **1 sklearn/linear_model/_bayes.py** | 277 | 328| 564 | 2216 | 6410 | 
| 4 | 2 examples/linear_model/plot_ard.py | 1 | 128| 822 | 3038 | 8138 | 
| **-> 5 <-** | **2 sklearn/linear_model/_bayes.py** | 416 | 542| 989 | 4027 | 8138 | 
| 6 | 3 sklearn/linear_model/_ridge.py | 783 | 820| 288 | 4315 | 29822 | 
| 7 | **3 sklearn/linear_model/_bayes.py** | 360 | 381| 224 | 4539 | 29822 | 
| 8 | 4 examples/linear_model/plot_huber_vs_ridge.py | 1 | 65| 575 | 5114 | 30419 | 
| **-> 9 <-** | **4 sklearn/linear_model/_bayes.py** | 544 | 583| 396 | 5510 | 30419 | 
| 10 | 5 asv_benchmarks/benchmarks/linear_model.py | 67 | 108| 255 | 5765 | 31902 | 
| 11 | 6 sklearn/linear_model/_base.py | 48 | 126| 618 | 6383 | 38877 | 
| 12 | 7 sklearn/kernel_ridge.py | 1 | 132| 1248 | 7631 | 40987 | 
| **-> 13 <-** | **7 sklearn/linear_model/_bayes.py** | 208 | 276| 528 | 8159 | 40987 | 
| 14 | 7 sklearn/linear_model/_ridge.py | 874 | 1149| 368 | 8527 | 40987 | 
| **-> 15 <-** | **7 sklearn/linear_model/_bayes.py** | 585 | 650| 510 | 9037 | 40987 | 
| 16 | 7 sklearn/kernel_ridge.py | 134 | 171| 321 | 9358 | 40987 | 
| 17 | 8 sklearn/utils/estimator_checks.py | 2707 | 3430| 6190 | 15548 | 77514 | 
| **-> 18 <-** | **8 sklearn/linear_model/_bayes.py** | 651 | 703| 550 | 16098 | 77514 | 
| 19 | 9 sklearn/linear_model/__init__.py | 1 | 103| 706 | 16804 | 78220 | 
| 20 | 10 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 91| 780 | 17584 | 79555 | 
| 21 | 11 sklearn/linear_model/_ransac.py | 24 | 622| 246 | 17830 | 84455 | 
| 22 | **11 sklearn/linear_model/_bayes.py** | 383 | 413| 317 | 18147 | 84455 | 
| 23 | 11 sklearn/linear_model/_ridge.py | 1 | 44| 268 | 18415 | 84455 | 
| 24 | 12 examples/inspection/plot_linear_model_coefficient_interpretation.py | 287 | 382| 796 | 19211 | 90760 | 
| 25 | 12 sklearn/linear_model/_ridge.py | 1397 | 1425| 174 | 19385 | 90760 | 
| 26 | 13 examples/linear_model/plot_bayesian_ridge_curvefit.py | 64 | 93| 363 | 19748 | 91591 | 
| 27 | 14 sklearn/impute/_iterative.py | 651 | 682| 272 | 20020 | 99420 | 
| 28 | 14 sklearn/linear_model/_ridge.py | 822 | 872| 497 | 20517 | 99420 | 
| 29 | 14 examples/inspection/plot_linear_model_coefficient_interpretation.py | 483 | 592| 894 | 21411 | 99420 | 
| 30 | 15 examples/linear_model/plot_ridge_path.py | 1 | 68| 435 | 21846 | 99884 | 
| 31 | 15 sklearn/linear_model/_ridge.py | 1095 | 1116| 129 | 21975 | 99884 | 
| 32 | 15 sklearn/utils/estimator_checks.py | 3433 | 3543| 957 | 22932 | 99884 | 
| 33 | 15 sklearn/linear_model/_ransac.py | 516 | 562| 402 | 23334 | 99884 | 
| 34 | 15 examples/inspection/plot_linear_model_coefficient_interpretation.py | 384 | 482| 752 | 24086 | 99884 | 
| 35 | 16 sklearn/linear_model/_theil_sen.py | 361 | 396| 320 | 24406 | 103605 | 
| 36 | 17 benchmarks/bench_sgd_regression.py | 4 | 153| 1324 | 25730 | 104956 | 
| 37 | 18 sklearn/metrics/_regression.py | 1000 | 1036| 256 | 25986 | 119077 | 
| 38 | 18 sklearn/linear_model/_ridge.py | 2103 | 2132| 243 | 26229 | 119077 | 
| 39 | 18 examples/linear_model/plot_ard.py | 129 | 211| 747 | 26976 | 119077 | 
| 40 | 18 sklearn/linear_model/_ransac.py | 61 | 234| 1664 | 28640 | 119077 | 
| 41 | 19 examples/release_highlights/plot_release_highlights_0_24_0.py | 211 | 265| 510 | 29150 | 121515 | 
| 42 | **19 sklearn/linear_model/_bayes.py** | 723 | 732| 134 | 29284 | 121515 | 
| 43 | **19 sklearn/linear_model/_bayes.py** | 705 | 721| 233 | 29517 | 121515 | 
| 44 | 19 sklearn/linear_model/_ridge.py | 922 | 1093| 1662 | 31179 | 121515 | 
| 45 | 20 sklearn/utils/fixes.py | 108 | 128| 195 | 31374 | 122940 | 
| 46 | 21 sklearn/preprocessing/_discretization.py | 329 | 352| 247 | 31621 | 126849 | 
| 47 | 22 sklearn/experimental/enable_iterative_imputer.py | 1 | 21| 160 | 31781 | 127009 | 
| 48 | 22 sklearn/linear_model/_ridge.py | 2027 | 2100| 618 | 32399 | 127009 | 
| 49 | 22 sklearn/linear_model/_theil_sen.py | 208 | 323| 1093 | 33492 | 127009 | 
| 50 | 23 examples/linear_model/plot_poisson_regression_non_normal_loss.py | 289 | 390| 988 | 34480 | 132342 | 
| 51 | 23 sklearn/linear_model/_ridge.py | 2512 | 2536| 177 | 34657 | 132342 | 
| 52 | 24 sklearn/utils/deprecation.py | 58 | 76| 128 | 34785 | 133054 | 
| 53 | 25 sklearn/linear_model/_huber.py | 250 | 274| 222 | 35007 | 136180 | 
| 54 | 25 examples/linear_model/plot_bayesian_ridge_curvefit.py | 1 | 63| 432 | 35439 | 136180 | 
| 55 | 25 sklearn/linear_model/_ransac.py | 431 | 515| 683 | 36122 | 136180 | 
| 56 | 26 sklearn/linear_model/_least_angle.py | 2016 | 2181| 1594 | 37716 | 156051 | 
| 57 | 27 sklearn/metrics/pairwise.py | 1634 | 1666| 214 | 37930 | 174944 | 
| 58 | 28 sklearn/decomposition/_dict_learning.py | 647 | 1070| 230 | 38160 | 193894 | 
| 59 | 28 sklearn/kernel_ridge.py | 173 | 217| 358 | 38518 | 193894 | 


### Hint

```
@glemaitre I would like to attempt this one !
@saucam please go ahead and propose a pull-request. You can refer to the following documentation page to follow our deprecation rule: https://scikit-learn.org/dev/developers/contributing.html#deprecation
@saucam ,let me know incase you need help. We can work together on this issue if it is fine with you. 
@jpangas sorry but I lost track of this one. You can go ahead with your changes as it looks like you already have some progress.
Thank you for getting back to me. I am working on the changes, should be done within the week. 
```

## Patch

```diff
diff --git a/sklearn/linear_model/_bayes.py b/sklearn/linear_model/_bayes.py
--- a/sklearn/linear_model/_bayes.py
+++ b/sklearn/linear_model/_bayes.py
@@ -5,6 +5,7 @@
 # Authors: V. Michel, F. Pedregosa, A. Gramfort
 # License: BSD 3 clause
 
+import warnings
 from math import log
 from numbers import Integral, Real
 import numpy as np
@@ -15,7 +16,49 @@
 from ..utils.extmath import fast_logdet
 from scipy.linalg import pinvh
 from ..utils.validation import _check_sample_weight
-from ..utils._param_validation import Interval
+from ..utils._param_validation import Interval, Hidden, StrOptions
+
+
+# TODO(1.5) Remove
+def _deprecate_n_iter(n_iter, max_iter):
+    """Deprecates n_iter in favour of max_iter. Checks if the n_iter has been
+    used instead of max_iter and generates a deprecation warning if True.
+
+    Parameters
+    ----------
+    n_iter : int,
+        Value of n_iter attribute passed by the estimator.
+
+    max_iter : int, default=None
+        Value of max_iter attribute passed by the estimator.
+        If `None`, it corresponds to `max_iter=300`.
+
+    Returns
+    -------
+    max_iter : int,
+        Value of max_iter which shall further be used by the estimator.
+
+    Notes
+    -----
+    This function should be completely removed in 1.5.
+    """
+    if n_iter != "deprecated":
+        if max_iter is not None:
+            raise ValueError(
+                "Both `n_iter` and `max_iter` attributes were set. Attribute"
+                " `n_iter` was deprecated in version 1.3 and will be removed in"
+                " 1.5. To avoid this error, only set the `max_iter` attribute."
+            )
+        warnings.warn(
+            "'n_iter' was renamed to 'max_iter' in version 1.3 and "
+            "will be removed in 1.5",
+            FutureWarning,
+        )
+        max_iter = n_iter
+    elif max_iter is None:
+        max_iter = 300
+    return max_iter
+
 
 ###############################################################################
 # BayesianRidge regression
@@ -32,8 +75,12 @@ class BayesianRidge(RegressorMixin, LinearModel):
 
     Parameters
     ----------
-    n_iter : int, default=300
-        Maximum number of iterations. Should be greater than or equal to 1.
+    max_iter : int, default=None
+        Maximum number of iterations over the complete dataset before
+        stopping independently of any early stopping criterion. If `None`, it
+        corresponds to `max_iter=300`.
+
+        .. versionchanged:: 1.3
 
     tol : float, default=1e-3
         Stop the algorithm if w has converged.
@@ -83,6 +130,13 @@ class BayesianRidge(RegressorMixin, LinearModel):
     verbose : bool, default=False
         Verbose mode when fitting the model.
 
+    n_iter : int
+        Maximum number of iterations. Should be greater than or equal to 1.
+
+        .. deprecated:: 1.3
+           `n_iter` is deprecated in 1.3 and will be removed in 1.5. Use
+           `max_iter` instead.
+
     Attributes
     ----------
     coef_ : array-like of shape (n_features,)
@@ -90,7 +144,7 @@ class BayesianRidge(RegressorMixin, LinearModel):
 
     intercept_ : float
         Independent term in decision function. Set to 0.0 if
-        ``fit_intercept = False``.
+        `fit_intercept = False`.
 
     alpha_ : float
        Estimated precision of the noise.
@@ -162,7 +216,7 @@ class BayesianRidge(RegressorMixin, LinearModel):
     """
 
     _parameter_constraints: dict = {
-        "n_iter": [Interval(Integral, 1, None, closed="left")],
+        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
         "tol": [Interval(Real, 0, None, closed="neither")],
         "alpha_1": [Interval(Real, 0, None, closed="left")],
         "alpha_2": [Interval(Real, 0, None, closed="left")],
@@ -174,12 +228,16 @@ class BayesianRidge(RegressorMixin, LinearModel):
         "fit_intercept": ["boolean"],
         "copy_X": ["boolean"],
         "verbose": ["verbose"],
+        "n_iter": [
+            Interval(Integral, 1, None, closed="left"),
+            Hidden(StrOptions({"deprecated"})),
+        ],
     }
 
     def __init__(
         self,
         *,
-        n_iter=300,
+        max_iter=None,  # TODO(1.5): Set to 300
         tol=1.0e-3,
         alpha_1=1.0e-6,
         alpha_2=1.0e-6,
@@ -191,8 +249,9 @@ def __init__(
         fit_intercept=True,
         copy_X=True,
         verbose=False,
+        n_iter="deprecated",  # TODO(1.5): Remove
     ):
-        self.n_iter = n_iter
+        self.max_iter = max_iter
         self.tol = tol
         self.alpha_1 = alpha_1
         self.alpha_2 = alpha_2
@@ -204,6 +263,7 @@ def __init__(
         self.fit_intercept = fit_intercept
         self.copy_X = copy_X
         self.verbose = verbose
+        self.n_iter = n_iter
 
     def fit(self, X, y, sample_weight=None):
         """Fit the model.
@@ -228,6 +288,8 @@ def fit(self, X, y, sample_weight=None):
         """
         self._validate_params()
 
+        max_iter = _deprecate_n_iter(self.n_iter, self.max_iter)
+
         X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True)
 
         if sample_weight is not None:
@@ -274,7 +336,7 @@ def fit(self, X, y, sample_weight=None):
         eigen_vals_ = S**2
 
         # Convergence loop of the bayesian ridge regression
-        for iter_ in range(self.n_iter):
+        for iter_ in range(max_iter):
 
             # update posterior mean coef_ based on alpha_ and lambda_ and
             # compute corresponding rmse
@@ -430,8 +492,10 @@ class ARDRegression(RegressorMixin, LinearModel):
 
     Parameters
     ----------
-    n_iter : int, default=300
-        Maximum number of iterations.
+    max_iter : int, default=None
+        Maximum number of iterations. If `None`, it corresponds to `max_iter=300`.
+
+        .. versionchanged:: 1.3
 
     tol : float, default=1e-3
         Stop the algorithm if w has converged.
@@ -470,6 +534,13 @@ class ARDRegression(RegressorMixin, LinearModel):
     verbose : bool, default=False
         Verbose mode when fitting the model.
 
+    n_iter : int
+        Maximum number of iterations.
+
+        .. deprecated:: 1.3
+           `n_iter` is deprecated in 1.3 and will be removed in 1.5. Use
+           `max_iter` instead.
+
     Attributes
     ----------
     coef_ : array-like of shape (n_features,)
@@ -487,6 +558,11 @@ class ARDRegression(RegressorMixin, LinearModel):
     scores_ : float
         if computed, value of the objective function (to be maximized)
 
+    n_iter_ : int
+        The actual number of iterations to reach the stopping criterion.
+
+        .. versionadded:: 1.3
+
     intercept_ : float
         Independent term in decision function. Set to 0.0 if
         ``fit_intercept = False``.
@@ -542,7 +618,7 @@ class ARDRegression(RegressorMixin, LinearModel):
     """
 
     _parameter_constraints: dict = {
-        "n_iter": [Interval(Integral, 1, None, closed="left")],
+        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
         "tol": [Interval(Real, 0, None, closed="left")],
         "alpha_1": [Interval(Real, 0, None, closed="left")],
         "alpha_2": [Interval(Real, 0, None, closed="left")],
@@ -553,12 +629,16 @@ class ARDRegression(RegressorMixin, LinearModel):
         "fit_intercept": ["boolean"],
         "copy_X": ["boolean"],
         "verbose": ["verbose"],
+        "n_iter": [
+            Interval(Integral, 1, None, closed="left"),
+            Hidden(StrOptions({"deprecated"})),
+        ],
     }
 
     def __init__(
         self,
         *,
-        n_iter=300,
+        max_iter=None,  # TODO(1.5): Set to 300
         tol=1.0e-3,
         alpha_1=1.0e-6,
         alpha_2=1.0e-6,
@@ -569,8 +649,9 @@ def __init__(
         fit_intercept=True,
         copy_X=True,
         verbose=False,
+        n_iter="deprecated",  # TODO(1.5): Remove
     ):
-        self.n_iter = n_iter
+        self.max_iter = max_iter
         self.tol = tol
         self.fit_intercept = fit_intercept
         self.alpha_1 = alpha_1
@@ -581,6 +662,7 @@ def __init__(
         self.threshold_lambda = threshold_lambda
         self.copy_X = copy_X
         self.verbose = verbose
+        self.n_iter = n_iter
 
     def fit(self, X, y):
         """Fit the model according to the given training data and parameters.
@@ -603,6 +685,8 @@ def fit(self, X, y):
 
         self._validate_params()
 
+        max_iter = _deprecate_n_iter(self.n_iter, self.max_iter)
+
         X, y = self._validate_data(
             X, y, dtype=[np.float64, np.float32], y_numeric=True, ensure_min_samples=2
         )
@@ -648,7 +732,7 @@ def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
             else self._update_sigma_woodbury
         )
         # Iterative procedure of ARDRegression
-        for iter_ in range(self.n_iter):
+        for iter_ in range(max_iter):
             sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
             coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
 
@@ -688,6 +772,8 @@ def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
             if not keep_lambda.any():
                 break
 
+        self.n_iter_ = iter_ + 1
+
         if keep_lambda.any():
             # update sigma and mu using updated params from the last iteration
             sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_bayes.py b/sklearn/linear_model/tests/test_bayes.py
--- a/sklearn/linear_model/tests/test_bayes.py
+++ b/sklearn/linear_model/tests/test_bayes.py
@@ -73,7 +73,7 @@ def test_bayesian_ridge_score_values():
         alpha_2=alpha_2,
         lambda_1=lambda_1,
         lambda_2=lambda_2,
-        n_iter=1,
+        max_iter=1,
         fit_intercept=False,
         compute_score=True,
     )
@@ -174,7 +174,7 @@ def test_update_of_sigma_in_ard():
     # of the ARDRegression algorithm. See issue #10128.
     X = np.array([[1, 0], [0, 0]])
     y = np.array([0, 0])
-    clf = ARDRegression(n_iter=1)
+    clf = ARDRegression(max_iter=1)
     clf.fit(X, y)
     # With the inputs above, ARDRegression prunes both of the two coefficients
     # in the first iteration. Hence, the expected shape of `sigma_` is (0, 0).
@@ -292,3 +292,33 @@ def test_dtype_correctness(Estimator):
     coef_32 = model.fit(X.astype(np.float32), y).coef_
     coef_64 = model.fit(X.astype(np.float64), y).coef_
     np.testing.assert_allclose(coef_32, coef_64, rtol=1e-4)
+
+
+# TODO(1.5) remove
+@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
+def test_bayesian_ridge_ard_n_iter_deprecated(Estimator):
+    """Check the deprecation warning of `n_iter`."""
+    depr_msg = (
+        "'n_iter' was renamed to 'max_iter' in version 1.3 and will be removed in 1.5"
+    )
+    X, y = diabetes.data, diabetes.target
+    model = Estimator(n_iter=5)
+
+    with pytest.warns(FutureWarning, match=depr_msg):
+        model.fit(X, y)
+
+
+# TODO(1.5) remove
+@pytest.mark.parametrize("Estimator", [BayesianRidge, ARDRegression])
+def test_bayesian_ridge_ard_max_iter_and_n_iter_both_set(Estimator):
+    """Check that a ValueError is raised when both `max_iter` and `n_iter` are set."""
+    err_msg = (
+        "Both `n_iter` and `max_iter` attributes were set. Attribute"
+        " `n_iter` was deprecated in version 1.3 and will be removed in"
+        " 1.5. To avoid this error, only set the `max_iter` attribute."
+    )
+    X, y = diabetes.data, diabetes.target
+    model = Estimator(n_iter=5, max_iter=5)
+
+    with pytest.raises(ValueError, match=err_msg):
+        model.fit(X, y)

```


## Code snippets

### 1 - sklearn/linear_model/_bayes.py:

Start line: 164, End line: 206

```python
class BayesianRidge(RegressorMixin, LinearModel):

    _parameter_constraints: dict = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        n_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose
```
### 2 - sklearn/linear_model/_bayes.py:

Start line: 1, End line: 162

```python
"""
Various bayesian regression
"""

from math import log
from numbers import Integral, Real
import numpy as np
from scipy import linalg

from ._base import LinearModel, _preprocess_data, _rescale_data
from ..base import RegressorMixin
from ..utils.extmath import fast_logdet
from scipy.linalg import pinvh
from ..utils.validation import _check_sample_weight
from ..utils._param_validation import Interval

###############################################################################
# BayesianRidge regression


class BayesianRidge(RegressorMixin, LinearModel):
    """Bayesian ridge regression.

    Fit a Bayesian ridge model. See the Notes section for details on this
    implementation and the optimization of the regularization parameters
    lambda (precision of the weights) and alpha (precision of the noise).

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations. Should be greater than or equal to 1.

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
        If not set, alpha_init is 1/Var(y).

            .. versionadded:: 0.22

    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
        If not set, lambda_init is 1.

            .. versionadded:: 0.22

    compute_score : bool, default=False
        If True, compute the log marginal likelihood at each iteration of the
        optimization.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        The intercept is not treated as a probabilistic parameter
        and thus has no associated variance. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    alpha_ : float
       Estimated precision of the noise.

    lambda_ : float
       Estimated precision of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights

    scores_ : array-like of shape (n_iter_+1,)
        If computed_score is True, value of the log marginal likelihood (to be
        maximized) at each iteration of the optimization. The array starts
        with the value of the log marginal likelihood obtained for the initial
        values of alpha and lambda and ends with the value obtained for the
        estimated alpha and lambda.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.

    X_offset_ : ndarray of shape (n_features,)
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : ndarray of shape (n_features,)
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    ARDRegression : Bayesian ARD regression.

    Notes
    -----
    There exist several strategies to perform Bayesian ridge regression. This
    implementation is based on the algorithm described in Appendix A of
    (Tipping, 2001) where updates of the regularization parameters are done as
    suggested in (MacKay, 1992). Note that according to A New
    View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
    update rules do not guarantee that the marginal likelihood is increasing
    between two consecutive iterations of the optimization.

    References
    ----------
    D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
    Vol. 4, No. 3, 1992.

    M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
    Journal of Machine Learning Research, Vol. 1, 2001.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.BayesianRidge()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    BayesianRidge()
    >>> clf.predict([[1, 1]])
    array([1.])
    """
```
### 3 - sklearn/linear_model/_bayes.py:

Start line: 277, End line: 328

```python
class BayesianRidge(RegressorMixin, LinearModel):

    def fit(self, X, y, sample_weight=None):
        # ... other code
        for iter_ in range(self.n_iter):

            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(
                X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
            )
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(
                    n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_
                )
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_))
            lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_**2) + 2 * lambda_2)
            alpha_ = (n_samples - gamma_ + 2 * alpha_1) / (rmse_ + 2 * alpha_2)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(
            X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
        )
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(
                n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_
            )
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(
            Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis]
        )
        self.sigma_ = (1.0 / alpha_) * scaled_sigma_

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self
```
### 4 - examples/linear_model/plot_ard.py:

Start line: 1, End line: 128

```python
"""
====================================
Comparing Linear Bayesian Regressors
====================================

This example compares two different bayesian regressors:

 - a :ref:`automatic_relevance_determination`
 - a :ref:`bayesian_ridge_regression`

In the first part, we use an :ref:`ordinary_least_squares` (OLS) model as a
baseline for comparing the models' coefficients with respect to the true
coefficients. Thereafter, we show that the estimation of such models is done by
iteratively maximizing the marginal log-likelihood of the observations.

In the last section we plot predictions and uncertainties for the ARD and the
Bayesian Ridge regressions using a polynomial feature expansion to fit a
non-linear relationship between `X` and `y`.

"""

from sklearn.datasets import make_regression

X, y, true_weights = make_regression(
    n_samples=100,
    n_features=100,
    n_informative=10,
    noise=8,
    coef=True,
    random_state=42,
)

# %%
# Fit the regressors
# ------------------
#
# We now fit both Bayesian models and the OLS to later compare the models'
# coefficients.

import pandas as pd
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge

olr = LinearRegression().fit(X, y)
brr = BayesianRidge(compute_score=True, n_iter=30).fit(X, y)
ard = ARDRegression(compute_score=True, n_iter=30).fit(X, y)
df = pd.DataFrame(
    {
        "Weights of true generative process": true_weights,
        "ARDRegression": ard.coef_,
        "BayesianRidge": brr.coef_,
        "LinearRegression": olr.coef_,
    }
)

# %%
# Plot the true and estimated coefficients
# ----------------------------------------
#
# Now we compare the coefficients of each model with the weights of
# the true generative model.
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    df.T,
    norm=SymLogNorm(linthresh=10e-4, vmin=-80, vmax=80),
    cbar_kws={"label": "coefficients' values"},
    cmap="seismic_r",
)
plt.ylabel("linear model")
plt.xlabel("coefficients")
plt.tight_layout(rect=(0, 0, 1, 0.95))
_ = plt.title("Models' coefficients")

# %%
# Due to the added noise, none of the models recover the true weights. Indeed,
# all models always have more than 10 non-zero coefficients. Compared to the OLS
# estimator, the coefficients using a Bayesian Ridge regression are slightly
# shifted toward zero, which stabilises them. The ARD regression provides a
# sparser solution: some of the non-informative coefficients are set exactly to
# zero, while shifting others closer to zero. Some non-informative coefficients
# are still present and retain large values.

# %%
# Plot the marginal log-likelihood
# --------------------------------
import numpy as np

ard_scores = -np.array(ard.scores_)
brr_scores = -np.array(brr.scores_)
plt.plot(ard_scores, color="navy", label="ARD")
plt.plot(brr_scores, color="red", label="BayesianRidge")
plt.ylabel("Log-likelihood")
plt.xlabel("Iterations")
plt.xlim(1, 30)
plt.legend()
_ = plt.title("Models log-likelihood")

# %%
# Indeed, both models minimize the log-likelihood up to an arbitrary cutoff
# defined by the `n_iter` parameter.
#
# Bayesian regressions with polynomial feature expansion
# ======================================================
# Generate synthetic dataset
# --------------------------
# We create a target that is a non-linear function of the input feature.
# Noise following a standard uniform distribution is added.

from sklearn.pipeline import make_pipeline
```
### 5 - sklearn/linear_model/_bayes.py:

Start line: 416, End line: 542

```python
###############################################################################
# ARD (Automatic Relevance Determination) regression


class ARDRegression(RegressorMixin, LinearModel):
    """Bayesian ARD regression.

    Fit the weights of a regression model, using an ARD prior. The weights of
    the regression model are assumed to be in Gaussian distributions.
    Also estimate the parameters lambda (precisions of the distributions of the
    weights) and alpha (precision of the distribution of the noise).
    The estimation is done by an iterative procedures (Evidence Maximization)

    Read more in the :ref:`User Guide <bayesian_regression>`.

    Parameters
    ----------
    n_iter : int, default=300
        Maximum number of iterations.

    tol : float, default=1e-3
        Stop the algorithm if w has converged.

    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the alpha parameter.

    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the alpha parameter.

    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior
        over the lambda parameter.

    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the
        Gamma distribution prior over the lambda parameter.

    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.

    threshold_lambda : float, default=10 000
        Threshold for removing (pruning) weights with high precision from
        the computation.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution)

    alpha_ : float
       estimated precision of the noise.

    lambda_ : array-like of shape (n_features,)
       estimated precisions of the weights.

    sigma_ : array-like of shape (n_features, n_features)
        estimated variance-covariance matrix of the weights

    scores_ : float
        if computed, value of the objective function (to be maximized)

    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    X_offset_ : float
        If `fit_intercept=True`, offset subtracted for centering data to a
        zero mean. Set to np.zeros(n_features) otherwise.

    X_scale_ : float
        Set to np.ones(n_features).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    BayesianRidge : Bayesian ridge regression.

    Notes
    -----
    For an example, see :ref:`examples/linear_model/plot_ard.py
    <sphx_glr_auto_examples_linear_model_plot_ard.py>`.

    References
    ----------
    D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
    competition, ASHRAE Transactions, 1994.

    R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
    http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
    Their beta is our ``self.alpha_``
    Their alpha is our ``self.lambda_``
    ARD is a little different than the slide: only dimensions/features for
    which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
    discarded.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.ARDRegression()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ARDRegression()
    >>> clf.predict([[1, 1]])
    array([1.])
    """
```
### 6 - sklearn/linear_model/_ridge.py:

Start line: 783, End line: 820

```python
class _BaseRidge(LinearModel, metaclass=ABCMeta):

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "solver": [
            StrOptions(
                {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"}
            )
        ],
        "positive": ["boolean"],
        "random_state": ["random_state"],
    }

    @abstractmethod
    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
```
### 7 - sklearn/linear_model/_bayes.py:

Start line: 360, End line: 381

```python
class BayesianRidge(RegressorMixin, LinearModel):

    def _update_coef_(
        self, X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
    ):
        """Update posterior mean and compute corresponding rmse.

        Posterior mean is given by coef_ = scaled_sigma_ * X.T * y where
        scaled_sigma_ = (lambda_/alpha_ * np.eye(n_features)
                         + np.dot(X.T, X))^-1
        """

        if n_samples > n_features:
            coef_ = np.linalg.multi_dot(
                [Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y]
            )
        else:
            coef_ = np.linalg.multi_dot(
                [X.T, U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T, y]
            )

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_
```
### 8 - examples/linear_model/plot_huber_vs_ridge.py:

Start line: 1, End line: 65

```python
"""
=======================================================
HuberRegressor vs Ridge on dataset with strong outliers
=======================================================

Fit Ridge and HuberRegressor on a dataset with outliers.

The example shows that the predictions in ridge are strongly influenced
by the outliers present in the dataset. The Huber regressor is less
influenced by the outliers since the model uses the linear loss for these.
As the parameter epsilon is increased for the Huber regressor, the decision
function approaches that of the ridge.

"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge

# Generate toy data.
rng = np.random.RandomState(0)
X, y = make_regression(
    n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0
)

# Add four strong outliers to the dataset.
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.0
X_outliers[2:, :] += X.min() - X.mean() / 4.0
y_outliers[:2] += y.min() - y.mean() / 4.0
y_outliers[2:] += y.max() + y.mean() / 4.0
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, "b.")

# Fit the huber regressor over a series of epsilon values.
colors = ["r-", "b-", "y-", "m-"]

x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

# Fit a ridge regressor to compare it to huber regressor.
ridge = Ridge(alpha=0.0, random_state=0)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, "g-", label="ridge regression")

plt.title("Comparison of HuberRegressor vs Ridge")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()
```
### 9 - sklearn/linear_model/_bayes.py:

Start line: 544, End line: 583

```python
class ARDRegression(RegressorMixin, LinearModel):

    _parameter_constraints: dict = {
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "threshold_lambda": [Interval(Real, 0, None, closed="left")],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        n_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        compute_score=False,
        threshold_lambda=1.0e4,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.threshold_lambda = threshold_lambda
        self.copy_X = copy_X
        self.verbose = verbose
```
### 10 - asv_benchmarks/benchmarks/linear_model.py:

Start line: 67, End line: 108

```python
class RidgeBenchmark(Predictor, Estimator, Benchmark):
    """
    Benchmarks for Ridge.
    """

    param_names = ["representation", "solver"]
    params = (
        ["dense", "sparse"],
        ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    )

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, solver = params

        if representation == "dense":
            data = _synth_regression_dataset(n_samples=500000, n_features=100)
        else:
            data = _synth_regression_sparse_dataset(
                n_samples=100000, n_features=10000, density=0.005
            )

        return data

    def make_estimator(self, params):
        representation, solver = params

        estimator = Ridge(solver=solver, fit_intercept=False, random_state=0)

        return estimator

    def make_scorers(self):
        make_gen_reg_scorers(self)

    def skip(self, params):
        representation, solver = params

        if representation == "sparse" and solver == "svd":
            return True
        return False
```
### 13 - sklearn/linear_model/_bayes.py:

Start line: 208, End line: 276

```python
class BayesianRidge(RegressorMixin, LinearModel):

    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        X, y = self._validate_data(X, y, dtype=[np.float64, np.float32], y_numeric=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X,
            y,
            self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y, _ = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1.0 / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.0

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S**2

        # Convergence loop of the bayesian ridge regression
        # ... other code
```
### 15 - sklearn/linear_model/_bayes.py:

Start line: 585, End line: 650

```python
class ARDRegression(RegressorMixin, LinearModel):

    def fit(self, X, y):
        """Fit the model according to the given training data and parameters.

        Iterative procedure to maximize the evidence

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Target values (integers). Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self._validate_params()

        X, y = self._validate_data(
            X, y, dtype=[np.float64, np.float32], y_numeric=True, ensure_min_samples=2
        )

        n_samples, n_features = X.shape
        coef_ = np.zeros(n_features, dtype=X.dtype)

        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X, y, self.fit_intercept, copy=self.copy_X
        )

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_

        # Launch the convergence loop
        keep_lambda = np.ones(n_features, dtype=bool)

        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        verbose = self.verbose

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero
        alpha_ = 1.0 / (np.var(y) + eps)
        lambda_ = np.ones(n_features, dtype=X.dtype)

        self.scores_ = list()
        coef_old_ = None

        def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
            coef_[keep_lambda] = alpha_ * np.linalg.multi_dot(
                [sigma_, X[:, keep_lambda].T, y]
            )
            return coef_

        update_sigma = (
            self._update_sigma
            if n_samples >= n_features
            else self._update_sigma_woodbury
        )
        # Iterative procedure of ARDRegression
        # ... other code
```
### 18 - sklearn/linear_model/_bayes.py:

Start line: 651, End line: 703

```python
class ARDRegression(RegressorMixin, LinearModel):

    def fit(self, X, y):
        # ... other code
        for iter_ in range(self.n_iter):
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)

            # Update alpha and lambda
            rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)
            gamma_ = 1.0 - lambda_[keep_lambda] * np.diag(sigma_)
            lambda_[keep_lambda] = (gamma_ + 2.0 * lambda_1) / (
                (coef_[keep_lambda]) ** 2 + 2.0 * lambda_2
            )
            alpha_ = (n_samples - gamma_.sum() + 2.0 * alpha_1) / (
                rmse_ + 2.0 * alpha_2
            )

            # Prune the weights with a precision over a threshold
            keep_lambda = lambda_ < self.threshold_lambda
            coef_[~keep_lambda] = 0

            # Compute the objective function
            if self.compute_score:
                s = (lambda_1 * np.log(lambda_) - lambda_2 * lambda_).sum()
                s += alpha_1 * log(alpha_) - alpha_2 * alpha_
                s += 0.5 * (
                    fast_logdet(sigma_)
                    + n_samples * log(alpha_)
                    + np.sum(np.log(lambda_))
                )
                s -= 0.5 * (alpha_ * rmse_ + (lambda_ * coef_**2).sum())
                self.scores_.append(s)

            # Check for convergence
            if iter_ > 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Converged after %s iterations" % iter_)
                break
            coef_old_ = np.copy(coef_)

            if not keep_lambda.any():
                break

        if keep_lambda.any():
            # update sigma and mu using updated params from the last iteration
            sigma_ = update_sigma(X, alpha_, lambda_, keep_lambda)
            coef_ = update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_)
        else:
            sigma_ = np.array([]).reshape(0, 0)

        self.coef_ = coef_
        self.alpha_ = alpha_
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self._set_intercept(X_offset_, y_offset_, X_scale_)
        return self
```
### 22 - sklearn/linear_model/_bayes.py:

Start line: 383, End line: 413

```python
class BayesianRidge(RegressorMixin, LinearModel):

    def _log_marginal_likelihood(
        self, n_samples, n_features, eigen_vals, alpha_, lambda_, coef, rmse
    ):
        """Log marginal likelihood."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = -np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_, dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = -np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (
            n_features * log(lambda_)
            + n_samples * log(alpha_)
            - alpha_ * rmse
            - lambda_ * np.sum(coef**2)
            + logdet_sigma
            - n_samples * log(2 * np.pi)
        )

        return score
```
### 42 - sklearn/linear_model/_bayes.py:

Start line: 723, End line: 732

```python
class ARDRegression(RegressorMixin, LinearModel):

    def _update_sigma(self, X, alpha_, lambda_, keep_lambda):
        # See slides as referenced in the docstring note
        # this function is used when n_samples >= n_features and will
        # invert a matrix of shape (n_features, n_features)
        X_keep = X[:, keep_lambda]
        gram = np.dot(X_keep.T, X_keep)
        eye = np.eye(gram.shape[0], dtype=X.dtype)
        sigma_inv = lambda_[keep_lambda] * eye + alpha_ * gram
        sigma_ = pinvh(sigma_inv)
        return sigma_
```
### 43 - sklearn/linear_model/_bayes.py:

Start line: 705, End line: 721

```python
class ARDRegression(RegressorMixin, LinearModel):

    def _update_sigma_woodbury(self, X, alpha_, lambda_, keep_lambda):
        # See slides as referenced in the docstring note
        # this function is used when n_samples < n_features and will invert
        # a matrix of shape (n_samples, n_samples) making use of the
        # woodbury formula:
        # https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        n_samples = X.shape[0]
        X_keep = X[:, keep_lambda]
        inv_lambda = 1 / lambda_[keep_lambda].reshape(1, -1)
        sigma_ = pinvh(
            np.eye(n_samples, dtype=X.dtype) / alpha_
            + np.dot(X_keep * inv_lambda, X_keep.T)
        )
        sigma_ = np.dot(sigma_, X_keep * inv_lambda)
        sigma_ = -np.dot(inv_lambda.reshape(-1, 1) * X_keep.T, sigma_)
        sigma_[np.diag_indices(sigma_.shape[1])] += 1.0 / lambda_[keep_lambda]
        return sigma_
```

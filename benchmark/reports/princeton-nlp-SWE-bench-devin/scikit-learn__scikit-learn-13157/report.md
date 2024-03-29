# scikit-learn__scikit-learn-13157

| **scikit-learn/scikit-learn** | `85440978f517118e78dc15f84e397d50d14c8097` |
| ---- | ---- |
| **No of patches** | 3 |
| **All found context length** | 1973 |
| **Any found context length** | 1973 |
| **Avg pos** | 5.666666666666667 |
| **Min pos** | 5 |
| **Max pos** | 12 |
| **Top file pos** | 2 |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sklearn/base.py b/sklearn/base.py
--- a/sklearn/base.py
+++ b/sklearn/base.py
@@ -359,10 +359,32 @@ def score(self, X, y, sample_weight=None):
         -------
         score : float
             R^2 of self.predict(X) wrt. y.
+
+        Notes
+        -----
+        The R2 score used when calling ``score`` on a regressor will use
+        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
+        with `metrics.r2_score`. This will influence the ``score`` method of
+        all the multioutput regressors (except for
+        `multioutput.MultiOutputRegressor`). To use the new default, please
+        either call `metrics.r2_score` directly or make a custom scorer with
+        `metric.make_scorer`.
         """
 
         from .metrics import r2_score
-        return r2_score(y, self.predict(X), sample_weight=sample_weight,
+        from .metrics.regression import _check_reg_targets
+        y_pred = self.predict(X)
+        # XXX: Remove the check in 0.23
+        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
+        if y_type == 'continuous-multioutput':
+            warnings.warn("The default value of multioutput (not exposed in "
+                          "score method) will change from 'variance_weighted' "
+                          "to 'uniform_average' in 0.23 to keep consistent "
+                          "with 'metrics.r2_score'. To use the new default, "
+                          "please either call 'metrics.r2_score' directly or "
+                          "make a custom scorer with 'metric.make_scorer'.",
+                          FutureWarning)
+        return r2_score(y, y_pred, sample_weight=sample_weight,
                         multioutput='variance_weighted')
 
 
diff --git a/sklearn/linear_model/coordinate_descent.py b/sklearn/linear_model/coordinate_descent.py
--- a/sklearn/linear_model/coordinate_descent.py
+++ b/sklearn/linear_model/coordinate_descent.py
@@ -2247,9 +2247,10 @@ class MultiTaskLassoCV(LinearModelCV, RegressorMixin):
     --------
     >>> from sklearn.linear_model import MultiTaskLassoCV
     >>> from sklearn.datasets import make_regression
+    >>> from sklearn.metrics import r2_score
     >>> X, y = make_regression(n_targets=2, noise=4, random_state=0)
     >>> reg = MultiTaskLassoCV(cv=5, random_state=0).fit(X, y)
-    >>> reg.score(X, y) # doctest: +ELLIPSIS
+    >>> r2_score(y, reg.predict(X)) # doctest: +ELLIPSIS
     0.9994...
     >>> reg.alpha_
     0.5713...
diff --git a/sklearn/multioutput.py b/sklearn/multioutput.py
--- a/sklearn/multioutput.py
+++ b/sklearn/multioutput.py
@@ -256,6 +256,7 @@ def partial_fit(self, X, y, sample_weight=None):
         super().partial_fit(
             X, y, sample_weight=sample_weight)
 
+    # XXX Remove this method in 0.23
     def score(self, X, y, sample_weight=None):
         """Returns the coefficient of determination R^2 of the prediction.
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/base.py | 362 | 362 | 12 | 4 | 5141
| sklearn/linear_model/coordinate_descent.py | 2250 | 2250 | - | - | -
| sklearn/multioutput.py | 259 | 259 | 5 | 2 | 1973


## Problem Statement

```
Different r2_score multioutput default in r2_score and base.RegressorMixin
We've changed multioutput default in r2_score to "uniform_average" in 0.19, but in base.RegressorMixin, we still use ``multioutput='variance_weighted'`` (#5143).
Also see the strange things below:
https://github.com/scikit-learn/scikit-learn/blob/4603e481e9ac67eaf906ae5936263b675ba9bc9c/sklearn/multioutput.py#L283-L286

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/metrics/regression.py | 537 | 584| 476 | 476 | 5294 | 
| 2 | **2 sklearn/multioutput.py** | 1 | 40| 264 | 740 | 10845 | 
| 3 | **2 sklearn/multioutput.py** | 350 | 380| 248 | 988 | 10845 | 
| 4 | 3 examples/ensemble/plot_random_forest_regression_multioutput.py | 1 | 78| 650 | 1638 | 11516 | 
| **-> 5 <-** | **3 sklearn/multioutput.py** | 259 | 294| 335 | 1973 | 11516 | 
| 6 | 3 sklearn/metrics/regression.py | 449 | 536| 785 | 2758 | 11516 | 
| 7 | **3 sklearn/multioutput.py** | 203 | 232| 238 | 2996 | 11516 | 
| 8 | **3 sklearn/multioutput.py** | 172 | 200| 218 | 3214 | 11516 | 
| 9 | **4 sklearn/base.py** | 547 | 611| 373 | 3587 | 15831 | 
| 10 | 5 sklearn/metrics/scorer.py | 466 | 540| 796 | 4383 | 20256 | 
| 11 | 6 sklearn/utils/estimator_checks.py | 2215 | 2256| 399 | 4782 | 42191 | 
| **-> 12 <-** | **6 sklearn/base.py** | 329 | 366| 359 | 5141 | 42191 | 
| 13 | 7 sklearn/linear_model/ransac.py | 327 | 424| 809 | 5950 | 46256 | 
| 14 | **7 sklearn/multioutput.py** | 234 | 257| 178 | 6128 | 46256 | 
| 15 | **7 sklearn/multioutput.py** | 123 | 170| 345 | 6473 | 46256 | 
| 16 | 7 sklearn/metrics/regression.py | 45 | 111| 590 | 7063 | 46256 | 
| 17 | 7 sklearn/linear_model/ransac.py | 425 | 454| 346 | 7409 | 46256 | 
| 18 | 7 sklearn/metrics/regression.py | 356 | 446| 793 | 8202 | 46256 | 
| 19 | **7 sklearn/multioutput.py** | 297 | 323| 212 | 8414 | 46256 | 
| 20 | 7 sklearn/linear_model/ransac.py | 210 | 228| 201 | 8615 | 46256 | 
| 21 | **7 sklearn/multioutput.py** | 383 | 452| 594 | 9209 | 46256 | 
| 22 | 8 sklearn/multiclass.py | 376 | 411| 305 | 9514 | 52695 | 
| 23 | 9 sklearn/ensemble/forest.py | 707 | 1035| 350 | 9864 | 70768 | 
| 24 | 10 sklearn/dummy.py | 515 | 552| 376 | 10240 | 75171 | 
| 25 | 11 examples/multioutput/plot_classifier_chain_yeast.py | 79 | 112| 274 | 10514 | 76215 | 
| 26 | 11 sklearn/multiclass.py | 1 | 63| 484 | 10998 | 76215 | 
| 27 | 12 examples/tree/plot_tree_regression_multioutput.py | 1 | 61| 567 | 11565 | 76782 | 
| 28 | 13 sklearn/feature_selection/rfe.py | 146 | 227| 668 | 12233 | 81404 | 
| 29 | 13 sklearn/ensemble/forest.py | 345 | 359| 151 | 12384 | 81404 | 
| 30 | 13 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 77| 753 | 13137 | 81404 | 
| 31 | 13 sklearn/utils/estimator_checks.py | 1829 | 1871| 469 | 13606 | 81404 | 
| 32 | 14 sklearn/neighbors/base.py | 122 | 160| 368 | 13974 | 89128 | 
| 33 | 15 sklearn/ensemble/bagging.py | 990 | 1020| 248 | 14222 | 97112 | 
| 34 | 15 sklearn/linear_model/ransac.py | 1 | 19| 111 | 14333 | 97112 | 
| 35 | 15 sklearn/utils/estimator_checks.py | 587 | 622| 472 | 14805 | 97112 | 
| 36 | **15 sklearn/multioutput.py** | 62 | 121| 485 | 15290 | 97112 | 
| 37 | **15 sklearn/multioutput.py** | 649 | 733| 643 | 15933 | 97112 | 
| 38 | **15 sklearn/multioutput.py** | 613 | 646| 296 | 16229 | 97112 | 
| 39 | 16 sklearn/kernel_ridge.py | 108 | 129| 186 | 16415 | 98794 | 
| 40 | 17 examples/ensemble/plot_bias_variance.py | 1 | 64| 761 | 17176 | 100608 | 
| 41 | 18 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 17673 | 101637 | 
| 42 | 19 benchmarks/bench_saga.py | 107 | 189| 637 | 18310 | 104103 | 
| 43 | 19 sklearn/utils/estimator_checks.py | 1801 | 1826| 277 | 18587 | 104103 | 
| 44 | 19 sklearn/metrics/scorer.py | 1 | 43| 316 | 18903 | 104103 | 
| 45 | 20 sklearn/metrics/cluster/supervised.py | 728 | 764| 456 | 19359 | 112873 | 
| 46 | **20 sklearn/multioutput.py** | 43 | 59| 123 | 19482 | 112873 | 
| 47 | 20 sklearn/metrics/regression.py | 186 | 252| 613 | 20095 | 112873 | 
| 48 | 21 sklearn/linear_model/logistic.py | 463 | 483| 217 | 20312 | 134915 | 
| 49 | 21 sklearn/utils/estimator_checks.py | 1687 | 1716| 334 | 20646 | 134915 | 
| 50 | 21 sklearn/neighbors/base.py | 162 | 247| 761 | 21407 | 134915 | 
| 51 | 21 sklearn/multiclass.py | 638 | 711| 690 | 22097 | 134915 | 
| 52 | 21 sklearn/metrics/scorer.py | 340 | 397| 532 | 22629 | 134915 | 
| 53 | 21 sklearn/multiclass.py | 762 | 780| 153 | 22782 | 134915 | 
| 54 | 22 sklearn/ensemble/weight_boosting.py | 72 | 90| 143 | 22925 | 144031 | 
| 55 | 23 sklearn/metrics/ranking.py | 321 | 355| 437 | 23362 | 152130 | 
| 56 | 24 sklearn/ensemble/gradient_boosting.py | 1476 | 1549| 672 | 24034 | 173371 | 
| 57 | 24 sklearn/multiclass.py | 110 | 130| 137 | 24171 | 173371 | 
| 58 | 24 sklearn/metrics/scorer.py | 46 | 62| 161 | 24332 | 173371 | 
| 59 | 24 sklearn/ensemble/gradient_boosting.py | 2472 | 2495| 322 | 24654 | 173371 | 
| 60 | 25 sklearn/ensemble/_gb_losses.py | 760 | 774| 156 | 24810 | 180298 | 
| 61 | 26 benchmarks/bench_mnist.py | 84 | 105| 314 | 25124 | 182016 | 
| 62 | 26 sklearn/multiclass.py | 276 | 315| 379 | 25503 | 182016 | 
| 63 | 27 sklearn/linear_model/stochastic_gradient.py | 1295 | 1595| 678 | 26181 | 196505 | 
| 64 | 28 examples/compose/plot_transformed_target.py | 99 | 178| 746 | 26927 | 198319 | 
| 65 | 28 sklearn/linear_model/stochastic_gradient.py | 7 | 43| 346 | 27273 | 198319 | 
| 66 | 28 sklearn/ensemble/weight_boosting.py | 1003 | 1093| 694 | 27967 | 198319 | 


## Missing Patch Files

 * 1: sklearn/base.py
 * 2: sklearn/linear_model/coordinate_descent.py
 * 3: sklearn/multioutput.py

### Hint

```
Should we be deprecating and changing the `multioutput` used in RegressorMixin? How do we allow the user to select the new approach in a deprecation period?
@agramfort @ogrisel can you explain the rational behind this?
It looks to me like the behavior before the PR was exactly what we wanted an the PR broke the deprecation which requires us to do another deprecation cycle?
I vote +1 to deprecat and change the multioutput used in RegressorMixin. It seems misleading that r2_score and RegressorMixin have different defaults.
But yes, the deprecation is not easy. Maybe we can just warn and change the behavior after 2 versions.
And can someone tell me the difference between these two multioutput choices (e..g, why do you prefer uniform_average)? Seems that the deprecation is introduced in https://github.com/scikit-learn/scikit-learn/commit/1b1e10c3251bda9240f115123f6c17c8fde50e35 and I can't find relevant discussions.
```

## Patch

```diff
diff --git a/sklearn/base.py b/sklearn/base.py
--- a/sklearn/base.py
+++ b/sklearn/base.py
@@ -359,10 +359,32 @@ def score(self, X, y, sample_weight=None):
         -------
         score : float
             R^2 of self.predict(X) wrt. y.
+
+        Notes
+        -----
+        The R2 score used when calling ``score`` on a regressor will use
+        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
+        with `metrics.r2_score`. This will influence the ``score`` method of
+        all the multioutput regressors (except for
+        `multioutput.MultiOutputRegressor`). To use the new default, please
+        either call `metrics.r2_score` directly or make a custom scorer with
+        `metric.make_scorer`.
         """
 
         from .metrics import r2_score
-        return r2_score(y, self.predict(X), sample_weight=sample_weight,
+        from .metrics.regression import _check_reg_targets
+        y_pred = self.predict(X)
+        # XXX: Remove the check in 0.23
+        y_type, _, _, _ = _check_reg_targets(y, y_pred, None)
+        if y_type == 'continuous-multioutput':
+            warnings.warn("The default value of multioutput (not exposed in "
+                          "score method) will change from 'variance_weighted' "
+                          "to 'uniform_average' in 0.23 to keep consistent "
+                          "with 'metrics.r2_score'. To use the new default, "
+                          "please either call 'metrics.r2_score' directly or "
+                          "make a custom scorer with 'metric.make_scorer'.",
+                          FutureWarning)
+        return r2_score(y, y_pred, sample_weight=sample_weight,
                         multioutput='variance_weighted')
 
 
diff --git a/sklearn/linear_model/coordinate_descent.py b/sklearn/linear_model/coordinate_descent.py
--- a/sklearn/linear_model/coordinate_descent.py
+++ b/sklearn/linear_model/coordinate_descent.py
@@ -2247,9 +2247,10 @@ class MultiTaskLassoCV(LinearModelCV, RegressorMixin):
     --------
     >>> from sklearn.linear_model import MultiTaskLassoCV
     >>> from sklearn.datasets import make_regression
+    >>> from sklearn.metrics import r2_score
     >>> X, y = make_regression(n_targets=2, noise=4, random_state=0)
     >>> reg = MultiTaskLassoCV(cv=5, random_state=0).fit(X, y)
-    >>> reg.score(X, y) # doctest: +ELLIPSIS
+    >>> r2_score(y, reg.predict(X)) # doctest: +ELLIPSIS
     0.9994...
     >>> reg.alpha_
     0.5713...
diff --git a/sklearn/multioutput.py b/sklearn/multioutput.py
--- a/sklearn/multioutput.py
+++ b/sklearn/multioutput.py
@@ -256,6 +256,7 @@ def partial_fit(self, X, y, sample_weight=None):
         super().partial_fit(
             X, y, sample_weight=sample_weight)
 
+    # XXX Remove this method in 0.23
     def score(self, X, y, sample_weight=None):
         """Returns the coefficient of determination R^2 of the prediction.
 

```

## Test Patch

```diff
diff --git a/sklearn/cross_decomposition/tests/test_pls.py b/sklearn/cross_decomposition/tests/test_pls.py
--- a/sklearn/cross_decomposition/tests/test_pls.py
+++ b/sklearn/cross_decomposition/tests/test_pls.py
@@ -1,3 +1,4 @@
+import pytest
 import numpy as np
 from numpy.testing import assert_approx_equal
 
@@ -377,6 +378,7 @@ def test_pls_errors():
                              clf.fit, X, Y)
 
 
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_pls_scaling():
     # sanity check for scale=True
     n_samples = 1000
diff --git a/sklearn/linear_model/tests/test_coordinate_descent.py b/sklearn/linear_model/tests/test_coordinate_descent.py
--- a/sklearn/linear_model/tests/test_coordinate_descent.py
+++ b/sklearn/linear_model/tests/test_coordinate_descent.py
@@ -232,6 +232,7 @@ def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
 
 
 @pytest.mark.filterwarnings('ignore: The default value of cv')  # 0.22
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_enet_path():
     # We use a large number of samples and of informative features so that
     # the l1_ratio selected is more toward ridge than lasso
diff --git a/sklearn/linear_model/tests/test_ransac.py b/sklearn/linear_model/tests/test_ransac.py
--- a/sklearn/linear_model/tests/test_ransac.py
+++ b/sklearn/linear_model/tests/test_ransac.py
@@ -1,3 +1,4 @@
+import pytest
 import numpy as np
 from scipy import sparse
 
@@ -333,6 +334,7 @@ def test_ransac_min_n_samples():
     assert_raises(ValueError, ransac_estimator7.fit, X, y)
 
 
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_ransac_multi_dimensional_targets():
 
     base_estimator = LinearRegression()
@@ -353,6 +355,7 @@ def test_ransac_multi_dimensional_targets():
     assert_equal(ransac_estimator.inlier_mask_, ref_inlier_mask)
 
 
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_ransac_residual_loss():
     loss_multi1 = lambda y_true, y_pred: np.sum(np.abs(y_true - y_pred), axis=1)
     loss_multi2 = lambda y_true, y_pred: np.sum((y_true - y_pred) ** 2, axis=1)
diff --git a/sklearn/linear_model/tests/test_ridge.py b/sklearn/linear_model/tests/test_ridge.py
--- a/sklearn/linear_model/tests/test_ridge.py
+++ b/sklearn/linear_model/tests/test_ridge.py
@@ -490,6 +490,7 @@ def check_dense_sparse(test_func):
 
 @pytest.mark.filterwarnings('ignore: The default of the `iid`')  # 0.22
 @pytest.mark.filterwarnings('ignore: The default value of cv')  # 0.22
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 @pytest.mark.parametrize(
         'test_func',
         (_test_ridge_loo, _test_ridge_cv, _test_ridge_cv_normalize,
diff --git a/sklearn/model_selection/tests/test_search.py b/sklearn/model_selection/tests/test_search.py
--- a/sklearn/model_selection/tests/test_search.py
+++ b/sklearn/model_selection/tests/test_search.py
@@ -1313,6 +1313,7 @@ def test_pickle():
 
 @pytest.mark.filterwarnings('ignore: The default of the `iid`')  # 0.22
 @pytest.mark.filterwarnings('ignore: The default value of n_split')  # 0.22
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_grid_search_with_multioutput_data():
     # Test search with multi-output estimator
 
diff --git a/sklearn/neural_network/tests/test_mlp.py b/sklearn/neural_network/tests/test_mlp.py
--- a/sklearn/neural_network/tests/test_mlp.py
+++ b/sklearn/neural_network/tests/test_mlp.py
@@ -5,6 +5,7 @@
 # Author: Issam H. Laradji
 # License: BSD 3 clause
 
+import pytest
 import sys
 import warnings
 
@@ -308,6 +309,7 @@ def test_multilabel_classification():
     assert_greater(mlp.score(X, y), 0.9)
 
 
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 def test_multioutput_regression():
     # Test that multi-output regression works as expected
     X, y = make_regression(n_samples=200, n_targets=5)
diff --git a/sklearn/tests/test_base.py b/sklearn/tests/test_base.py
--- a/sklearn/tests/test_base.py
+++ b/sklearn/tests/test_base.py
@@ -486,3 +486,23 @@ def test_tag_inheritance():
     diamond_tag_est = DiamondOverwriteTag()
     with pytest.raises(TypeError, match="Inconsistent values for tag"):
         diamond_tag_est._get_tags()
+
+
+# XXX: Remove in 0.23
+def test_regressormixin_score_multioutput():
+    from sklearn.linear_model import LinearRegression
+    # no warnings when y_type is continuous
+    X = [[1], [2], [3]]
+    y = [1, 2, 3]
+    reg = LinearRegression().fit(X, y)
+    assert_no_warnings(reg.score, X, y)
+    # warn when y_type is continuous-multioutput
+    y = [[1, 2], [2, 3], [3, 4]]
+    reg = LinearRegression().fit(X, y)
+    msg = ("The default value of multioutput (not exposed in "
+           "score method) will change from 'variance_weighted' "
+           "to 'uniform_average' in 0.23 to keep consistent "
+           "with 'metrics.r2_score'. To use the new default, "
+           "please either call 'metrics.r2_score' directly or "
+           "make a custom scorer with 'metric.make_scorer'.")
+    assert_warns_message(FutureWarning, msg, reg.score, X, y)
diff --git a/sklearn/tests/test_dummy.py b/sklearn/tests/test_dummy.py
--- a/sklearn/tests/test_dummy.py
+++ b/sklearn/tests/test_dummy.py
@@ -675,6 +675,7 @@ def test_dummy_regressor_return_std():
     assert_array_equal(y_pred_list[1], y_std_expected)
 
 
+@pytest.mark.filterwarnings('ignore: The default value of multioutput')  # 0.23
 @pytest.mark.parametrize("y,y_test", [
     ([1, 1, 1, 2], [1.25] * 4),
     (np.array([[2, 2],

```


## Code snippets

### 1 - sklearn/metrics/regression.py:

Start line: 537, End line: 584

```python
def r2_score(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

    if _num_samples(y_pred) < 2:
        msg = "R^2 score is not well-defined with less than two samples."
        warnings.warn(msg, UndefinedMetricWarning)
        return float('nan')

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
        weight = sample_weight[:, np.newaxis]
    else:
        weight = 1.

    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[1]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
    if isinstance(multioutput, str):
        if multioutput == 'raw_values':
            # return scores individually
            return output_scores
        elif multioutput == 'uniform_average':
            # passing None as weights results is uniform mean
            avg_weights = None
        elif multioutput == 'variance_weighted':
            avg_weights = denominator
            # avoid fail on constant y or one-element arrays
            if not np.any(nonzero_denominator):
                if not np.any(nonzero_numerator):
                    return 1.0
                else:
                    return 0.0
    else:
        avg_weights = multioutput

    return np.average(output_scores, weights=avg_weights)
```
### 2 - sklearn/multioutput.py:

Start line: 1, End line: 40

```python
"""
This module implements multioutput regression and classification.

The estimators provided in this module are meta-estimators: they require
a base estimator to be provided in their constructor. The meta-estimator
extends single output estimators to multioutput estimators.
"""

import numpy as np
import scipy.sparse as sp
from abc import ABCMeta, abstractmethod
from .base import BaseEstimator, clone, MetaEstimatorMixin
from .base import RegressorMixin, ClassifierMixin, is_classifier
from .model_selection import cross_val_predict
from .utils import check_array, check_X_y, check_random_state
from .utils.fixes import parallel_helper
from .utils.metaestimators import if_delegate_has_method
from .utils.validation import check_is_fitted, has_fit_parameter
from .utils.multiclass import check_classification_targets
from .utils._joblib import Parallel, delayed

__all__ = ["MultiOutputRegressor", "MultiOutputClassifier",
           "ClassifierChain", "RegressorChain"]


def _fit_estimator(estimator, X, y, sample_weight=None):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator
```
### 3 - sklearn/multioutput.py:

Start line: 350, End line: 380

```python
class MultiOutputClassifier(MultiOutputEstimator, ClassifierMixin):

    def score(self, X, y):
        """"Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Test samples

        y : array-like, shape [n_samples, n_outputs]
            True values for X

        Returns
        -------
        scores : float
            accuracy_score of self.predict(X) versus y
        """
        check_is_fitted(self, 'estimators_')
        n_outputs_ = len(self.estimators_)
        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi target classification but has only one")
        if y.shape[1] != n_outputs_:
            raise ValueError("The number of outputs of Y for fit {0} and"
                             " score {1} should be same".
                             format(n_outputs_, y.shape[1]))
        y_pred = self.predict(X)
        return np.mean(np.all(y == y_pred, axis=1))

    def _more_tags(self):
        # FIXME
        return {'_skip_test': True}
```
### 4 - examples/ensemble/plot_random_forest_regression_multioutput.py:

Start line: 1, End line: 78

```python
"""
============================================================
Comparing random forests and the multi-output meta estimator
============================================================

An example to compare multi-output regression with random forest and
the :ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator.

This example illustrates the use of the
:ref:`multioutput.MultiOutputRegressor <multiclass>` meta-estimator
to perform multi-output regression. A random forest regressor is used,
which supports multi-output regression natively, so the results can be
compared.

The random forest regressor will only ever predict values within the
range of observations or closer to zero for each of the targets. As a
result the predictions are biased towards the centre of the circle.

Using a single underlying feature the model learns both the
x and y coordinate as output.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y += (0.5 - rng.rand(*y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=400, test_size=200, random_state=4)

max_depth = 30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
            c="navy", s=s, marker="s", alpha=a, label="Data")
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
            c="cornflowerblue", s=s, alpha=a,
            label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
            c="c", s=s, marker="^", alpha=a,
            label="RF score=%.2f" % regr_rf.score(X_test, y_test))
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()
```
### 5 - sklearn/multioutput.py:

Start line: 259, End line: 294

```python
class MultiOutputRegressor(MultiOutputEstimator, RegressorMixin):

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the regression
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Notes
        -----
        R^2 is calculated by weighting all the targets equally using
        `multioutput='uniform_average'`.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.

        y : array-like, shape (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        # XXX remove in 0.19 when r2_score default for multioutput changes
        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='uniform_average')
```
### 6 - sklearn/metrics/regression.py:

Start line: 449, End line: 536

```python
def r2_score(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average"):
    """R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Read more in the :ref:`User Guide <r2_score>`.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.

    multioutput : string in ['raw_values', 'uniform_average', \
    iance_weighted'] or None or array-like of shape (n_outputs)

        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".

        'raw_values' :
            Returns a full set of scores in case of multioutput input.

        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.

        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

        .. versionchanged:: 0.19
            Default value of multioutput is 'uniform_average'.

    Returns
    -------
    z : float or ndarray of floats
        The R^2 score or ndarray of scores if 'multioutput' is
        'raw_values'.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).

    This metric is not well-defined for single samples and will return a NaN
    value if n_samples is less than two.

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred,
    ...          multioutput='variance_weighted') # doctest: +ELLIPSIS
    0.938...
    >>> y_true = [1, 2, 3]
    >>> y_pred = [1, 2, 3]
    >>> r2_score(y_true, y_pred)
    1.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 2, 2]
    >>> r2_score(y_true, y_pred)
    0.0
    >>> y_true = [1, 2, 3]
    >>> y_pred = [3, 2, 1]
    >>> r2_score(y_true, y_pred)
    -3.0
    """
    # ... other code
```
### 7 - sklearn/multioutput.py:

Start line: 203, End line: 232

```python
class MultiOutputRegressor(MultiOutputEstimator, RegressorMixin):
    """Multi target regression

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for `fit`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        When individual estimators are fast to train or predict
        using `n_jobs>1` can result in slower performance due
        to the overhead of spawning processes.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.
    """

    def __init__(self, estimator, n_jobs=None):
        super().__init__(estimator, n_jobs)
```
### 8 - sklearn/multioutput.py:

Start line: 172, End line: 200

```python
class MultiOutputEstimator(BaseEstimator, MetaEstimatorMixin,
                           metaclass=ABCMeta):

    def predict(self, X):
        """Predict multi-output variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimator, "predict"):
            raise ValueError("The base estimator should implement a predict method")

        X = check_array(X, accept_sparse=True)

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(parallel_helper)(e, 'predict', X)
            for e in self.estimators_)

        return np.asarray(y).T

    def _more_tags(self):
        return {'multioutput_only': True}
```
### 9 - sklearn/base.py:

Start line: 547, End line: 611

```python
class MetaEstimatorMixin:
    _required_parameters = ["estimator"]
    """Mixin class for all meta estimators in scikit-learn."""


class MultiOutputMixin(object):
    """Mixin to mark estimators that support multioutput."""
    def _more_tags(self):
        return {'multioutput': True}


class _UnstableArchMixin(object):
    """Mark estimators that are non-determinstic on 32bit or PowerPC"""
    def _more_tags(self):
        return {'non_deterministic': (
            _IS_32BIT or platform.machine().startswith(('ppc', 'powerpc')))}


def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "regressor"


def is_outlier_detector(estimator):
    """Returns True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "outlier_detector"
```
### 10 - sklearn/metrics/scorer.py:

Start line: 466, End line: 540

```python
# Standard regression scores
explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error,
                               greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)

neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)

# Standard Classification Scores
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)

# Score function for probabilistic classification
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)


# Clustering scores
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)


SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               max_error=max_error_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)


for name, metric in [('precision', precision_score),
                     ('recall', recall_score), ('f1', f1_score)]:
    SCORERS[name] = make_scorer(metric)
    for average in ['macro', 'micro', 'samples', 'weighted']:
        qualified_name = '{0}_{1}'.format(name, average)
        SCORERS[qualified_name] = make_scorer(metric, pos_label=None,
                                              average=average)
```
### 12 - sklearn/base.py:

Start line: 329, End line: 366

```python
class RegressorMixin:
    """Mixin class for all regression estimators in scikit-learn."""
    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """

        from .metrics import r2_score
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')
```
### 14 - sklearn/multioutput.py:

Start line: 234, End line: 257

```python
class MultiOutputRegressor(MultiOutputEstimator, RegressorMixin):

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
        """
        super().partial_fit(
            X, y, sample_weight=sample_weight)
```
### 15 - sklearn/multioutput.py:

Start line: 123, End line: 170

```python
class MultiOutputEstimator(BaseEstimator, MetaEstimatorMixin,
                           metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None):
        """ Fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
        """

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement"
                             "  a fit method")

        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight)
            for i in range(y.shape[1]))
        return self
```
### 19 - sklearn/multioutput.py:

Start line: 297, End line: 323

```python
class MultiOutputClassifier(MultiOutputEstimator, ClassifierMixin):
    """Multi target classification

    This strategy consists of fitting one classifier per target. This is a
    simple strategy for extending classifiers that do not natively support
    multi-target classification

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit`, `score` and `predict_proba`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation.
        It does each target variable in y in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.
    """

    def __init__(self, estimator, n_jobs=None):
        super().__init__(estimator, n_jobs)
```
### 21 - sklearn/multioutput.py:

Start line: 383, End line: 452

```python
class _BaseChain(BaseEstimator, metaclass=ABCMeta):
    def __init__(self, base_estimator, order=None, cv=None, random_state=None):
        self.base_estimator = base_estimator
        self.order = order
        self.cv = cv
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples, n_classes)
            The target values.

        Returns
        -------
        self : object
        """
        X, Y = check_X_y(X, Y, multi_output=True, accept_sparse=True)

        random_state = check_random_state(self.random_state)
        check_array(X, accept_sparse=True)
        self.order_ = self.order
        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == 'random':
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
                raise ValueError("invalid order")

        self.estimators_ = [clone(self.base_estimator)
                            for _ in range(Y.shape[1])]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            if sp.issparse(X):
                X_aug = sp.hstack((X, Y_pred_chain), format='lil')
                X_aug = X_aug.tocsr()
            else:
                X_aug = np.hstack((X, Y_pred_chain))

        elif sp.issparse(X):
            Y_pred_chain = sp.lil_matrix((X.shape[0], Y.shape[1]))
            X_aug = sp.hstack((X, Y_pred_chain), format='lil')

        else:
            Y_pred_chain = np.zeros((X.shape[0], Y.shape[1]))
            X_aug = np.hstack((X, Y_pred_chain))

        del Y_pred_chain

        for chain_idx, estimator in enumerate(self.estimators_):
            y = Y[:, self.order_[chain_idx]]
            estimator.fit(X_aug[:, :(X.shape[1] + chain_idx)], y)
            if self.cv is not None and chain_idx < len(self.estimators_) - 1:
                col_idx = X.shape[1] + chain_idx
                cv_result = cross_val_predict(
                    self.base_estimator, X_aug[:, :col_idx],
                    y=y, cv=self.cv)
                if sp.issparse(X_aug):
                    X_aug[:, col_idx] = np.expand_dims(cv_result, 1)
                else:
                    X_aug[:, col_idx] = cv_result

        return self
```
### 36 - sklearn/multioutput.py:

Start line: 62, End line: 121

```python
class MultiOutputEstimator(BaseEstimator, MetaEstimatorMixin,
                           metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit the model to data.
        Fit a separate model for each output variable.

        Parameters
        ----------
        X : (sparse) array-like, shape (n_samples, n_features)
            Data.

        y : (sparse) array-like, shape (n_samples, n_outputs)
            Multi-output targets.

        classes : list of numpy arrays, shape (n_outputs)
            Each array is unique classes for one output in str/int
            Can be obtained by via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where y is the
            target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape = (n_samples) or None
            Sample weights. If None, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)

        if y.ndim == 1:
            raise ValueError("y must have at least two dimensions for "
                             "multi-output regression but has only one.")

        if (sample_weight is not None and
                not has_fit_parameter(self.estimator, 'sample_weight')):
            raise ValueError("Underlying estimator does not support"
                             " sample weights.")

        first_time = not hasattr(self, 'estimators_')

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X, y[:, i],
                classes[i] if classes is not None else None,
                sample_weight, first_time) for i in range(y.shape[1]))
        return self
```
### 37 - sklearn/multioutput.py:

Start line: 649, End line: 733

```python
class RegressorChain(_BaseChain, RegressorMixin, MetaEstimatorMixin):
    """A multi-label model that arranges regressions into a chain.

    Each model makes a prediction in the order specified by the chain using
    all of the available features provided to the model plus the predictions
    of models that are earlier in the chain.

    Read more in the :ref:`User Guide <regressorchain>`.

    Parameters
    ----------
    base_estimator : estimator
        The base estimator from which the classifier chain is built.

    order : array-like, shape=[n_outputs] or 'random', optional
        By default the order will be determined by the order of columns in
        the label matrix Y.::

            order = [0, 1, 2, ..., Y.shape[1] - 1]

        The order of the chain can be explicitly set by providing a list of
        integers. For example, for a chain of length 5.::

            order = [1, 3, 2, 4, 0]

        means that the first model in the chain will make predictions for
        column 1 in the Y matrix, the second model will make predictions
        for column 3, etc.

        If order is 'random' a random ordering will be used.

    cv : int, cross-validation generator or an iterable, optional \
    (default=None)
        Determines whether to use cross validated predictions or true
        labels for the results of previous estimators in the chain.
        If cv is None the true labels are used when fitting. Otherwise
        possible inputs for cv are:

        - integer, to specify the number of folds in a (Stratified)KFold,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

        The random number generator is used to generate random chain orders.

    Attributes
    ----------
    estimators_ : list
        A list of clones of base_estimator.

    order_ : list
        The order of labels in the classifier chain.

    See also
    --------
    ClassifierChain: Equivalent for classification
    MultioutputRegressor: Learns each output independently rather than
        chaining.

    """
    def fit(self, X, Y):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Y : array-like, shape (n_samples, n_classes)
            The target values.

        Returns
        -------
        self : object
        """
        super().fit(X, Y)
        return self

    def _more_tags(self):
        return {'multioutput_only': True}
```
### 38 - sklearn/multioutput.py:

Start line: 613, End line: 646

```python
class ClassifierChain(_BaseChain, ClassifierMixin, MetaEstimatorMixin):

    @if_delegate_has_method('base_estimator')
    def decision_function(self, X):
        """Evaluate the decision_function of the models in the chain.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        Y_decision : array-like, shape (n_samples, n_classes )
            Returns the decision function of the sample for each model
            in the chain.
        """
        Y_decision_chain = np.zeros((X.shape[0], len(self.estimators_)))
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            if sp.issparse(X):
                X_aug = sp.hstack((X, previous_predictions))
            else:
                X_aug = np.hstack((X, previous_predictions))
            Y_decision_chain[:, chain_idx] = estimator.decision_function(X_aug)
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_decision = Y_decision_chain[:, inv_order]

        return Y_decision

    def _more_tags(self):
        return {'_skip_test': True,
                'multioutput_only': True}
```
### 46 - sklearn/multioutput.py:

Start line: 43, End line: 59

```python
def _partial_fit_estimator(estimator, X, y, classes=None, sample_weight=None,
                           first_time=True):
    if first_time:
        estimator = clone(estimator)

    if sample_weight is not None:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes,
                                  sample_weight=sample_weight)
        else:
            estimator.partial_fit(X, y, sample_weight=sample_weight)
    else:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    return estimator
```

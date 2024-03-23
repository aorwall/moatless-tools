# scikit-learn__scikit-learn-13157

* repo: scikit-learn/scikit-learn
* base_commit: `85440978f517118e78dc15f84e397d50d14c8097`

## Problem statement

Different r2_score multioutput default in r2_score and base.RegressorMixin
We've changed multioutput default in r2_score to "uniform_average" in 0.19, but in base.RegressorMixin, we still use ``multioutput='variance_weighted'`` (#5143).
Also see the strange things below:
https://github.com/scikit-learn/scikit-learn/blob/4603e481e9ac67eaf906ae5936263b675ba9bc9c/sklearn/multioutput.py#L283-L286


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

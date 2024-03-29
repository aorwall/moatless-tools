# scikit-learn__scikit-learn-10581

| **scikit-learn/scikit-learn** | `b27e285ea39450550fc8c81f308a91a660c03a56` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 8062 |
| **Any found context length** | 8062 |
| **Avg pos** | 15.0 |
| **Min pos** | 15 |
| **Max pos** | 15 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/coordinate_descent.py | 703 | 710 | 15 | 1 | 8062


## Problem Statement

```
ElasticNet overwrites X even with copy_X=True
The `fit` function of an `ElasticNet`, called with `check_input=False`, overwrites X, even when `copy_X=True`:
\`\`\`python
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
\`\`\`
ElasticNet overwrites X even with copy_X=True
The `fit` function of an `ElasticNet`, called with `check_input=False`, overwrites X, even when `copy_X=True`:
\`\`\`python
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
\`\`\`
[MRG] FIX #10540 ElasticNet overwrites X even with copy_X=True
Made changes as suggested by @gxyd.
please review and suggest changes @jnothman @gxyd 

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/linear_model/coordinate_descent.py** | 738 | 769| 343 | 343 | 20402 | 
| 2 | 2 sklearn/utils/estimator_checks.py | 1008 | 1076| 601 | 944 | 38321 | 
| 3 | **2 sklearn/linear_model/coordinate_descent.py** | 1385 | 1561| 1689 | 2633 | 38321 | 
| 4 | 3 examples/linear_model/plot_lasso_and_elasticnet.py | 1 | 70| 535 | 3168 | 38856 | 
| 5 | **3 sklearn/linear_model/coordinate_descent.py** | 649 | 666| 182 | 3350 | 38856 | 
| 6 | **3 sklearn/linear_model/coordinate_descent.py** | 383 | 452| 776 | 4126 | 38856 | 
| 7 | 3 sklearn/utils/estimator_checks.py | 571 | 619| 441 | 4567 | 38856 | 
| 8 | 3 sklearn/utils/estimator_checks.py | 1675 | 1710| 352 | 4919 | 38856 | 
| 9 | 4 sklearn/utils/validation.py | 654 | 668| 205 | 5124 | 46119 | 
| 10 | 4 sklearn/utils/estimator_checks.py | 817 | 891| 702 | 5826 | 46119 | 
| 11 | 5 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 606 | 6432 | 46750 | 
| 12 | 5 sklearn/utils/estimator_checks.py | 1782 | 1802| 200 | 6632 | 46750 | 
| 13 | **5 sklearn/linear_model/coordinate_descent.py** | 1719 | 1786| 612 | 7244 | 46750 | 
| 14 | 5 sklearn/utils/estimator_checks.py | 141 | 161| 203 | 7447 | 46750 | 
| **-> 15 <-** | **5 sklearn/linear_model/coordinate_descent.py** | 668 | 736| 615 | 8062 | 46750 | 
| 16 | 6 sklearn/preprocessing/_function_transformer.py | 100 | 119| 126 | 8188 | 48138 | 
| 17 | 7 sklearn/decomposition/incremental_pca.py | 191 | 283| 851 | 9039 | 50687 | 
| 18 | **7 sklearn/linear_model/coordinate_descent.py** | 1587 | 1704| 1136 | 10175 | 50687 | 
| 19 | 7 sklearn/utils/estimator_checks.py | 761 | 776| 139 | 10314 | 50687 | 
| 20 | **7 sklearn/linear_model/coordinate_descent.py** | 268 | 382| 989 | 11303 | 50687 | 
| 21 | **7 sklearn/linear_model/coordinate_descent.py** | 1705 | 1717| 147 | 11450 | 50687 | 
| 22 | **7 sklearn/linear_model/coordinate_descent.py** | 506 | 648| 1385 | 12835 | 50687 | 
| 23 | **7 sklearn/linear_model/coordinate_descent.py** | 454 | 503| 588 | 13423 | 50687 | 
| 24 | **7 sklearn/linear_model/coordinate_descent.py** | 1908 | 2065| 1540 | 14963 | 50687 | 
| 25 | **7 sklearn/linear_model/coordinate_descent.py** | 1562 | 1584| 234 | 15197 | 50687 | 
| 26 | 7 sklearn/utils/estimator_checks.py | 406 | 447| 427 | 15624 | 50687 | 
| 27 | 8 examples/linear_model/plot_lasso_coordinate_descent_path.py | 1 | 78| 695 | 16319 | 51541 | 
| 28 | 9 sklearn/neural_network/rbm.py | 324 | 366| 347 | 16666 | 54371 | 
| 29 | **9 sklearn/linear_model/coordinate_descent.py** | 32 | 123| 845 | 17511 | 54371 | 
| 30 | 10 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 17796 | 55165 | 
| 31 | 11 sklearn/covariance/elliptic_envelope.py | 97 | 148| 402 | 18198 | 56820 | 
| 32 | 12 sklearn/preprocessing/data.py | 2047 | 2059| 115 | 18313 | 82718 | 
| 33 | **12 sklearn/linear_model/coordinate_descent.py** | 2066 | 2085| 213 | 18526 | 82718 | 
| 34 | 12 sklearn/preprocessing/data.py | 1450 | 1480| 278 | 18804 | 82718 | 
| 35 | 12 sklearn/preprocessing/data.py | 1544 | 1565| 229 | 19033 | 82718 | 
| 36 | 12 sklearn/utils/validation.py | 553 | 653| 979 | 20012 | 82718 | 
| 37 | **12 sklearn/linear_model/coordinate_descent.py** | 917 | 928| 155 | 20167 | 82718 | 
| 38 | 12 sklearn/utils/estimator_checks.py | 727 | 758| 297 | 20464 | 82718 | 
| 39 | 13 sklearn/utils/extmath.py | 583 | 614| 209 | 20673 | 88954 | 
| 40 | 13 sklearn/utils/estimator_checks.py | 959 | 982| 269 | 20942 | 88954 | 
| 41 | 13 sklearn/utils/estimator_checks.py | 1414 | 1443| 324 | 21266 | 88954 | 
| 42 | 13 sklearn/utils/estimator_checks.py | 1511 | 1549| 442 | 21708 | 88954 | 
| 43 | 14 sklearn/decomposition/online_lda.py | 454 | 507| 355 | 22063 | 95613 | 
| 44 | 15 sklearn/linear_model/omp.py | 225 | 271| 550 | 22613 | 103454 | 
| 45 | 15 sklearn/utils/estimator_checks.py | 931 | 956| 282 | 22895 | 103454 | 
| 46 | 16 sklearn/neighbors/lof.py | 172 | 222| 513 | 23408 | 106759 | 
| 47 | 16 sklearn/utils/estimator_checks.py | 1873 | 1917| 455 | 23863 | 106759 | 
| 48 | **16 sklearn/linear_model/coordinate_descent.py** | 771 | 793| 170 | 24033 | 106759 | 
| 49 | 16 sklearn/preprocessing/data.py | 2386 | 2403| 192 | 24225 | 106759 | 
| 50 | 17 sklearn/linear_model/base.py | 518 | 572| 599 | 24824 | 111727 | 
| 51 | 17 sklearn/utils/estimator_checks.py | 622 | 644| 221 | 25045 | 111727 | 
| 52 | 18 sklearn/model_selection/_validation.py | 759 | 804| 462 | 25507 | 123459 | 
| 53 | 18 sklearn/utils/estimator_checks.py | 792 | 814| 248 | 25755 | 123459 | 
| 54 | 19 sklearn/linear_model/least_angle.py | 949 | 976| 362 | 26117 | 137266 | 
| 55 | 19 sklearn/utils/estimator_checks.py | 1483 | 1508| 277 | 26394 | 137266 | 
| 56 | 20 examples/neural_networks/plot_rbm_logistic_classification.py | 45 | 76| 271 | 26665 | 138357 | 
| 57 | **20 sklearn/linear_model/coordinate_descent.py** | 1892 | 2085| 161 | 26826 | 138357 | 
| 58 | 20 sklearn/utils/estimator_checks.py | 1765 | 1779| 211 | 27037 | 138357 | 
| 59 | 21 sklearn/neighbors/base.py | 748 | 802| 457 | 27494 | 145037 | 
| 60 | 22 benchmarks/bench_plot_nmf.py | 231 | 278| 530 | 28024 | 148950 | 
| 61 | 22 sklearn/linear_model/omp.py | 98 | 142| 509 | 28533 | 148950 | 
| 62 | 23 examples/linear_model/plot_theilsen.py | 89 | 112| 196 | 28729 | 149954 | 
| 63 | 24 sklearn/naive_bayes.py | 168 | 192| 195 | 28924 | 158363 | 
| 64 | 24 sklearn/preprocessing/_function_transformer.py | 87 | 98| 127 | 29051 | 158363 | 
| 65 | 24 sklearn/utils/estimator_checks.py | 1363 | 1379| 154 | 29205 | 158363 | 
| 66 | 25 sklearn/isotonic.py | 234 | 250| 145 | 29350 | 161741 | 
| 67 | 26 sklearn/manifold/isomap.py | 169 | 185| 118 | 29468 | 163425 | 
| 68 | 26 sklearn/utils/estimator_checks.py | 1119 | 1142| 185 | 29653 | 163425 | 
| 69 | 26 sklearn/preprocessing/data.py | 2405 | 2413| 119 | 29772 | 163425 | 
| 70 | 27 sklearn/ensemble/forest.py | 1902 | 1935| 306 | 30078 | 180364 | 
| 71 | 28 sklearn/manifold/t_sne.py | 632 | 696| 763 | 30841 | 188668 | 
| 72 | 29 sklearn/linear_model/randomized_l1.py | 148 | 182| 314 | 31155 | 194439 | 
| 73 | 29 sklearn/isotonic.py | 6 | 75| 534 | 31689 | 194439 | 
| 74 | 29 sklearn/decomposition/online_lda.py | 586 | 615| 249 | 31938 | 194439 | 
| 75 | 29 sklearn/utils/estimator_checks.py | 647 | 658| 120 | 32058 | 194439 | 


### Hint

```
I think this will be easy to fix. The culprit is this line https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/coordinate_descent.py#L715 which passes `copy=False`, instead of `copy=self.copy_X`. That'll probably fix the issue.

Thanks for reporting it.
Thanks for the diagnosis, @gxyd!
@jnothman, @gxyd: May I work on this?
go ahead

Thanks. On it!!
problem comes from this old PR:

https://github.com/scikit-learn/scikit-learn/pull/5133

basically we should now remove the "check_input=True" fit param from the objects now that we have the context manager to avoid slow checks.

see http://scikit-learn.org/stable/modules/generated/sklearn.config_context.html
I don't think the diagnosis is correct. The problem is that if you pass check_input=True, you bypass the check_input call that is meant to make the copy of X
Thanks for the additional clarifications @agramfort . Could you please  help reviewing [the PR](https://github.com/scikit-learn/scikit-learn/pull/10581)?
I think this will be easy to fix. The culprit is this line https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/coordinate_descent.py#L715 which passes `copy=False`, instead of `copy=self.copy_X`. That'll probably fix the issue.

Thanks for reporting it.
Thanks for the diagnosis, @gxyd!
@jnothman, @gxyd: May I work on this?
go ahead

Thanks. On it!!
problem comes from this old PR:

https://github.com/scikit-learn/scikit-learn/pull/5133

basically we should now remove the "check_input=True" fit param from the objects now that we have the context manager to avoid slow checks.

see http://scikit-learn.org/stable/modules/generated/sklearn.config_context.html
I don't think the diagnosis is correct. The problem is that if you pass check_input=True, you bypass the check_input call that is meant to make the copy of X
Thanks for the additional clarifications @agramfort . Could you please  help reviewing [the PR](https://github.com/scikit-learn/scikit-learn/pull/10581)?
you need to add a non-regression test case that fails without this patch
and shows that we do not modify the data when told to copy.

```

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


## Code snippets

### 1 - sklearn/linear_model/coordinate_descent.py:

Start line: 738, End line: 769

```python
class ElasticNet(LinearModel, RegressorMixin):

    def fit(self, X, y, check_input=True):
        # ... other code

        for k in xrange(n_targets):
            if Xy is not None:
                this_Xy = Xy[:, k]
            else:
                this_Xy = None
            _, this_coef, this_dual_gap, this_iter = \
                self.path(X, y[:, k],
                          l1_ratio=self.l1_ratio, eps=None,
                          n_alphas=None, alphas=[self.alpha],
                          precompute=precompute, Xy=this_Xy,
                          fit_intercept=False, normalize=False, copy_X=True,
                          verbose=False, tol=self.tol, positive=self.positive,
                          X_offset=X_offset, X_scale=X_scale, return_n_iter=True,
                          coef_init=coef_[k], max_iter=self.max_iter,
                          random_state=self.random_state,
                          selection=self.selection,
                          check_input=False)
            coef_[k] = this_coef[:, 0]
            dual_gaps_[k] = this_dual_gap[0]
            self.n_iter_.append(this_iter[0])

        if n_targets == 1:
            self.n_iter_ = self.n_iter_[0]

        self.coef_, self.dual_gap_ = map(np.squeeze, [coef_, dual_gaps_])
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        # return self for chaining fit and predict calls
        return self
```
### 2 - sklearn/utils/estimator_checks.py:

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
### 3 - sklearn/linear_model/coordinate_descent.py:

Start line: 1385, End line: 1561

```python
class ElasticNetCV(LinearModelCV, RegressorMixin):
    """Elastic Net model with iterative fitting along a regularization path

    The best model is selected by cross-validation.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    l1_ratio : float or array of floats, optional
        float between 0 and 1 passed to ElasticNet (scaling between
        l1 and l2 penalties). For ``l1_ratio = 0``
        the penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path, used for each l1_ratio.

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or integer
        Amount of verbosity.

    n_jobs : integer, optional
        Number of CPUs to use during the cross validation. If ``-1``, use
        all the CPUs.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    alpha_ : float
        The amount of penalization chosen by cross validation

    l1_ratio_ : float
        The compromise between l1 and l2 penalization chosen by
        cross validation

    coef_ : array, shape (n_features,) | (n_targets, n_features)
        Parameter vector (w in the cost function formula),

    intercept_ : float | array, shape (n_targets, n_features)
        Independent term in the decision function.

    mse_path_ : array, shape (n_l1_ratio, n_alpha, n_folds)
        Mean square error for the test set on each fold, varying l1_ratio and
        alpha.

    alphas_ : numpy array, shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNetCV
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = ElasticNetCV(cv=5, random_state=0)
    >>> regr.fit(X, y)
    ElasticNetCV(alphas=None, copy_X=True, cv=5, eps=0.001, fit_intercept=True,
           l1_ratio=0.5, max_iter=1000, n_alphas=100, n_jobs=1,
           normalize=False, positive=False, precompute='auto', random_state=0,
           selection='cyclic', tol=0.0001, verbose=0)
    >>> print(regr.alpha_) # doctest: +ELLIPSIS
    0.19947279427
    >>> print(regr.intercept_) # doctest: +ELLIPSIS
    0.398882965428
    >>> print(regr.predict([[0, 0]])) # doctest: +ELLIPSIS
    [ 0.39888297]


    Notes
    -----
    For an example, see
    :ref:`examples/linear_model/plot_lasso_model_selection.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py>`.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    The parameter l1_ratio corresponds to alpha in the glmnet R package
    while alpha corresponds to the lambda parameter in glmnet.
    More specifically, the optimization objective is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

        a * L1 + b * L2

    for::

        alpha = a + b and l1_ratio = a / (a + b).

    See also
    --------
    enet_path
    ElasticNet

    """
```
### 4 - examples/linear_model/plot_lasso_and_elasticnet.py:

Start line: 1, End line: 70

```python
"""
========================================
Lasso and Elastic Net for Sparse Signals
========================================

Estimates Lasso and Elastic-Net regression models on a manually generated
sparse signal corrupted with an additive noise. Estimated coefficients are
compared with the ground-truth.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# #############################################################################
# Generate some sparse data to play with
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# #############################################################################
# Lasso
from sklearn.linear_model import Lasso

alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
```
### 5 - sklearn/linear_model/coordinate_descent.py:

Start line: 649, End line: 666

```python
class ElasticNet(LinearModel, RegressorMixin):
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
```
### 6 - sklearn/linear_model/coordinate_descent.py:

Start line: 383, End line: 452

```python
def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    # We expect X and y to be already Fortran ordered when bypassing
    # checks
    if check_input:
        X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                        order='F', copy=copy_X)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)
        if Xy is not None:
            # Xy should be a 1d contiguous array or a 2D C ordered array
            Xy = check_array(Xy, dtype=X.dtype.type, order='C', copy=False,
                             ensure_2d=False)

    n_samples, n_features = X.shape

    multi_output = False
    if y.ndim != 1:
        multi_output = True
        _, n_outputs = y.shape

    if multi_output and positive:
        raise ValueError('positive=True is not allowed for multi-output'
                         ' (y.ndim != 1)')

    # MultiTaskElasticNet does not support sparse matrices
    if not multi_output and sparse.isspmatrix(X):
        if 'X_offset' in params:
            # As sparse matrices are not actually centered we need this
            # to be passed to the CD solver.
            X_sparse_scaling = params['X_offset'] / params['X_scale']
            X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
        else:
            X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X should be normalized and fit already if function is called
    # from ElasticNet.fit
    if check_input:
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, Xy, precompute, normalize=False,
                     fit_intercept=False, copy=False)
    if alphas is None:
        # No need to normalize of fit_intercept: it has been done
        # above
        alphas = _alpha_grid(X, y, Xy=Xy, l1_ratio=l1_ratio,
                             fit_intercept=False, eps=eps, n_alphas=n_alphas,
                             normalize=False, copy_X=False)
    else:
        alphas = np.sort(alphas)[::-1]  # make sure alphas are properly ordered

    n_alphas = len(alphas)
    tol = params.get('tol', 1e-4)
    max_iter = params.get('max_iter', 1000)
    dual_gaps = np.empty(n_alphas)
    n_iters = []

    rng = check_random_state(params.get('random_state', None))
    selection = params.get('selection', 'cyclic')
    if selection not in ['random', 'cyclic']:
        raise ValueError("selection should be either random or cyclic.")
    random = (selection == 'random')

    if not multi_output:
        coefs = np.empty((n_features, n_alphas), dtype=X.dtype)
    else:
        coefs = np.empty((n_outputs, n_features, n_alphas),
                         dtype=X.dtype)

    if coef_init is None:
        coef_ = np.asfortranarray(np.zeros(coefs.shape[:-1], dtype=X.dtype))
    else:
        coef_ = np.asfortranarray(coef_init, dtype=X.dtype)
    # ... other code
```
### 7 - sklearn/utils/estimator_checks.py:

Start line: 571, End line: 619

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [key for key in dict_after_fit.keys()
                             if is_public_parameter(key)]

    attrs_added_by_fit = [key for key in public_keys_after_fit
                          if key not in dict_before_fit.keys()]

    # check that fit doesn't add any public attribute
    assert_true(not attrs_added_by_fit,
                ('Estimator adds public attribute(s) during'
                 ' the fit method.'
                 ' Estimators are only allowed to add private attributes'
                 ' either started with _ or ended'
                 ' with _ but %s added' % ', '.join(attrs_added_by_fit)))

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [key for key in public_keys_after_fit
                            if (dict_before_fit[key]
                                is not dict_after_fit[key])]

    assert_true(not attrs_changed_by_fit,
                ('Estimator changes public attribute(s) during'
                 ' the fit method. Estimators are only allowed'
                 ' to change attributes started'
                 ' or ended with _, but'
                 ' %s changed' % ', '.join(attrs_changed_by_fit)))
```
### 8 - sklearn/utils/estimator_checks.py:

Start line: 1675, End line: 1710

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_overwrite_params(name, estimator_orig):
    X, y = make_blobs(random_state=0, n_samples=9)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert_equal(hash(new_value), hash(original_value),
                     "Estimator %s should not change or mutate "
                     " the parameter %s from %s to %s during fit."
                     % (name, param_name, original_value, new_value))
```
### 9 - sklearn/utils/validation.py:

Start line: 654, End line: 668

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
### 10 - sklearn/utils/estimator_checks.py:

Start line: 817, End line: 891

```python
def _check_transformer(name, transformer_orig, X, y):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # on numpy & scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    if name in CROSS_DECOMPOSITION:
        y_ = np.c_[y, y]
        y_[::2, 1] *= 2
    else:
        y_ = y

    transformer.fit(X, y_)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y_)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert_equal(x_pred.shape[0], n_samples)
    else:
        # check for consistent n_samples
        assert_equal(X_pred.shape[0], n_samples)

    if hasattr(transformer, 'transform'):
        if name in CROSS_DECOMPOSITION:
            X_pred2 = transformer.transform(X, y_)
            X_pred3 = transformer.fit_transform(X, y=y_)
        else:
            X_pred2 = transformer.transform(X)
            X_pred3 = transformer.fit_transform(X, y=y_)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred, x_pred2, atol=1e-2,
                    err_msg="fit_transform and transform outcomes "
                            "not consistent in %s"
                    % transformer)
                assert_allclose_dense_sparse(
                    x_pred, x_pred3, atol=1e-2,
                    err_msg="consecutive fit_transform outcomes "
                            "not consistent in %s"
                    % transformer)
        else:
            assert_allclose_dense_sparse(
                X_pred, X_pred2,
                err_msg="fit_transform and transform outcomes "
                        "not consistent in %s"
                % transformer, atol=1e-2)
            assert_allclose_dense_sparse(
                X_pred, X_pred3, atol=1e-2,
                err_msg="consecutive fit_transform outcomes "
                        "not consistent in %s"
                % transformer)
            assert_equal(_num_samples(X_pred2), n_samples)
            assert_equal(_num_samples(X_pred3), n_samples)

        # raises error on malformed input for transform
        if hasattr(X, 'T'):
            # If it's not an array, it does not have a 'T' property
            with assert_raises(ValueError, msg="The transformer {} does "
                               "not raise an error when the number of "
                               "features in transform is different from"
                               " the number of features in "
                               "fit.".format(name)):
                transformer.transform(X.T)
```
### 13 - sklearn/linear_model/coordinate_descent.py:

Start line: 1719, End line: 1786

```python
class MultiTaskElasticNet(Lasso):

    def fit(self, X, y):
        """Fit MultiTaskElasticNet model with coordinate descent

        Parameters
        -----------
        X : ndarray, shape (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples, n_tasks)
            Target. Will be cast to X's dtype if necessary

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        X = check_array(X, dtype=[np.float64, np.float32], order='F',
                        copy=self.copy_X and self.fit_intercept)
        y = check_array(y, dtype=X.dtype.type, ensure_2d=False)

        if hasattr(self, 'l1_ratio'):
            model_str = 'ElasticNet'
        else:
            model_str = 'Lasso'
        if y.ndim == 1:
            raise ValueError("For mono-task outputs, use %s" % model_str)

        n_samples, n_features = X.shape
        _, n_tasks = y.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, self.fit_intercept, self.normalize, copy=False)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_tasks, n_features), dtype=X.dtype.type,
                                  order='F')

        l1_reg = self.alpha * self.l1_ratio * n_samples
        l2_reg = self.alpha * (1.0 - self.l1_ratio) * n_samples

        self.coef_ = np.asfortranarray(self.coef_)  # coef contiguous in memory

        if self.selection not in ['random', 'cyclic']:
            raise ValueError("selection should be either random or cyclic.")
        random = (self.selection == 'random')

        self.coef_, self.dual_gap_, self.eps_, self.n_iter_ = \
            cd_fast.enet_coordinate_descent_multi_task(
                self.coef_, l1_reg, l2_reg, X, y, self.max_iter, self.tol,
                check_random_state(self.random_state), random)

        self._set_intercept(X_offset, y_offset, X_scale)

        if self.dual_gap_ > self.eps_:
            warnings.warn('Objective did not converge, you might want'
                          ' to increase the number of iterations',
                          ConvergenceWarning)

        # return self for chaining fit and predict calls
        return self
```
### 15 - sklearn/linear_model/coordinate_descent.py:

Start line: 668, End line: 736

```python
class ElasticNet(LinearModel, RegressorMixin):

    def fit(self, X, y, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Notes
        -----

        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator", stacklevel=2)

        if isinstance(self.precompute, six.string_types):
            raise ValueError('precompute should be one of True, False or'
                             ' array-like. Got %r' % self.precompute)

        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X, y = check_X_y(X, y, accept_sparse='csc',
                             order='F', dtype=[np.float64, np.float32],
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=True, y_numeric=True)
            y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=False)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        if self.selection not in ['cyclic', 'random']:
            raise ValueError("selection should be either random or cyclic.")

        if not self.warm_start or not hasattr(self, "coef_"):
            coef_ = np.zeros((n_targets, n_features), dtype=X.dtype,
                             order='F')
        else:
            coef_ = self.coef_
            if coef_.ndim == 1:
                coef_ = coef_[np.newaxis, :]

        dual_gaps_ = np.zeros(n_targets, dtype=X.dtype)
        self.n_iter_ = []
        # ... other code
```
### 18 - sklearn/linear_model/coordinate_descent.py:

Start line: 1587, End line: 1704

```python
###############################################################################
# Multi Task ElasticNet and Lasso models (with joint feature selection)


class MultiTaskElasticNet(Lasso):
    r"""Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1/L2 term. Defaults to 1.0

    l1_ratio : float
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
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
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    intercept_ : array, shape (n_tasks,)
        Independent term in decision function.

    coef_ : array, shape (n_tasks, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is \
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
    ... #doctest: +NORMALIZE_WHITESPACE
    MultiTaskElasticNet(alpha=0.1, copy_X=True, fit_intercept=True,
            l1_ratio=0.5, max_iter=1000, normalize=False, random_state=None,
            selection='cyclic', tol=0.0001, warm_start=False)
    >>> print(clf.coef_)
    [[ 0.45663524  0.45612256]
     [ 0.45663524  0.45612256]]
    >>> print(clf.intercept_)
    [ 0.0872422  0.0872422]

    See also
    --------
    MultiTaskElasticNet : Multi-task L1/L2 ElasticNet with built-in
        cross-validation.
    ElasticNet
    MultiTaskLasso

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    """
```
### 20 - sklearn/linear_model/coordinate_descent.py:

Start line: 268, End line: 382

```python
def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    r"""Compute elastic net path with coordinate descent

    The elastic net optimization function varies for mono and multi-outputs.

    For mono-output tasks it is::

        1 / (2 * n_samples) * ||y - Xw||^2_2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    For multi-output tasks it is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values

    l1_ratio : float, optional
        float between 0 and 1 passed to elastic net (scaling between
        l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso

    eps : float
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If None alphas are set automatically

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : array, shape (n_features, ) | None
        The initial values of the coefficients.

    verbose : bool or integer
        Amount of verbosity.

    return_n_iter : bool
        whether to return the number of iterations or not.

    positive : bool, default False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    check_input : bool, default True
        Skip input validation checks, including the Gram matrix when provided
        assuming there are handled by the caller when check_input=False.

    **params : kwargs
        keyword arguments passed to the coordinate descent solver.

    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas) or \
            (n_outputs, n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : array-like, shape (n_alphas,)
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.
        (Is returned when ``return_n_iter`` is set to True).

    Notes
    -----
    For an example, see
    :ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
    <sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.

    See also
    --------
    MultiTaskElasticNet
    MultiTaskElasticNetCV
    ElasticNet
    ElasticNetCV
    """
    # ... other code
```
### 21 - sklearn/linear_model/coordinate_descent.py:

Start line: 1705, End line: 1717

```python
class MultiTaskElasticNet(Lasso):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, copy_X=True, max_iter=1000, tol=1e-4,
                 warm_start=False, random_state=None, selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
```
### 22 - sklearn/linear_model/coordinate_descent.py:

Start line: 506, End line: 648

```python
###############################################################################
# ElasticNet model


class ElasticNet(LinearModel, RegressorMixin):
    """Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * L1 + b * L2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.

    Read more in the :ref:`User Guide <elastic_net>`.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``True`` to preserve sparsity.

    max_iter : int, optional
        The maximum number of iterations

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> from sklearn.datasets import make_regression
    >>>
    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> regr = ElasticNet(random_state=0)
    >>> regr.fit(X, y)
    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=0, selection='cyclic', tol=0.0001, warm_start=False)
    >>> print(regr.coef_) # doctest: +ELLIPSIS
    [ 18.83816048  64.55968825]
    >>> print(regr.intercept_) # doctest: +ELLIPSIS
    1.45126075617
    >>> print(regr.predict([[0, 0]])) # doctest: +ELLIPSIS
    [ 1.45126076]


    Notes
    -----
    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    See also
    --------
    ElasticNetCV : Elastic net model with best model selection by
        cross-validation.
    SGDRegressor: implements elastic net regression with incremental training.
    SGDClassifier: implements logistic regression with elastic net penalty
        (``SGDClassifier(loss="log", penalty="elasticnet")``).
    """
```
### 23 - sklearn/linear_model/coordinate_descent.py:

Start line: 454, End line: 503

```python
def enet_path(X, y, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
              precompute='auto', Xy=None, copy_X=True, coef_init=None,
              verbose=False, return_n_iter=False, positive=False,
              check_input=True, **params):
    # ... other code

    for i, alpha in enumerate(alphas):
        l1_reg = alpha * l1_ratio * n_samples
        l2_reg = alpha * (1.0 - l1_ratio) * n_samples
        if not multi_output and sparse.isspmatrix(X):
            model = cd_fast.sparse_enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X.data, X.indices,
                X.indptr, y, X_sparse_scaling,
                max_iter, tol, rng, random, positive)
        elif multi_output:
            model = cd_fast.enet_coordinate_descent_multi_task(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random)
        elif isinstance(precompute, np.ndarray):
            # We expect precompute to be already Fortran ordered when bypassing
            # checks
            if check_input:
                precompute = check_array(precompute, dtype=X.dtype.type,
                                         order='C')
            model = cd_fast.enet_coordinate_descent_gram(
                coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter,
                tol, rng, random, positive)
        elif precompute is False:
            model = cd_fast.enet_coordinate_descent(
                coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random,
                positive)
        else:
            raise ValueError("Precompute should be one of True, False, "
                             "'auto' or array-like. Got %r" % precompute)
        coef_, dual_gap_, eps_, n_iter_ = model
        coefs[..., i] = coef_
        dual_gaps[i] = dual_gap_
        n_iters.append(n_iter_)
        if dual_gap_ > eps_:
            warnings.warn('Objective did not converge.' +
                          ' You might want' +
                          ' to increase the number of iterations.' +
                          ' Fitting data with very small alpha' +
                          ' may cause precision problems.',
                          ConvergenceWarning)

        if verbose:
            if verbose > 2:
                print(model)
            elif verbose > 1:
                print('Path: %03i out of %03i' % (i, n_alphas))
            else:
                sys.stderr.write('.')

    if return_n_iter:
        return alphas, coefs, dual_gaps, n_iters
    return alphas, coefs, dual_gaps
```
### 24 - sklearn/linear_model/coordinate_descent.py:

Start line: 1908, End line: 2065

```python
class MultiTaskElasticNetCV(LinearModelCV, RegressorMixin):
    r"""Multi-task L1/L2 ElasticNet with built-in cross-validation.

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    Parameters
    ----------
    l1_ratio : float or array of floats
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : array-like, optional
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or integer
        Amount of verbosity.

    n_jobs : integer, optional
        Number of CPUs to use during the cross validation. If ``-1``, use
        all the CPUs. Note that this is used only if multiple values for
        l1_ratio are given.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    intercept_ : array, shape (n_tasks,)
        Independent term in decision function.

    coef_ : array, shape (n_tasks, n_features)
        Parameter vector (W in the cost function formula).
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    alpha_ : float
        The amount of penalization chosen by cross validation

    mse_path_ : array, shape (n_alphas, n_folds) or \
                (n_l1_ratio, n_alphas, n_folds)
        mean square error for the test set on each fold, varying alpha

    alphas_ : numpy array, shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio

    l1_ratio_ : float
        best l1_ratio obtained by cross-validation.

    n_iter_ : int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNetCV()
    >>> clf.fit([[0,0], [1, 1], [2, 2]],
    ...         [[0, 0], [1, 1], [2, 2]])
    ... #doctest: +NORMALIZE_WHITESPACE
    MultiTaskElasticNetCV(alphas=None, copy_X=True, cv=None, eps=0.001,
           fit_intercept=True, l1_ratio=0.5, max_iter=1000, n_alphas=100,
           n_jobs=1, normalize=False, random_state=None, selection='cyclic',
           tol=0.0001, verbose=0)
    >>> print(clf.coef_)
    [[ 0.52875032  0.46958558]
     [ 0.52875032  0.46958558]]
    >>> print(clf.intercept_)
    [ 0.00166409  0.00166409]

    See also
    --------
    MultiTaskElasticNet
    ElasticNetCV
    MultiTaskLassoCV

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    """
```
### 25 - sklearn/linear_model/coordinate_descent.py:

Start line: 1562, End line: 1584

```python
class ElasticNetCV(LinearModelCV, RegressorMixin):
    path = staticmethod(enet_path)

    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False, precompute='auto',
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, positive=False, random_state=None,
                 selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.positive = positive
        self.random_state = random_state
        self.selection = selection
```
### 29 - sklearn/linear_model/coordinate_descent.py:

Start line: 32, End line: 123

```python
###############################################################################
# Paths functions

def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """ Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray, shape (n_samples,)
        Target values

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    fit_intercept : boolean, default True
        Whether to fit an intercept or not

    normalize : boolean, optional, default False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, 'csc',
                        copy=(copy_X and fit_intercept and not X_sparse))
        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                      normalize,
                                                      return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)[::-1]
```
### 33 - sklearn/linear_model/coordinate_descent.py:

Start line: 2066, End line: 2085

```python
class MultiTaskElasticNetCV(LinearModelCV, RegressorMixin):
    path = staticmethod(enet_path)

    def __init__(self, l1_ratio=0.5, eps=1e-3, n_alphas=100, alphas=None,
                 fit_intercept=True, normalize=False,
                 max_iter=1000, tol=1e-4, cv=None, copy_X=True,
                 verbose=0, n_jobs=1, random_state=None, selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selection = selection
```
### 37 - sklearn/linear_model/coordinate_descent.py:

Start line: 917, End line: 928

```python
class Lasso(ElasticNet):
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(Lasso, self).__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)
```
### 48 - sklearn/linear_model/coordinate_descent.py:

Start line: 771, End line: 793

```python
class ElasticNet(LinearModel, RegressorMixin):

    @property
    def sparse_coef_(self):
        """ sparse representation of the fitted ``coef_`` """
        return sparse.csr_matrix(self.coef_)

    def _decision_function(self, X):
        """Decision function of the linear model

        Parameters
        ----------
        X : numpy array or scipy.sparse matrix of shape (n_samples, n_features)

        Returns
        -------
        T : array, shape (n_samples,)
            The predicted decision function
        """
        check_is_fitted(self, 'n_iter_')
        if sparse.isspmatrix(X):
            return safe_sparse_dot(X, self.coef_.T,
                                   dense_output=True) + self.intercept_
        else:
            return super(ElasticNet, self)._decision_function(X)
```
### 57 - sklearn/linear_model/coordinate_descent.py:

Start line: 1892, End line: 2085

```python
class MultiTaskLasso(MultiTaskElasticNet):
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=1000, tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.l1_ratio = 1.0
        self.random_state = random_state
        self.selection = selection


class MultiTaskElasticNetCV(LinearModelCV, RegressorMixin):
```

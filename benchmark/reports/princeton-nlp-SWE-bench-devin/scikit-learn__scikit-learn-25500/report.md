# scikit-learn__scikit-learn-25500

| **scikit-learn/scikit-learn** | `4db04923a754b6a2defa1b172f55d492b85d165e` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | - |
| **Missing snippets** | 3 |
| **Missing patch files** | 1 |


## Expected patch

```diff
diff --git a/sklearn/isotonic.py b/sklearn/isotonic.py
--- a/sklearn/isotonic.py
+++ b/sklearn/isotonic.py
@@ -360,23 +360,16 @@ def fit(self, X, y, sample_weight=None):
         self._build_f(X, y)
         return self
 
-    def transform(self, T):
-        """Transform new data by linear interpolation.
-
-        Parameters
-        ----------
-        T : array-like of shape (n_samples,) or (n_samples, 1)
-            Data to transform.
+    def _transform(self, T):
+        """`_transform` is called by both `transform` and `predict` methods.
 
-            .. versionchanged:: 0.24
-               Also accepts 2d array with 1 feature.
+        Since `transform` is wrapped to output arrays of specific types (e.g.
+        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`
+        directly.
 
-        Returns
-        -------
-        y_pred : ndarray of shape (n_samples,)
-            The transformed data.
+        The above behaviour could be changed in the future, if we decide to output
+        other type of arrays when calling `predict`.
         """
-
         if hasattr(self, "X_thresholds_"):
             dtype = self.X_thresholds_.dtype
         else:
@@ -397,6 +390,24 @@ def transform(self, T):
 
         return res
 
+    def transform(self, T):
+        """Transform new data by linear interpolation.
+
+        Parameters
+        ----------
+        T : array-like of shape (n_samples,) or (n_samples, 1)
+            Data to transform.
+
+            .. versionchanged:: 0.24
+               Also accepts 2d array with 1 feature.
+
+        Returns
+        -------
+        y_pred : ndarray of shape (n_samples,)
+            The transformed data.
+        """
+        return self._transform(T)
+
     def predict(self, T):
         """Predict new data by linear interpolation.
 
@@ -410,7 +421,7 @@ def predict(self, T):
         y_pred : ndarray of shape (n_samples,)
             Transformed data.
         """
-        return self.transform(T)
+        return self._transform(T)
 
     # We implement get_feature_names_out here instead of using
     # `ClassNamePrefixFeaturesOutMixin`` because `input_features` are ignored.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/isotonic.py | 363 | 379 | - | - | -
| sklearn/isotonic.py | 400 | 400 | - | - | -
| sklearn/isotonic.py | 413 | 413 | - | - | -


## Problem Statement

```
CalibratedClassifierCV doesn't work with `set_config(transform_output="pandas")`
### Describe the bug

CalibratedClassifierCV with isotonic regression doesn't work when we previously set `set_config(transform_output="pandas")`.
The IsotonicRegression seems to return a dataframe, which is a problem for `_CalibratedClassifier`  in `predict_proba` where it tries to put the dataframe in a numpy array row `proba[:, class_idx] = calibrator.predict(this_pred)`.

### Steps/Code to Reproduce

\`\`\`python
import numpy as np
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

set_config(transform_output="pandas")
model = CalibratedClassifierCV(SGDClassifier(), method='isotonic')
model.fit(np.arange(90).reshape(30, -1), np.arange(30) % 2)
model.predict(np.arange(90).reshape(30, -1))
\`\`\`

### Expected Results

It should not crash.

### Actual Results

\`\`\`
../core/model_trainer.py:306: in train_model
    cv_predictions = cross_val_predict(pipeline,
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/model_selection/_validation.py:968: in cross_val_predict
    predictions = parallel(
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/parallel.py:1085: in __call__
    if self.dispatch_one_batch(iterator):
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/parallel.py:901: in dispatch_one_batch
    self._dispatch(tasks)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/parallel.py:819: in _dispatch
    job = self._backend.apply_async(batch, callback=cb)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/_parallel_backends.py:208: in apply_async
    result = ImmediateResult(func)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/_parallel_backends.py:597: in __init__
    self.results = batch()
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/parallel.py:288: in __call__
    return [func(*args, **kwargs)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/joblib/parallel.py:288: in <listcomp>
    return [func(*args, **kwargs)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/utils/fixes.py:117: in __call__
    return self.function(*args, **kwargs)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/model_selection/_validation.py:1052: in _fit_and_predict
    predictions = func(X_test)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/pipeline.py:548: in predict_proba
    return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/calibration.py:477: in predict_proba
    proba = calibrated_classifier.predict_proba(X)
../../../../.anaconda3/envs/strategy-training/lib/python3.9/site-packages/sklearn/calibration.py:764: in predict_proba
    proba[:, class_idx] = calibrator.predict(this_pred)
E   ValueError: could not broadcast input array from shape (20,1) into shape (20,)
\`\`\`

### Versions

\`\`\`shell
System:
    python: 3.9.15 (main, Nov 24 2022, 14:31:59)  [GCC 11.2.0]
executable: /home/philippe/.anaconda3/envs/strategy-training/bin/python
   machine: Linux-5.15.0-57-generic-x86_64-with-glibc2.31

Python dependencies:
      sklearn: 1.2.0
          pip: 22.2.2
   setuptools: 62.3.2
        numpy: 1.23.5
        scipy: 1.9.3
       Cython: None
       pandas: 1.4.1
   matplotlib: 3.6.3
       joblib: 1.2.0
threadpoolctl: 3.1.0

Built with OpenMP: True

threadpoolctl info:
       user_api: openmp
   internal_api: openmp
         prefix: libgomp
       filepath: /home/philippe/.anaconda3/envs/strategy-training/lib/python3.9/site-packages/scikit_learn.libs/libgomp-a34b3233.so.1.0.0
        version: None
    num_threads: 12

       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: /home/philippe/.anaconda3/envs/strategy-training/lib/python3.9/site-packages/numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so
        version: 0.3.20
threading_layer: pthreads
   architecture: Haswell
    num_threads: 12

       user_api: blas
   internal_api: openblas
         prefix: libopenblas
       filepath: /home/philippe/.anaconda3/envs/strategy-training/lib/python3.9/site-packages/scipy.libs/libopenblasp-r0-41284840.3.18.so
        version: 0.3.18
threading_layer: pthreads
   architecture: Haswell
    num_threads: 12
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/calibration.py | 56 | 244| 1916 | 1916 | 10987 | 
| 2 | 1 sklearn/calibration.py | 336 | 453| 932 | 2848 | 10987 | 
| 3 | 1 sklearn/calibration.py | 246 | 279| 227 | 3075 | 10987 | 
| 4 | 2 examples/calibration/plot_calibration_curve.py | 103 | 197| 853 | 3928 | 13965 | 
| 5 | 3 examples/calibration/plot_calibration.py | 1 | 88| 768 | 4696 | 15229 | 
| 6 | 4 examples/calibration/plot_calibration_multiclass.py | 229 | 275| 545 | 5241 | 18092 | 
| 7 | 4 sklearn/calibration.py | 1 | 511| 281 | 5522 | 18092 | 
| 8 | 4 examples/calibration/plot_calibration.py | 89 | 145| 412 | 5934 | 18092 | 
| 9 | 5 examples/calibration/plot_compare_calibration.py | 85 | 211| 1338 | 7272 | 20129 | 
| 10 | 5 examples/calibration/plot_calibration_curve.py | 1 | 102| 702 | 7974 | 20129 | 
| 11 | 5 sklearn/calibration.py | 993 | 1042| 557 | 8531 | 20129 | 
| 12 | 5 examples/calibration/plot_calibration_curve.py | 297 | 337| 414 | 8945 | 20129 | 
| 13 | 5 examples/calibration/plot_calibration_multiclass.py | 171 | 228| 771 | 9716 | 20129 | 
| 14 | 5 examples/calibration/plot_calibration_multiclass.py | 1 | 96| 774 | 10490 | 20129 | 
| 15 | 6 examples/release_highlights/plot_release_highlights_1_0_0.py | 167 | 242| 729 | 11219 | 22464 | 
| 16 | 6 examples/calibration/plot_calibration_curve.py | 222 | 296| 659 | 11878 | 22464 | 
| 17 | 7 examples/miscellaneous/plot_set_output.py | 88 | 139| 364 | 12242 | 23529 | 
| 18 | 8 sklearn/linear_model/_stochastic_gradient.py | 9 | 56| 398 | 12640 | 43118 | 
| 19 | 9 examples/release_highlights/plot_release_highlights_0_23_0.py | 1 | 97| 857 | 13497 | 44900 | 
| 20 | 9 sklearn/calibration.py | 281 | 335| 405 | 13902 | 44900 | 
| 21 | 10 sklearn/utils/estimator_checks.py | 2554 | 3236| 5959 | 19861 | 81215 | 
| 22 | 11 examples/compose/plot_column_transformer_mixed_types.py | 124 | 215| 758 | 20619 | 83012 | 
| 23 | 12 examples/linear_model/plot_poisson_regression_non_normal_loss.py | 101 | 161| 441 | 21060 | 88345 | 
| 24 | 12 sklearn/calibration.py | 655 | 700| 372 | 21432 | 88345 | 
| 25 | 12 sklearn/utils/estimator_checks.py | 3239 | 4012| 6276 | 27708 | 88345 | 
| 26 | 12 sklearn/utils/estimator_checks.py | 1807 | 2553| 6231 | 33939 | 88345 | 
| 27 | 13 examples/release_highlights/plot_release_highlights_0_22_0.py | 195 | 281| 764 | 34703 | 90755 | 
| 28 | 14 sklearn/linear_model/_coordinate_descent.py | 8 | 36| 198 | 34901 | 115770 | 
| 29 | 15 examples/release_highlights/plot_release_highlights_0_24_0.py | 121 | 210| 777 | 35678 | 118208 | 
| 30 | 16 examples/ensemble/plot_gradient_boosting_categorical.py | 106 | 178| 592 | 36270 | 120495 | 
| 31 | 16 sklearn/utils/estimator_checks.py | 904 | 1695| 6467 | 42737 | 120495 | 
| 32 | 16 examples/release_highlights/plot_release_highlights_0_24_0.py | 1 | 120| 1150 | 43887 | 120495 | 
| 33 | 16 examples/calibration/plot_compare_calibration.py | 1 | 62| 387 | 44274 | 120495 | 
| 34 | 17 setup.py | 70 | 128| 643 | 44917 | 126462 | 
| 35 | 17 sklearn/calibration.py | 1295 | 1313| 185 | 45102 | 126462 | 
| 36 | 18 examples/release_highlights/plot_release_highlights_1_1_0.py | 1 | 96| 775 | 45877 | 128599 | 
| 37 | 19 examples/release_highlights/plot_release_highlights_1_2_0.py | 92 | 167| 718 | 46595 | 130084 | 
| 38 | 19 examples/release_highlights/plot_release_highlights_1_2_0.py | 1 | 90| 766 | 47361 | 130084 | 
| 39 | 19 examples/release_highlights/plot_release_highlights_0_22_0.py | 1 | 89| 749 | 48110 | 130084 | 
| 40 | 19 examples/miscellaneous/plot_set_output.py | 1 | 87| 701 | 48811 | 130084 | 
| 41 | 20 benchmarks/bench_20newsgroups.py | 1 | 95| 754 | 49565 | 130838 | 
| 42 | 21 examples/ensemble/plot_gradient_boosting_quantile.py | 170 | 251| 754 | 50319 | 133945 | 
| 43 | 22 benchmarks/bench_mnist.py | 83 | 117| 323 | 50642 | 135746 | 
| 44 | 23 benchmarks/bench_hist_gradient_boosting_categorical_only.py | 1 | 81| 651 | 51293 | 136397 | 
| 45 | 23 sklearn/calibration.py | 1315 | 1429| 814 | 52107 | 136397 | 
| 46 | 24 examples/ensemble/plot_stack_predictors.py | 83 | 194| 786 | 52893 | 138272 | 
| 47 | 24 examples/ensemble/plot_gradient_boosting_quantile.py | 252 | 336| 822 | 53715 | 138272 | 
| 48 | 25 sklearn/model_selection/__init__.py | 1 | 90| 632 | 54347 | 138904 | 
| 49 | 25 examples/calibration/plot_calibration_multiclass.py | 97 | 170| 746 | 55093 | 138904 | 
| 50 | 26 examples/preprocessing/plot_scaling_importance.py | 157 | 256| 911 | 56004 | 141220 | 
| 51 | 27 sklearn/model_selection/_validation.py | 957 | 1015| 592 | 56596 | 157172 | 
| 52 | 27 sklearn/calibration.py | 514 | 587| 506 | 57102 | 157172 | 
| 53 | 28 benchmarks/bench_hist_gradient_boosting_threading.py | 87 | 139| 382 | 57484 | 159910 | 
| 54 | 28 examples/release_highlights/plot_release_highlights_0_23_0.py | 98 | 175| 789 | 58273 | 159910 | 
| 55 | 28 benchmarks/bench_hist_gradient_boosting_threading.py | 142 | 238| 894 | 59167 | 159910 | 
| 56 | 28 sklearn/calibration.py | 732 | 781| 446 | 59613 | 159910 | 
| 57 | 28 sklearn/utils/estimator_checks.py | 4015 | 4401| 3145 | 62758 | 159910 | 
| 58 | 29 benchmarks/bench_hist_gradient_boosting_higgsboson.py | 78 | 125| 391 | 63149 | 160970 | 
| 59 | 29 examples/ensemble/plot_gradient_boosting_categorical.py | 218 | 278| 687 | 63836 | 160970 | 
| 60 | 29 examples/release_highlights/plot_release_highlights_0_22_0.py | 90 | 193| 896 | 64732 | 160970 | 
| 61 | 30 examples/compose/plot_compare_reduction.py | 1 | 110| 778 | 65510 | 161956 | 
| 62 | 30 sklearn/linear_model/_coordinate_descent.py | 1563 | 1649| 739 | 66249 | 161956 | 
| 63 | 31 benchmarks/bench_covertype.py | 113 | 232| 820 | 67069 | 163874 | 
| 64 | 32 sklearn/conftest.py | 1 | 50| 425 | 67494 | 165689 | 
| 65 | 32 examples/linear_model/plot_poisson_regression_non_normal_loss.py | 289 | 390| 988 | 68482 | 165689 | 
| 66 | 33 examples/model_selection/plot_cv_predict.py | 1 | 79| 602 | 69084 | 166291 | 
| 67 | 34 examples/applications/plot_cyclical_feature_engineering.py | 110 | 220| 753 | 69837 | 173433 | 
| 68 | 35 examples/inspection/plot_linear_model_coefficient_interpretation.py | 384 | 482| 752 | 70589 | 179738 | 
| 69 | 35 sklearn/calibration.py | 1189 | 1293| 757 | 71346 | 179738 | 
| 70 | 36 examples/model_selection/plot_grid_search_text_feature_extraction.py | 110 | 219| 845 | 72191 | 181959 | 
| 71 | 36 benchmarks/bench_mnist.py | 120 | 234| 761 | 72952 | 181959 | 


## Missing Patch Files

 * 1: sklearn/isotonic.py

### Hint

```
I can reproduce it. We need to investigate but I would expect the inner estimator not being able to handle some dataframe because we expected NumPy arrays before.
This could be a bit like https://github.com/scikit-learn/scikit-learn/pull/25370 where things get confused when pandas output is configured. I think the solution is different (TSNE's PCA is truely "internal only") but it seems like there might be something more general to investigate/think about related to pandas output and nested estimators.
There is something quite smelly regarding the interaction between `IsotonicRegression` and pandas output:

<img width="1079" alt="image" src="https://user-images.githubusercontent.com/7454015/215147695-8aa08b83-705b-47a4-ab7c-43acb222098f.png">

It seems that we output a pandas Series when calling `predict` which is something that we don't do for any other estimator. `IsotonicRegression` is already quite special since it accepts a single feature. I need to investigate more to understand why we wrap the output of the `predict` method.
OK the reason is that `IsotonicRegression().predict(X)` call `IsotonicRegression().transform(X)` ;)
I don't know if we should have:

\`\`\`python
def predict(self, T):
    with config_context(transform_output="default"):
        return self.transform(T)
\`\`\`

or

\`\`\`python
def predict(self, T):
    return np.array(self.transform(T), copy=False).squeeze()
\`\`\`
Another solution would be to have a private `_transform` function called by both `transform` and `predict`. In this way, the `predict` call will not call the wrapper that is around the public `transform` method. I think this is even cleaner than the previous code.
/take
```

## Patch

```diff
diff --git a/sklearn/isotonic.py b/sklearn/isotonic.py
--- a/sklearn/isotonic.py
+++ b/sklearn/isotonic.py
@@ -360,23 +360,16 @@ def fit(self, X, y, sample_weight=None):
         self._build_f(X, y)
         return self
 
-    def transform(self, T):
-        """Transform new data by linear interpolation.
-
-        Parameters
-        ----------
-        T : array-like of shape (n_samples,) or (n_samples, 1)
-            Data to transform.
+    def _transform(self, T):
+        """`_transform` is called by both `transform` and `predict` methods.
 
-            .. versionchanged:: 0.24
-               Also accepts 2d array with 1 feature.
+        Since `transform` is wrapped to output arrays of specific types (e.g.
+        NumPy arrays, pandas DataFrame), we cannot make `predict` call `transform`
+        directly.
 
-        Returns
-        -------
-        y_pred : ndarray of shape (n_samples,)
-            The transformed data.
+        The above behaviour could be changed in the future, if we decide to output
+        other type of arrays when calling `predict`.
         """
-
         if hasattr(self, "X_thresholds_"):
             dtype = self.X_thresholds_.dtype
         else:
@@ -397,6 +390,24 @@ def transform(self, T):
 
         return res
 
+    def transform(self, T):
+        """Transform new data by linear interpolation.
+
+        Parameters
+        ----------
+        T : array-like of shape (n_samples,) or (n_samples, 1)
+            Data to transform.
+
+            .. versionchanged:: 0.24
+               Also accepts 2d array with 1 feature.
+
+        Returns
+        -------
+        y_pred : ndarray of shape (n_samples,)
+            The transformed data.
+        """
+        return self._transform(T)
+
     def predict(self, T):
         """Predict new data by linear interpolation.
 
@@ -410,7 +421,7 @@ def predict(self, T):
         y_pred : ndarray of shape (n_samples,)
             Transformed data.
         """
-        return self.transform(T)
+        return self._transform(T)
 
     # We implement get_feature_names_out here instead of using
     # `ClassNamePrefixFeaturesOutMixin`` because `input_features` are ignored.

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_isotonic.py b/sklearn/tests/test_isotonic.py
--- a/sklearn/tests/test_isotonic.py
+++ b/sklearn/tests/test_isotonic.py
@@ -5,6 +5,7 @@
 
 import pytest
 
+import sklearn
 from sklearn.datasets import make_regression
 from sklearn.isotonic import (
     check_increasing,
@@ -680,3 +681,24 @@ def test_get_feature_names_out(shape):
     assert isinstance(names, np.ndarray)
     assert names.dtype == object
     assert_array_equal(["isotonicregression0"], names)
+
+
+def test_isotonic_regression_output_predict():
+    """Check that `predict` does return the expected output type.
+
+    We need to check that `transform` will output a DataFrame and a NumPy array
+    when we set `transform_output` to `pandas`.
+
+    Non-regression test for:
+    https://github.com/scikit-learn/scikit-learn/issues/25499
+    """
+    pd = pytest.importorskip("pandas")
+    X, y = make_regression(n_samples=10, n_features=1, random_state=42)
+    regressor = IsotonicRegression()
+    with sklearn.config_context(transform_output="pandas"):
+        regressor.fit(X, y)
+        X_trans = regressor.transform(X)
+        y_pred = regressor.predict(X)
+
+    assert isinstance(X_trans, pd.DataFrame)
+    assert isinstance(y_pred, np.ndarray)

```


## Code snippets

### 1 - sklearn/calibration.py:

Start line: 56, End line: 244

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Probability calibration with isotonic regression or logistic regression.

    This class uses cross-validation to both estimate the parameters of a
    classifier and subsequently calibrate a classifier. With default
    `ensemble=True`, for each cv split it
    fits a copy of the base estimator to the training subset, and calibrates it
    using the testing subset. For prediction, predicted probabilities are
    averaged across these individual calibrated classifiers. When
    `ensemble=False`, cross-validation is used to obtain unbiased predictions,
    via :func:`~sklearn.model_selection.cross_val_predict`, which are then
    used for calibration. For prediction, the base estimator, trained using all
    the data, is used. This is the method implemented when `probabilities=True`
    for :mod:`sklearn.svm` estimators.

    Already fitted classifiers can be calibrated via the parameter
    `cv="prefit"`. In this case, no cross-validation is used and all provided
    data is used for calibration. The user has to take care manually that data
    for model fitting and calibration are disjoint.

    The calibration is based on the :term:`decision_function` method of the
    `estimator` if it exists, else on :term:`predict_proba`.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    estimator : estimator instance, default=None
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs. The default classifier is
        a :class:`~sklearn.svm.LinearSVC`.

        .. versionadded:: 1.2

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method (i.e. a logistic regression model) or
        'isotonic' which is a non-parametric approach. It is not advised to
        use isotonic calibration with too few calibration samples
        ``(<<1000)`` since it tends to overfit.

    cv : int, cross-validation generator, iterable or "prefit", \
            default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`~sklearn.model_selection.KFold`
        is used.

        Refer to the :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that `estimator` has been
        fitted already and all data is used for calibration.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        Base estimator clones are fitted in parallel across cross-validation
        iterations. Therefore parallelism happens only when `cv != "prefit"`.

        See :term:`Glossary <n_jobs>` for more details.

        .. versionadded:: 0.24

    ensemble : bool, default=True
        Determines how the calibrator is fitted when `cv` is not `'prefit'`.
        Ignored if `cv='prefit'`.

        If `True`, the `estimator` is fitted using training data, and
        calibrated using testing data, for each `cv` fold. The final estimator
        is an ensemble of `n_cv` fitted classifier and calibrator pairs, where
        `n_cv` is the number of cross-validation folds. The output is the
        average predicted probabilities of all pairs.

        If `False`, `cv` is used to compute unbiased predictions, via
        :func:`~sklearn.model_selection.cross_val_predict`, which are then
        used for calibration. At prediction time, the classifier used is the
        `estimator` trained on all the data.
        Note that this method is also internally implemented  in
        :mod:`sklearn.svm` estimators with the `probabilities=True` parameter.

        .. versionadded:: 0.24

    base_estimator : estimator instance
        This parameter is deprecated. Use `estimator` instead.

        .. deprecated:: 1.2
           The parameter `base_estimator` is deprecated in 1.2 and will be
           removed in 1.4. Use `estimator` instead.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    calibrated_classifiers_ : list (len() equal to cv or 1 if `cv="prefit"` \
            or `ensemble=False`)
        The list of classifier and calibrator pairs.

        - When `cv="prefit"`, the fitted `estimator` and fitted
          calibrator.
        - When `cv` is not "prefit" and `ensemble=True`, `n_cv` fitted
          `estimator` and calibrator pairs. `n_cv` is the number of
          cross-validation folds.
        - When `cv` is not "prefit" and `ensemble=False`, the `estimator`,
          fitted on all the data, and fitted calibrator.

        .. versionchanged:: 0.24
            Single calibrated classifier case when `ensemble=False`.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities
        for a calibration curve.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> base_clf = GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv=3)
    >>> calibrated_clf.fit(X, y)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    3
    >>> calibrated_clf.predict_proba(X)[:5, :]
    array([[0.110..., 0.889...],
           [0.072..., 0.927...],
           [0.928..., 0.071...],
           [0.928..., 0.071...],
           [0.071..., 0.928...]])
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=2,
    ...                            n_redundant=0, random_state=42)
    >>> X_train, X_calib, y_train, y_calib = train_test_split(
    ...        X, y, random_state=42
    ... )
    >>> base_clf = GaussianNB()
    >>> base_clf.fit(X_train, y_train)
    GaussianNB()
    >>> calibrated_clf = CalibratedClassifierCV(base_clf, cv="prefit")
    >>> calibrated_clf.fit(X_calib, y_calib)
    CalibratedClassifierCV(...)
    >>> len(calibrated_clf.calibrated_classifiers_)
    1
    >>> calibrated_clf.predict_proba([[-0.5, 0.5]])
    array([[0.936..., 0.063...]])
    """
```
### 2 - sklearn/calibration.py:

Start line: 336, End line: 453

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):

    def fit(self, X, y, sample_weight=None, **fit_params):
        # ... other code
        if self.cv == "prefit":
            # `classes_` should be consistent with that of estimator
            check_is_fitted(self.estimator, attributes=["classes_"])
            self.classes_ = self.estimator.classes_

            pred_method, method_name = _get_prediction_method(estimator)
            n_classes = len(self.classes_)
            predictions = _compute_predictions(pred_method, method_name, X, n_classes)

            calibrated_classifier = _fit_calibrator(
                estimator,
                predictions,
                y,
                self.classes_,
                self.method,
                sample_weight,
            )
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            # Set `classes_` using all `y`
            label_encoder_ = LabelEncoder().fit(y)
            self.classes_ = label_encoder_.classes_
            n_classes = len(self.classes_)

            # sample_weight checks
            fit_parameters = signature(estimator.fit).parameters
            supports_sw = "sample_weight" in fit_parameters
            if sample_weight is not None and not supports_sw:
                estimator_name = type(estimator).__name__
                warnings.warn(
                    f"Since {estimator_name} does not appear to accept sample_weight, "
                    "sample weights will only be used for the calibration itself. This "
                    "can be caused by a limitation of the current scikit-learn API. "
                    "See the following issue for more details: "
                    "https://github.com/scikit-learn/scikit-learn/issues/21134. Be "
                    "warned that the result of the calibration is likely to be "
                    "incorrect."
                )

            # Check that each cross-validation fold can have at least one
            # example per class
            if isinstance(self.cv, int):
                n_folds = self.cv
            elif hasattr(self.cv, "n_splits"):
                n_folds = self.cv.n_splits
            else:
                n_folds = None
            if n_folds and np.any(
                [np.sum(y == class_) < n_folds for class_ in self.classes_]
            ):
                raise ValueError(
                    f"Requesting {n_folds}-fold "
                    "cross-validation but provided less than "
                    f"{n_folds} examples for at least one class."
                )
            cv = check_cv(self.cv, y, classifier=True)

            if self.ensemble:
                parallel = Parallel(n_jobs=self.n_jobs)
                self.calibrated_classifiers_ = parallel(
                    delayed(_fit_classifier_calibrator_pair)(
                        clone(estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        method=self.method,
                        classes=self.classes_,
                        supports_sw=supports_sw,
                        sample_weight=sample_weight,
                        **fit_params,
                    )
                    for train, test in cv.split(X, y)
                )
            else:
                this_estimator = clone(estimator)
                _, method_name = _get_prediction_method(this_estimator)
                fit_params = (
                    {"sample_weight": sample_weight}
                    if sample_weight is not None and supports_sw
                    else None
                )
                pred_method = partial(
                    cross_val_predict,
                    estimator=this_estimator,
                    X=X,
                    y=y,
                    cv=cv,
                    method=method_name,
                    n_jobs=self.n_jobs,
                    fit_params=fit_params,
                )
                predictions = _compute_predictions(
                    pred_method, method_name, X, n_classes
                )

                if sample_weight is not None and supports_sw:
                    this_estimator.fit(X, y, sample_weight=sample_weight)
                else:
                    this_estimator.fit(X, y)
                # Note: Here we don't pass on fit_params because the supported
                # calibrators don't support fit_params anyway
                calibrated_classifier = _fit_calibrator(
                    this_estimator,
                    predictions,
                    y,
                    self.classes_,
                    self.method,
                    sample_weight,
                )
                self.calibrated_classifiers_.append(calibrated_classifier)

        first_clf = self.calibrated_classifiers_[0].estimator
        if hasattr(first_clf, "n_features_in_"):
            self.n_features_in_ = first_clf.n_features_in_
        if hasattr(first_clf, "feature_names_in_"):
            self.feature_names_in_ = first_clf.feature_names_in_
        return self
```
### 3 - sklearn/calibration.py:

Start line: 246, End line: 279

```python
class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
        ],
        "method": [StrOptions({"isotonic", "sigmoid"})],
        "cv": ["cv_object", StrOptions({"prefit"})],
        "n_jobs": [Integral, None],
        "ensemble": ["boolean"],
        "base_estimator": [
            HasMethods(["fit", "predict_proba"]),
            HasMethods(["fit", "decision_function"]),
            None,
            Hidden(StrOptions({"deprecated"})),
        ],
    }

    def __init__(
        self,
        estimator=None,
        *,
        method="sigmoid",
        cv=None,
        n_jobs=None,
        ensemble=True,
        base_estimator="deprecated",
    ):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.base_estimator = base_estimator
```
### 4 - examples/calibration/plot_calibration_curve.py:

Start line: 103, End line: 197

```python
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()

# %%
# Uncalibrated :class:`~sklearn.naive_bayes.GaussianNB` is poorly calibrated
# because of
# the redundant features which violate the assumption of feature-independence
# and result in an overly confident classifier, which is indicated by the
# typical transposed-sigmoid curve. Calibration of the probabilities of
# :class:`~sklearn.naive_bayes.GaussianNB` with :ref:`isotonic` can fix
# this issue as can be seen from the nearly diagonal calibration curve.
# :ref:`Sigmoid regression <sigmoid_regressor>` also improves calibration
# slightly,
# albeit not as strongly as the non-parametric isotonic regression. This can be
# attributed to the fact that we have plenty of calibration data such that the
# greater flexibility of the non-parametric model can be exploited.
#
# Below we will make a quantitative analysis considering several classification
# metrics: :ref:`brier_score_loss`, :ref:`log_loss`,
# :ref:`precision, recall, F1 score <precision_recall_f_measure_metrics>` and
# :ref:`ROC AUC <roc_metrics>`.

from collections import defaultdict

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

scores = defaultdict(list)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)

    for metric in [brier_score_loss, log_loss]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))

    for metric in [precision_score, recall_score, f1_score, roc_auc_score]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_pred))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df.round(decimals=3)

score_df

# %%
# Notice that although calibration improves the :ref:`brier_score_loss` (a
# metric composed
# of calibration term and refinement term) and :ref:`log_loss`, it does not
# significantly alter the prediction accuracy measures (precision, recall and
# F1 score).
# This is because calibration should not significantly change prediction
# probabilities at the location of the decision threshold (at x = 0.5 on the
# graph). Calibration should however, make the predicted probabilities more
# accurate and thus more useful for making allocation decisions under
# uncertainty.
# Further, ROC AUC, should not change at all because calibration is a
# monotonic transformation. Indeed, no rank metrics are affected by
# calibration.
#
# Linear support vector classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we will compare:
#
# * :class:`~sklearn.linear_model.LogisticRegression` (baseline)
# * Uncalibrated :class:`~sklearn.svm.LinearSVC`. Since SVC does not output
#   probabilities by default, we naively scale the output of the
#   :term:`decision_function` into [0, 1] by applying min-max scaling.
# * :class:`~sklearn.svm.LinearSVC` with isotonic and sigmoid
#   calibration (see :ref:`User Guide <calibration>`)

import numpy as np
```
### 5 - examples/calibration/plot_calibration.py:

Start line: 1, End line: 88

```python
"""
======================================
Probability calibration of classifiers
======================================

When performing classification you often want to predict not only
the class label, but also the associated probability. This probability
gives you some kind of confidence on the prediction. However, not all
classifiers provide well-calibrated probabilities, some being over-confident
while others being under-confident. Thus, a separate calibration of predicted
probabilities is often desirable as a postprocessing. This example illustrates
two different methods for this calibration and evaluates the quality of the
returned probabilities using Brier's score
(see https://en.wikipedia.org/wiki/Brier_score).

Compared are the estimated probability using a Gaussian naive Bayes classifier
without calibration, with a sigmoid calibration, and with a non-parametric
isotonic calibration. One can observe that only the non-parametric model is
able to provide a probability calibration that returns probabilities close
to the expected 0.5 for most of the samples belonging to the middle
cluster with heterogeneous labels. This results in a significantly improved
Brier score.

"""
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

# Generate 3 blobs with 2 classes where the second blob contains
# half positive samples and half negative samples. Probability in this
# blob is therefore 0.5.
centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(n_samples=n_samples, centers=centers, shuffle=False, random_state=42)

y[: n_samples // 2] = 0
y[n_samples // 2 :] = 1
sample_weight = np.random.RandomState(42).rand(y.shape[0])

# split train, test for calibration
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, test_size=0.9, random_state=42
)

# %%
# Gaussian Naive-Bayes
# --------------------
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB

# With no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# With isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
clf_isotonic.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# With sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method="sigmoid")
clf_sigmoid.fit(X_train, y_train, sample_weight=sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier score losses: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf, sample_weight=sw_test)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sample_weight=sw_test)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sample_weight=sw_test)
```
### 6 - examples/calibration/plot_calibration_multiclass.py:

Start line: 229, End line: 275

```python
p0, p1 = np.meshgrid(p1d, p1d)
p2 = 1 - p0 - p1
p = np.c_[p0.ravel(), p1.ravel(), p2.ravel()]
p = p[p[:, 2] >= 0]

# Use the three class-wise calibrators to compute calibrated probabilities
calibrated_classifier = cal_clf.calibrated_classifiers_[0]
prediction = np.vstack(
    [
        calibrator.predict(this_p)
        for calibrator, this_p in zip(calibrated_classifier.calibrators, p.T)
    ]
).T

# Re-normalize the calibrated predictions to make sure they stay inside the
# simplex. This same renormalization step is performed internally by the
# predict method of CalibratedClassifierCV on multiclass problems.
prediction /= prediction.sum(axis=1)[:, None]

# Plot changes in predicted probabilities induced by the calibrators
for i in range(prediction.shape[0]):
    plt.arrow(
        p[i, 0],
        p[i, 1],
        prediction[i, 0] - p[i, 0],
        prediction[i, 1] - p[i, 1],
        head_width=1e-2,
        color=colors[np.argmax(p[i])],
    )

# Plot the boundaries of the unit simplex
plt.plot([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], "k", label="Simplex")

plt.grid(False)
for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    plt.plot([0, x], [x, 0], "k", alpha=0.2)
    plt.plot([0, 0 + (1 - x) / 2], [x, x + (1 - x) / 2], "k", alpha=0.2)
    plt.plot([x, x + (1 - x) / 2], [0, 0 + (1 - x) / 2], "k", alpha=0.2)

plt.title("Learned sigmoid calibration map")
plt.xlabel("Probability class 1")
plt.ylabel("Probability class 2")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)

plt.show()
```
### 7 - sklearn/calibration.py:

Start line: 1, End line: 511

```python
"""Calibration of predicted probabilities."""

from numbers import Integral
import warnings
from inspect import signature
from functools import partial

from math import log
import numpy as np

from scipy.special import expit
from scipy.special import xlogy
from scipy.optimize import fmin_bfgs

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    MetaEstimatorMixin,
    is_classifier,
)
from .preprocessing import label_binarize, LabelEncoder
from .utils import (
    column_or_1d,
    indexable,
    check_matplotlib_support,
)

from .utils.multiclass import check_classification_targets
from .utils.parallel import delayed, Parallel
from .utils._param_validation import StrOptions, HasMethods, Hidden
from .utils.validation import (
    _check_fit_params,
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
)
from .utils import _safe_indexing
from .isotonic import IsotonicRegression
from .svm import LinearSVC
from .model_selection import check_cv, cross_val_predict
from .metrics._base import _check_pos_label_consistency
from .metrics._plot.base import _get_response


class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
```
### 8 - examples/calibration/plot_calibration.py:

Start line: 89, End line: 145

```python
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

# %%
# Plot data and the predicted probabilities
# -----------------------------------------
from matplotlib import cm
import matplotlib.pyplot as plt

plt.figure()
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X_train[y_train == this_y]
    this_sw = sw_train[y_train == this_y]
    plt.scatter(
        this_X[:, 0],
        this_X[:, 1],
        s=this_sw * 50,
        c=color[np.newaxis, :],
        alpha=0.5,
        edgecolor="k",
        label="Class %s" % this_y,
    )
plt.legend(loc="best")
plt.title("Data")

plt.figure()

order = np.lexsort((prob_pos_clf,))
plt.plot(prob_pos_clf[order], "r", label="No calibration (%1.3f)" % clf_score)
plt.plot(
    prob_pos_isotonic[order],
    "g",
    linewidth=3,
    label="Isotonic calibration (%1.3f)" % clf_isotonic_score,
)
plt.plot(
    prob_pos_sigmoid[order],
    "b",
    linewidth=3,
    label="Sigmoid calibration (%1.3f)" % clf_sigmoid_score,
)
plt.plot(
    np.linspace(0, y_test.size, 51)[1::2],
    y_test[order].reshape(25, -1).mean(1),
    "k",
    linewidth=3,
    label=r"Empirical",
)
plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability (uncalibrated GNB)")
plt.ylabel("P(y=1)")
plt.legend(loc="upper left")
plt.title("Gaussian naive Bayes probabilities")

plt.show()
```
### 9 - examples/calibration/plot_compare_calibration.py:

Start line: 85, End line: 211

```python
# %%

from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = NaivelyCalibratedLinearSVC(C=1.0)
rfc = RandomForestClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
]

# %%

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()

# %%
# :class:`~sklearn.linear_model.LogisticRegression` returns well calibrated
# predictions as it directly optimizes log-loss. In contrast, the other methods
# return biased probabilities, with different biases for each method:
#
# * :class:`~sklearn.naive_bayes.GaussianNB` tends to push
#   probabilities to 0 or 1 (see histogram). This is mainly
#   because the naive Bayes equation only provides correct estimate of
#   probabilities when the assumption that features are conditionally
#   independent holds [2]_. However, features tend to be positively correlated
#   and is the case with this dataset, which contains 2 features
#   generated as random linear combinations of the informative features. These
#   correlated features are effectively being 'counted twice', resulting in
#   pushing the predicted probabilities towards 0 and 1 [3]_.
#
# * :class:`~sklearn.ensemble.RandomForestClassifier` shows the opposite
#   behavior: the histograms show peaks at approx. 0.2 and 0.9 probability,
#   while probabilities close to 0 or 1 are very rare. An explanation for this
#   is given by Niculescu-Mizil and Caruana [1]_: "Methods such as bagging and
#   random forests that average predictions from a base set of models can have
#   difficulty making predictions near 0 and 1 because variance in the
#   underlying base models will bias predictions that should be near zero or
#   one away from these values. Because predictions are restricted to the
#   interval [0,1], errors caused by variance tend to be one- sided near zero
#   and one. For example, if a model should predict p = 0 for a case, the only
#   way bagging can achieve this is if all bagged trees predict zero. If we add
#   noise to the trees that bagging is averaging over, this noise will cause
#   some trees to predict values larger than 0 for this case, thus moving the
#   average prediction of the bagged ensemble away from 0. We observe this
#   effect most strongly with random forests because the base-level trees
#   trained with random forests have relatively high variance due to feature
#   subsetting." As a result, the calibration curve shows a characteristic
#   sigmoid shape, indicating that the classifier is under-confident
#   and could return probabilities closer to 0 or 1.
#
# * To show the performance of :class:`~sklearn.svm.LinearSVC`, we naively
#   scale the output of the :term:`decision_function` into [0, 1] by applying
#   min-max scaling, since SVC does not output probabilities by default.
#   :class:`~sklearn.svm.LinearSVC` shows an
#   even more sigmoid curve than the
#   :class:`~sklearn.ensemble.RandomForestClassifier`, which is typical for
#   maximum-margin methods [1]_ as they focus on difficult to classify samples
#   that are close to the decision boundary (the support vectors).
#
# References
# ----------
#
# .. [1] `Predicting Good Probabilities with Supervised Learning
#        <https://dl.acm.org/doi/pdf/10.1145/1102351.1102430>`_,
#        A. Niculescu-Mizil & R. Caruana, ICML 2005
# .. [2] `Beyond independence: Conditions for the optimality of the simple
#        bayesian classifier
#        <https://www.ics.uci.edu/~pazzani/Publications/mlc96-pedro.pdf>`_
#        Domingos, P., & Pazzani, M., Proc. 13th Intl. Conf. Machine Learning.
#        1996.
# .. [3] `Obtaining calibrated probability estimates from decision trees and
#        naive Bayesian classifiers
#        <https://citeseerx.ist.psu.edu/doc_view/pid/4f67a122ec3723f08ad5cbefecad119b432b3304>`_
#        Zadrozny, Bianca, and Charles Elkan. Icml. Vol. 1. 2001.
```
### 10 - examples/calibration/plot_calibration_curve.py:

Start line: 1, End line: 102

```python
"""
==============================
Probability Calibration curves
==============================

When performing classification one often wants to predict not only the class
label, but also the associated probability. This probability gives some
kind of confidence on the prediction. This example demonstrates how to
visualize how well calibrated the predicted probabilities are using calibration
curves, also known as reliability diagrams. Calibration of an uncalibrated
classifier will also be demonstrated.

"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=10, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)

# %%
# Calibration curves
# ------------------
#
# Gaussian Naive Bayes
# ^^^^^^^^^^^^^^^^^^^^
#
# First, we will compare:
#
# * :class:`~sklearn.linear_model.LogisticRegression` (used as baseline
#   since very often, properly regularized logistic regression is well
#   calibrated by default thanks to the use of the log-loss)
# * Uncalibrated :class:`~sklearn.naive_bayes.GaussianNB`
# * :class:`~sklearn.naive_bayes.GaussianNB` with isotonic and sigmoid
#   calibration (see :ref:`User Guide <calibration>`)
#
# Calibration curves for all 4 conditions are plotted below, with the average
# predicted probability for each bin on the x-axis and the fraction of positive
# classes in each bin on the y-axis.

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

lr = LogisticRegression(C=1.0)
gnb = GaussianNB()
gnb_isotonic = CalibratedClassifierCV(gnb, cv=2, method="isotonic")
gnb_sigmoid = CalibratedClassifierCV(gnb, cv=2, method="sigmoid")

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (gnb_isotonic, "Naive Bayes + Isotonic"),
    (gnb_sigmoid, "Naive Bayes + Sigmoid"),
]

# %%
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
```

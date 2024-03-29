# scikit-learn__scikit-learn-26400

| **scikit-learn/scikit-learn** | `1e8a5b833d1b58f3ab84099c4582239af854b23a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4077 |
| **Any found context length** | 4077 |
| **Avg pos** | 8.0 |
| **Min pos** | 8 |
| **Max pos** | 8 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/preprocessing/_data.py b/sklearn/preprocessing/_data.py
--- a/sklearn/preprocessing/_data.py
+++ b/sklearn/preprocessing/_data.py
@@ -3311,9 +3311,13 @@ def _box_cox_optimize(self, x):
 
         We here use scipy builtins which uses the brent optimizer.
         """
+        mask = np.isnan(x)
+        if np.all(mask):
+            raise ValueError("Column must not be all nan.")
+
         # the computation of lambda is influenced by NaNs so we need to
         # get rid of them
-        _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)
+        _, lmbda = stats.boxcox(x[~mask], lmbda=None)
 
         return lmbda
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/preprocessing/_data.py | 3314 | 3314 | 8 | 1 | 4077


## Problem Statement

```
PowerTransformer fails with unhelpful stack trace with all-nan feature and method='box-cox'
### Describe the bug

`PowerTransformer("box-cox").fit(x)` throws a difficult-to-debug error if x contains an all-nan column. 

### Steps/Code to Reproduce

\`\`\`python
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer, StandardScaler

x = np.ones((20, 5))
y = np.ones((20, 1))

x[:, 0] = np.nan

PowerTransformer().fit_transform(x)  # preserves all-nan column
PowerTransformer('box-cox').fit_transform(x)  # Throws an error when calling stats.boxcox
\`\`\`

### Expected Results

Either no error is thrown and the all-nan column is preserved, or a descriptive error is thrown indicating that there is an unfittable column 

### Actual Results

\`\`\`
ValueError                                Traceback (most recent call last)

[<ipython-input-12-563273596add>](https://localhost:8080/#) in <cell line: 1>()
----> 1 PowerTransformer('box-cox').fit_transform(x)

4 frames

[/usr/local/lib/python3.10/dist-packages/sklearn/utils/_set_output.py](https://localhost:8080/#) in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

[/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py](https://localhost:8080/#) in fit_transform(self, X, y)
   3101         """
   3102         self._validate_params()
-> 3103         return self._fit(X, y, force_transform=True)
   3104 
   3105     def _fit(self, X, y=None, force_transform=False):

[/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py](https://localhost:8080/#) in _fit(self, X, y, force_transform)
   3114         }[self.method]
   3115         with np.errstate(invalid="ignore"):  # hide NaN warnings
-> 3116             self.lambdas_ = np.array([optim_function(col) for col in X.T])
   3117 
   3118         if self.standardize or force_transform:

[/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py](https://localhost:8080/#) in <listcomp>(.0)
   3114         }[self.method]
   3115         with np.errstate(invalid="ignore"):  # hide NaN warnings
-> 3116             self.lambdas_ = np.array([optim_function(col) for col in X.T])
   3117 
   3118         if self.standardize or force_transform:

[/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_data.py](https://localhost:8080/#) in _box_cox_optimize(self, x)
   3272         # the computation of lambda is influenced by NaNs so we need to
   3273         # get rid of them
-> 3274         _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)
   3275 
   3276         return lmbda

ValueError: not enough values to unpack (expected 2, got 0)
\`\`\`

### Versions

\`\`\`shell
System:
    python: 3.10.11 (main, Apr  5 2023, 14:15:10) [GCC 9.4.0]
executable: /usr/bin/python3
   machine: Linux-5.10.147+-x86_64-with-glibc2.31

Python dependencies:
      sklearn: 1.2.2
          pip: 23.0.1
   setuptools: 67.7.2
        numpy: 1.22.4
        scipy: 1.10.1
       Cython: 0.29.34
       pandas: 1.5.3
   matplotlib: 3.7.1
       joblib: 1.2.0
threadpoolctl: 3.1.0
\`\`\`
\`\`\`


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/preprocessing/_data.py** | 2998 | 3091| 788 | 788 | 27877 | 
| 2 | **1 sklearn/preprocessing/_data.py** | 3147 | 3176| 303 | 1091 | 27877 | 
| 3 | **1 sklearn/preprocessing/_data.py** | 3400 | 3500| 926 | 2017 | 27877 | 
| 4 | **1 sklearn/preprocessing/_data.py** | 3351 | 3397| 376 | 2393 | 27877 | 
| 5 | 2 examples/preprocessing/plot_all_scaling.py | 249 | 350| 964 | 3357 | 31319 | 
| 6 | **2 sklearn/preprocessing/_data.py** | 3093 | 3125| 225 | 3582 | 31319 | 
| 7 | **2 sklearn/preprocessing/_data.py** | 3127 | 3145| 145 | 3727 | 31319 | 
| **-> 8 <-** | **2 sklearn/preprocessing/_data.py** | 3286 | 3318| 350 | 4077 | 31319 | 
| 9 | 3 sklearn/utils/estimator_checks.py | 834 | 1553| 6001 | 10078 | 67769 | 
| 10 | 4 sklearn/preprocessing/_polynomial.py | 1125 | 1170| 470 | 10548 | 78476 | 
| 11 | **4 sklearn/preprocessing/_data.py** | 3178 | 3205| 225 | 10773 | 78476 | 
| 12 | **4 sklearn/preprocessing/_data.py** | 3254 | 3284| 342 | 11115 | 78476 | 
| 13 | 5 examples/preprocessing/plot_map_data_to_normal.py | 1 | 102| 760 | 11875 | 79762 | 
| 14 | 5 sklearn/utils/estimator_checks.py | 4002 | 4409| 3326 | 15201 | 79762 | 
| 15 | **5 sklearn/preprocessing/_data.py** | 3207 | 3252| 413 | 15614 | 79762 | 
| 16 | 6 examples/compose/plot_column_transformer_mixed_types.py | 1 | 122| 879 | 16493 | 81559 | 
| 17 | **6 sklearn/preprocessing/_data.py** | 3320 | 3349| 310 | 16803 | 81559 | 
| 18 | **6 sklearn/preprocessing/_data.py** | 11 | 71| 322 | 17125 | 81559 | 
| 19 | 7 examples/miscellaneous/plot_set_output.py | 88 | 139| 367 | 17492 | 82626 | 
| 20 | 7 examples/preprocessing/plot_all_scaling.py | 352 | 401| 389 | 17881 | 82626 | 
| 21 | 8 sklearn/compose/_column_transformer.py | 1 | 912| 302 | 18183 | 92063 | 
| 22 | 9 examples/release_highlights/plot_release_highlights_1_0_0.py | 92 | 165| 775 | 18958 | 94398 | 
| 23 | 9 examples/preprocessing/plot_all_scaling.py | 1 | 85| 632 | 19590 | 94398 | 
| 24 | 10 examples/neighbors/approximate_nearest_neighbors.py | 103 | 212| 983 | 20573 | 97132 | 
| 25 | 10 examples/preprocessing/plot_all_scaling.py | 86 | 118| 274 | 20847 | 97132 | 
| 26 | 11 examples/release_highlights/plot_release_highlights_1_1_0.py | 1 | 96| 775 | 21622 | 99267 | 
| 27 | 11 sklearn/compose/_column_transformer.py | 43 | 215| 1827 | 23449 | 99267 | 
| 28 | 12 benchmarks/bench_random_projections.py | 96 | 304| 1338 | 24787 | 101088 | 
| 29 | 13 examples/compose/plot_column_transformer.py | 1 | 67| 430 | 25217 | 102324 | 
| 30 | 13 examples/compose/plot_column_transformer_mixed_types.py | 124 | 215| 758 | 25975 | 102324 | 
| 31 | 13 sklearn/utils/estimator_checks.py | 3225 | 3999| 6285 | 32260 | 102324 | 
| 32 | 13 sklearn/preprocessing/_polynomial.py | 436 | 574| 1334 | 33594 | 102324 | 
| 33 | 14 sklearn/cross_decomposition/_pls.py | 645 | 739| 852 | 34446 | 111407 | 
| 34 | 15 sklearn/preprocessing/__init__.py | 1 | 73| 432 | 34878 | 111839 | 
| 35 | 16 sklearn/impute/_base.py | 457 | 513| 497 | 35375 | 120215 | 
| 36 | 17 examples/preprocessing/plot_target_encoder.py | 105 | 228| 1113 | 36488 | 122019 | 
| 37 | **17 sklearn/preprocessing/_data.py** | 2752 | 2774| 195 | 36683 | 122019 | 
| 38 | 18 examples/linear_model/plot_poisson_regression_non_normal_loss.py | 101 | 161| 441 | 37124 | 127352 | 
| 39 | 19 sklearn/preprocessing/_function_transformer.py | 129 | 170| 329 | 37453 | 130080 | 
| 40 | 20 sklearn/preprocessing/_encoders.py | 70 | 173| 744 | 38197 | 144180 | 
| 41 | 21 examples/inspection/plot_partial_dependence.py | 86 | 178| 833 | 39030 | 149153 | 
| 42 | 22 examples/release_highlights/plot_release_highlights_0_22_0.py | 90 | 193| 896 | 39926 | 151563 | 
| 43 | 22 examples/miscellaneous/plot_set_output.py | 1 | 87| 700 | 40626 | 151563 | 
| 44 | 22 examples/preprocessing/plot_all_scaling.py | 199 | 246| 416 | 41042 | 151563 | 
| 45 | 22 sklearn/preprocessing/_encoders.py | 175 | 241| 554 | 41596 | 151563 | 
| 46 | **22 sklearn/preprocessing/_data.py** | 2855 | 2995| 1271 | 42867 | 151563 | 
| 47 | **22 sklearn/preprocessing/_data.py** | 902 | 987| 706 | 43573 | 151563 | 
| 48 | 22 sklearn/impute/_base.py | 415 | 455| 366 | 43939 | 151563 | 
| 49 | 23 examples/compose/plot_compare_reduction.py | 1 | 110| 778 | 44717 | 152549 | 
| 50 | 23 examples/release_highlights/plot_release_highlights_1_1_0.py | 98 | 177| 845 | 45562 | 152549 | 
| 51 | 24 examples/release_highlights/plot_release_highlights_1_2_0.py | 92 | 167| 718 | 46280 | 154034 | 
| 52 | 24 sklearn/compose/_column_transformer.py | 710 | 765| 456 | 46736 | 154034 | 
| 53 | 24 sklearn/compose/_column_transformer.py | 687 | 708| 159 | 46895 | 154034 | 
| 54 | 24 sklearn/preprocessing/_polynomial.py | 987 | 1123| 1449 | 48344 | 154034 | 
| 55 | 25 examples/release_highlights/plot_release_highlights_0_24_0.py | 121 | 210| 777 | 49121 | 156472 | 
| 56 | 26 examples/applications/plot_cyclical_feature_engineering.py | 110 | 221| 759 | 49880 | 163732 | 
| 57 | **26 sklearn/preprocessing/_data.py** | 2807 | 2826| 179 | 50059 | 163732 | 
| 58 | 26 sklearn/preprocessing/_function_transformer.py | 1 | 127| 1012 | 51071 | 163732 | 
| 59 | 26 sklearn/compose/_column_transformer.py | 614 | 627| 146 | 51217 | 163732 | 
| 60 | 27 sklearn/inspection/_partial_dependence.py | 100 | 136| 364 | 51581 | 170674 | 
| 61 | 27 sklearn/impute/_base.py | 5 | 22| 129 | 51710 | 170674 | 
| 62 | 28 sklearn/pipeline.py | 926 | 958| 275 | 51985 | 181321 | 
| 63 | 29 sklearn/manifold/_t_sne.py | 11 | 33| 222 | 52207 | 191338 | 
| 64 | 29 examples/release_highlights/plot_release_highlights_1_2_0.py | 1 | 90| 766 | 52973 | 191338 | 
| 65 | 30 sklearn/feature_extraction/_dict_vectorizer.py | 190 | 287| 729 | 53702 | 194541 | 
| 66 | 30 examples/applications/plot_cyclical_feature_engineering.py | 451 | 524| 747 | 54449 | 194541 | 
| 67 | **30 sklearn/preprocessing/_data.py** | 2592 | 2635| 474 | 54923 | 194541 | 
| 68 | 31 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 91| 780 | 55703 | 195876 | 
| 69 | **31 sklearn/preprocessing/_data.py** | 2534 | 2558| 210 | 55913 | 195876 | 
| 70 | 31 sklearn/inspection/_partial_dependence.py | 668 | 743| 876 | 56789 | 195876 | 


### Hint

```
Thank you for opening the issue. I agree this is a bug. It is reasonable to return all nans to be consistent with `yeo-johnson`.
Would the following approach be neat enough?

\`\`\`python
def _box_cox_optimize(self, x):
    # The computation of lambda is influenced by NaNs so we need to
    # get rid of them
    x = x[~np.isnan(x)]
        
    # if the whole column is nan, we do not care about lambda
    if len(x) == 0:
        return 0
        
    _, lmbda = stats.boxcox(x, lmbda=None)
    return lmbda

\`\`\`
If this is okay, I can open a PR for this.
On second thought, `box-cox` does not work when the data is constant:

\`\`\`python
from sklearn.preprocessing import PowerTransformer

x = [[1], [1], [1], [1]]

pt = PowerTransformer(method="box-cox")
pt.fit_transform(x)
# ValueError: Data must not be constant.
\`\`\`

A feature that is all `np.nan` can be considered constant. If we want to stay consistent, then we raise a similar error for all `np.nan`.

With that in mind, I'm in favor of raising an informative error.
@thomasjpfan That's indeed reasonable. I have two proposed solutions:

1. Let scipy raise the error, so that the message will be consistent with scipy:

\`\`\`python
def _box_cox_optimize(self, x):
    if not np.all(np.isnan(x)):
        x = x[~np.isnan(x)]

    _, lmbda = stats.boxcox(x, lmbda=None)
    return lmbda
\`\`\`

2. Raise our own error, specifically claiming that column cannot be all nan (rather than cannot be constant):

\`\`\`python
def _box_cox_optimize(self, x):
    if np.all(np.isnan(x)):
        raise ValueError("Column must not be all nan.")

    _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)
    return lmbda
\`\`\`

Which one would you prefer, our do you have any other recommended solution? (I'm thinking that maybe my proposed solutions are not efficient enough.)
Since there is no reply, I'm going to open a PR that takes the second approach. The reason is that the second approach is clearer IMO and the first approach seems to trigger some unexpected behavior.
I like the second approach in https://github.com/scikit-learn/scikit-learn/issues/26303#issuecomment-1536899848, but store the `np.isnan(x)` as a variable so it is not computed twice.
I see, thanks for the comment!
```

## Patch

```diff
diff --git a/sklearn/preprocessing/_data.py b/sklearn/preprocessing/_data.py
--- a/sklearn/preprocessing/_data.py
+++ b/sklearn/preprocessing/_data.py
@@ -3311,9 +3311,13 @@ def _box_cox_optimize(self, x):
 
         We here use scipy builtins which uses the brent optimizer.
         """
+        mask = np.isnan(x)
+        if np.all(mask):
+            raise ValueError("Column must not be all nan.")
+
         # the computation of lambda is influenced by NaNs so we need to
         # get rid of them
-        _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)
+        _, lmbda = stats.boxcox(x[~mask], lmbda=None)
 
         return lmbda
 

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_data.py b/sklearn/preprocessing/tests/test_data.py
--- a/sklearn/preprocessing/tests/test_data.py
+++ b/sklearn/preprocessing/tests/test_data.py
@@ -2527,6 +2527,21 @@ def test_power_transformer_copy_False(method, standardize):
     assert X_trans is X_inv_trans
 
 
+def test_power_transformer_box_cox_raise_all_nans_col():
+    """Check that box-cox raises informative when a column contains all nans.
+
+    Non-regression test for gh-26303
+    """
+    X = rng.random_sample((4, 5))
+    X[:, 0] = np.nan
+
+    err_msg = "Column must not be all nan."
+
+    pt = PowerTransformer(method="box-cox")
+    with pytest.raises(ValueError, match=err_msg):
+        pt.fit_transform(X)
+
+
 @pytest.mark.parametrize(
     "X_2",
     [

```


## Code snippets

### 1 - sklearn/preprocessing/_data.py:

Start line: 2998, End line: 3091

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : ndarray of float of shape (n_features,)
        The parameters of the power transformation for the selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer()
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]
    """
```
### 2 - sklearn/preprocessing/_data.py:

Start line: 3147, End line: 3176

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _fit(self, X, y=None, force_transform=False):
        X = self._check_input(X, in_fit=True, check_positive=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        optim_function = {
            "box-cox": self._box_cox_optimize,
            "yeo-johnson": self._yeo_johnson_optimize,
        }[self.method]
        with np.errstate(invalid="ignore"):  # hide NaN warnings
            self.lambdas_ = np.array([optim_function(col) for col in X.T])

        if self.standardize or force_transform:
            transform_function = {
                "box-cox": boxcox,
                "yeo-johnson": self._yeo_johnson_transform,
            }[self.method]
            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid="ignore"):  # hide NaN warnings
                    X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            self._scaler = StandardScaler(copy=False)
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

        return X
```
### 3 - sklearn/preprocessing/_data.py:

Start line: 3400, End line: 3500

```python
@validate_params({"X": ["array-like"]})
def power_transform(X, method="yeo-johnson", *, standardize=True, copy=True):
    """Parametric, monotonic transformation to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        .. versionchanged:: 0.23
            The default value of the `method` parameter changed from
            'box-cox' to 'yeo-johnson' in 0.23.

    standardize : bool, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : bool, default=True
        Set to False to perform inplace computation during transformation.

    Returns
    -------
    X_trans : ndarray of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    PowerTransformer : Equivalent transformation with the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import power_transform
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(power_transform(data, method='box-cox'))
    [[-1.332... -0.707...]
     [ 0.256... -0.707...]
     [ 1.076...  1.414...]]

    .. warning:: Risk of data leak.
        Do not use :func:`~sklearn.preprocessing.power_transform` unless you
        know what you are doing. A common mistake is to apply it to the entire
        data *before* splitting into training and test sets. This will bias the
        model evaluation because information would have leaked from the test
        set to the training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.PowerTransformer` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking, e.g.: `pipe = make_pipeline(PowerTransformer(),
        LogisticRegression())`.
    """
    pt = PowerTransformer(method=method, standardize=standardize, copy=copy)
    return pt.fit_transform(X)
```
### 4 - sklearn/preprocessing/_data.py:

Start line: 3351, End line: 3397

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _check_input(self, X, in_fit, check_positive=False, check_shape=False):
        """Validate the input before fit and transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        in_fit : bool
            Whether or not `_check_input` is called from `fit` or other
            methods, e.g. `predict`, `transform`, etc.

        check_positive : bool, default=False
            If True, check that all data is positive and non-zero (only if
            ``self.method=='box-cox'``).

        check_shape : bool, default=False
            If True, check that n_features matches the length of self.lambdas_
        """
        X = self._validate_data(
            X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            if check_positive and self.method == "box-cox" and np.nanmin(X) <= 0:
                raise ValueError(
                    "The Box-Cox transformation can only be "
                    "applied to strictly positive data"
                )

        if check_shape and not X.shape[1] == len(self.lambdas_):
            raise ValueError(
                "Input data has a different number of features "
                "than fitting data. Should have {n}, data has {m}".format(
                    n=len(self.lambdas_), m=X.shape[1]
                )
            )

        return X

    def _more_tags(self):
        return {"allow_nan": True}
```
### 5 - examples/preprocessing/plot_all_scaling.py:

Start line: 249, End line: 350

```python
# %%
# .. _results:
#
# Original data
# -------------
#
# Each transformation is plotted showing two transformed features, with the
# left plot showing the entire dataset, and the right zoomed-in to show the
# dataset without the marginal outliers. A large majority of the samples are
# compacted to a specific range, [0, 10] for the median income and [0, 6] for
# the average house occupancy. Note that there are some marginal outliers (some
# blocks have average occupancy of more than 1200). Therefore, a specific
# pre-processing can be very beneficial depending of the application. In the
# following, we present some insights and behaviors of those pre-processing
# methods in the presence of marginal outliers.

make_plot(0)

# %%
# StandardScaler
# --------------
#
# :class:`~sklearn.preprocessing.StandardScaler` removes the mean and scales
# the data to unit variance. The scaling shrinks the range of the feature
# values as shown in the left figure below.
# However, the outliers have an influence when computing the empirical mean and
# standard deviation. Note in particular that because the outliers on each
# feature have different magnitudes, the spread of the transformed data on
# each feature is very different: most of the data lie in the [-2, 4] range for
# the transformed median income feature while the same data is squeezed in the
# smaller [-0.2, 0.2] range for the transformed average house occupancy.
#
# :class:`~sklearn.preprocessing.StandardScaler` therefore cannot guarantee
# balanced feature scales in the
# presence of outliers.

make_plot(1)

# %%
# MinMaxScaler
# ------------
#
# :class:`~sklearn.preprocessing.MinMaxScaler` rescales the data set such that
# all feature values are in
# the range [0, 1] as shown in the right panel below. However, this scaling
# compresses all inliers into the narrow range [0, 0.005] for the transformed
# average house occupancy.
#
# Both :class:`~sklearn.preprocessing.StandardScaler` and
# :class:`~sklearn.preprocessing.MinMaxScaler` are very sensitive to the
# presence of outliers.

make_plot(2)

# %%
# MaxAbsScaler
# ------------
#
# :class:`~sklearn.preprocessing.MaxAbsScaler` is similar to
# :class:`~sklearn.preprocessing.MinMaxScaler` except that the
# values are mapped across several ranges depending on whether negative
# OR positive values are present. If only positive values are present, the
# range is [0, 1]. If only negative values are present, the range is [-1, 0].
# If both negative and positive values are present, the range is [-1, 1].
# On positive only data, both :class:`~sklearn.preprocessing.MinMaxScaler`
# and :class:`~sklearn.preprocessing.MaxAbsScaler` behave similarly.
# :class:`~sklearn.preprocessing.MaxAbsScaler` therefore also suffers from
# the presence of large outliers.

make_plot(3)

# %%
# RobustScaler
# ------------
#
# Unlike the previous scalers, the centering and scaling statistics of
# :class:`~sklearn.preprocessing.RobustScaler`
# are based on percentiles and are therefore not influenced by a small
# number of very large marginal outliers. Consequently, the resulting range of
# the transformed feature values is larger than for the previous scalers and,
# more importantly, are approximately similar: for both features most of the
# transformed values lie in a [-2, 3] range as seen in the zoomed-in figure.
# Note that the outliers themselves are still present in the transformed data.
# If a separate outlier clipping is desirable, a non-linear transformation is
# required (see below).

make_plot(4)

# %%
# PowerTransformer
# ----------------
#
# :class:`~sklearn.preprocessing.PowerTransformer` applies a power
# transformation to each feature to make the data more Gaussian-like in order
# to stabilize variance and minimize skewness. Currently the Yeo-Johnson
# and Box-Cox transforms are supported and the optimal
# scaling factor is determined via maximum likelihood estimation in both
# methods. By default, :class:`~sklearn.preprocessing.PowerTransformer` applies
# zero-mean, unit variance normalization. Note that
# Box-Cox can only be applied to strictly positive data. Income and average
# house occupancy happen to be strictly positive, but if negative values are
# present the Yeo-Johnson transformed is preferred.
```
### 6 - sklearn/preprocessing/_data.py:

Start line: 3093, End line: 3125

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "method": [StrOptions({"yeo-johnson", "box-cox"})],
        "standardize": ["boolean"],
        "copy": ["boolean"],
    }

    def __init__(self, method="yeo-johnson", *, standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy

    def fit(self, X, y=None):
        """Estimate the optimal parameter lambda for each feature.

        The optimal lambda parameter for minimizing skewness is estimated on
        each feature independently using maximum likelihood.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self._validate_params()
        self._fit(X, y=y, force_transform=False)
        return self
```
### 7 - sklearn/preprocessing/_data.py:

Start line: 3127, End line: 3145

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def fit_transform(self, X, y=None):
        """Fit `PowerTransformer` to `X`, then transform `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to estimate the optimal transformation parameters
            and to be transformed using a power transformation.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        self._validate_params()
        return self._fit(X, y, force_transform=True)
```
### 8 - sklearn/preprocessing/_data.py:

Start line: 3286, End line: 3318

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _yeo_johnson_transform(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        """

        out = np.zeros_like(x)
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            out[pos] = np.log1p(x[pos])
        else:  # lmbda != 0
            out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            out[~pos] = -np.log1p(-x[~pos])

        return out

    def _box_cox_optimize(self, x):
        """Find and return optimal lambda parameter of the Box-Cox transform by
        MLE, for observed data x.

        We here use scipy builtins which uses the brent optimizer.
        """
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        _, lmbda = stats.boxcox(x[~np.isnan(x)], lmbda=None)

        return lmbda
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 834, End line: 1553

```python
def check_estimator_sparse_data(name, estimator_orig):
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(40, 3))
    X[X < 0.8] = 0
    X = _enforce_estimator_tags_X(estimator_orig, X)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.uniform(size=40)).astype(int)
    # catch deprecation warnings
    with ignore_warnings(category=FutureWarning):
        estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    tags = _safe_tags(estimator_orig)
    for matrix_format, X in _generate_sparse_matrix(X_csr):
        # catch deprecation warnings
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            if name in ["Scaler", "StandardScaler"]:
                estimator.set_params(with_mean=False)
        # fit and predict
        if "64" in matrix_format:
            err_msg = (
                f"Estimator {name} doesn't seem to support {matrix_format} "
                "matrix, and is not failing gracefully, e.g. by using "
                "check_array(X, accept_large_sparse=False)"
            )
        else:
            err_msg = (
                f"Estimator {name} doesn't seem to fail gracefully on sparse "
                "data: error message should state explicitly that sparse "
                "input is not supported if this is not the case."
            )
        with raises(
            (TypeError, ValueError),
            match=["sparse", "Sparse"],
            may_pass=True,
            err_msg=err_msg,
        ):
            with ignore_warnings(category=FutureWarning):
                estimator.fit(X, y)
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                if tags["multioutput_only"]:
                    assert pred.shape == (X.shape[0], 1)
                else:
                    assert pred.shape == (X.shape[0],)
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X)
                if tags["binary_only"]:
                    expected_probs_shape = (X.shape[0], 2)
                else:
                    expected_probs_shape = (X.shape[0], 4)
                assert probs.shape == expected_probs_shape


@ignore_warnings(category=FutureWarning)
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    estimator = clone(estimator_orig)
    try:
        import pandas as pd

        X = np.array(
            [
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]
        )
        X = pd.DataFrame(_enforce_estimator_tags_X(estimator_orig, X), copy=False)
        y = pd.Series([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
        weights = pd.Series([1] * 12)
        if _safe_tags(estimator, key="multioutput_only"):
            y = pd.DataFrame(y, copy=False)
        try:
            estimator.fit(X, y, sample_weight=weights)
        except ValueError:
            raise ValueError(
                "Estimator {0} raises error if "
                "'sample_weight' parameter is of "
                "type pandas.Series".format(name)
            )
    except ImportError:
        raise SkipTest(
            "pandas is not installed: not testing for "
            "input of type pandas.Series to class weight."
        )


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_not_an_array(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type _NotAnArray in the 'fit' function.
    estimator = clone(estimator_orig)
    X = np.array(
        [
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    )
    X = _NotAnArray(_enforce_estimator_tags_X(estimator_orig, X))
    y = _NotAnArray([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2])
    weights = _NotAnArray([1] * 12)
    if _safe_tags(estimator, key="multioutput_only"):
        y = _NotAnArray(y.data.reshape(-1, 1))
    estimator.fit(X, y, sample_weight=weights)


@ignore_warnings(category=(FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _enforce_estimator_tags_X(estimator_orig, rnd.uniform(size=(n_samples, 3)))
    y = np.arange(n_samples) % 3
    y = _enforce_estimator_tags_y(estimator, y)
    sample_weight = [3] * n_samples
    # Test that estimators don't raise any exception
    estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_shape(name, estimator_orig):
    # check that estimators raise an error if sample_weight
    # shape mismatches the input
    estimator = clone(estimator_orig)
    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ]
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2])
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y, sample_weight=np.ones(len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones(2 * len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones((len(y), 2)))


@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance(name, estimator_orig, kind="ones"):
    # For kind="ones" check that the estimators yield same results for
    # unit weights and no weights
    # For kind="zeros" check that setting sample_weight to 0 is equivalent
    # to removing corresponding samples.
    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    X1 = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)

    if kind == "ones":
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        err_msg = (
            f"For {name} sample_weight=None is not equivalent to sample_weight=ones"
        )
    elif kind == "zeros":
        # Construct a dataset that is very different to (X, y) if weights
        # are disregarded, but identical to (X, y) given weights.
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1) :] = 0
        X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)

        err_msg = (
            f"For {name}, a zero sample_weight is not equivalent to removing the sample"
        )
    else:  # pragma: no cover
        raise ValueError

    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)

    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)

    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator_orig, method):
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)


def check_sample_weights_not_overwritten(name, estimator_orig):
    # check that estimators don't override the passed sample_weight parameter
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)

    X = np.array(
        [
            [1, 3],
            [1, 3],
            [1, 3],
            [1, 3],
            [2, 1],
            [2, 1],
            [2, 1],
            [2, 1],
            [3, 3],
            [3, 3],
            [3, 3],
            [3, 3],
            [4, 1],
            [4, 1],
            [4, 1],
            [4, 1],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int)
    y = _enforce_estimator_tags_y(estimator, y)

    sample_weight_original = np.ones(y.shape[0])
    sample_weight_original[0] = 10.0

    sample_weight_fit = sample_weight_original.copy()

    estimator.fit(X, y, sample_weight=sample_weight_fit)

    err_msg = f"{name} overwrote the original `sample_weight` given during fit"
    assert_allclose(sample_weight_fit, sample_weight_original, err_msg=err_msg)


@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = _enforce_estimator_tags_X(estimator_orig, rng.uniform(size=(40, 10)))
    X = X.astype(object)
    tags = _safe_tags(estimator_orig)
    y = (X[:, 0] * 4).astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    with raises(Exception, match="Unknown label type", may_pass=True):
        estimator.fit(X, y.astype(object))

    if "string" not in tags["X_types"]:
        X[0, 0] = {"foo": "bar"}
        msg = "argument must be a string.* number"
        with raises(TypeError, match=msg):
            estimator.fit(X, y)
    else:
        # Estimators supporting string will not call np.asarray to convert the
        # data to numeric and therefore, the error will not be raised.
        # Checking for each element dtype in the input array will be costly.
        # Refer to #11401 for full discussion.
        estimator.fit(X, y)


def check_complex_data(name, estimator_orig):
    rng = np.random.RandomState(42)
    # check that estimators raise an exception on providing complex data
    X = rng.uniform(size=10) + 1j * rng.uniform(size=10)
    X = X.reshape(-1, 1)

    # Something both valid for classification and regression
    y = rng.randint(low=0, high=2, size=10) + 1j
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)
    with raises(ValueError, match="Complex data not supported"):
        estimator.fit(X, y)


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    # this estimator raises
    # ValueError: Found array with 0 feature(s) (shape=(23, 0))
    # while a minimum of 1 is required.
    # error
    if name in ["SpectralCoclustering"]:
        return
    rnd = np.random.RandomState(0)
    if name in ["RANSACRegressor"]:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))

    X = _enforce_estimator_tags_X(estimator_orig, X)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert estimator.__dict__ == dict_before, (
                "Estimator changes __dict__ during %s" % method
            )


def _is_public_parameter(attr):
    return not (attr.startswith("_") or attr.endswith("_"))


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    if hasattr(estimator_orig.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but %s added"
        % ", ".join(attrs_added_by_fit)
    )

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        " %s changed"
        % ", ".join(attrs_changed_by_fit)
    )


@ignore_warnings(category=FutureWarning)
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(
                ValueError, "Reshape your data", getattr(estimator, method), X[0]
            )


def _apply_on_subsets(func, X):
    # apply function on the whole set and on mini batches
    result_full = func(X)
    n_features = X.shape[1]
    result_by_batch = [func(batch.reshape(1, n_features)) for batch in X]

    # func can output tuple (e.g. score_samples)
    if type(result_full) == tuple:
        result_full = result_full[0]
        result_by_batch = list(map(lambda x: x[0], result_by_batch))

    if sparse.issparse(result_full):
        result_full = result_full.A
        result_by_batch = [x.A for x in result_by_batch]

    return np.ravel(result_full), np.ravel(result_by_batch)


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on mini batches or the whole set
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        msg = ("{method} of {name} is not invariant when applied to a subset.").format(
            method=method, name=name
        )

        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X
            )
            assert_allclose(result_full, result_by_batch, atol=1e-7, err_msg=msg)


@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    # check that method gives invariant results if applied
    # on a subset with different sample order
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(np.int64)
    if _safe_tags(estimator_orig, key="binary_only"):
        y[y == 2] = 1
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 2

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    idx = np.random.permutation(X.shape[0])

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        msg = (
            "{method} of {name} is not invariant when applied to a dataset"
            "with different sample order."
        ).format(method=method, name=name)

        if hasattr(estimator, method):
            assert_allclose_dense_sparse(
                getattr(estimator, method)(X)[idx],
                getattr(estimator, method)(X[idx]),
                atol=1e-9,
                err_msg=msg,
            )


@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    X = _enforce_estimator_tags_X(estimator_orig, X)

    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    # min_cluster_size cannot be less than the data size for OPTICS.
    if name == "OPTICS":
        estimator.set_params(min_samples=1.0)

    # perplexity cannot be more than the number of samples for TSNE.
    if name == "TSNE":
        estimator.set_params(perplexity=0.5)

    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]

    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == "RandomizedLogisticRegression":
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == "RANSACRegressor":
        estimator.residual_threshold = 0.5

    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = [r"1 feature\(s\)", "n_features = 1", "n_features=1"]

    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    with raises(ValueError):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X = _enforce_estimator_tags_X(transformer, X)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    _check_transformer(name, transformer, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(
        n_samples=30,
        centers=[[0, 0, 0], [1, 1, 1]],
        random_state=0,
        n_features=2,
        cluster_std=0.1,
    )
    X = StandardScaler().fit_transform(X)
    X = _enforce_estimator_tags_X(transformer, X)
    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)
    # try the same with some list
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    X, y = _regression_dataset()

    transformer = clone(transformer)
    with raises(
        (AttributeError, ValueError),
        err_msg=(
            "The unfitted "
            f"transformer {name} does not raise an error when "
            "transform is called. Perhaps use "
            "check_is_fitted in transform."
        ),
    ):
        transformer.transform(X)


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted_stateless(name, transformer):
    """Check that using transform without prior fitting
    doesn't raise a NotFittedError for stateless transformers.
    """
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(20, 5))
    X = _enforce_estimator_tags_X(transformer, X)

    transformer = clone(transformer)
    X_trans = transformer.transform(X)

    assert X_trans.shape[0] == X.shape[0]
```
### 10 - sklearn/preprocessing/_polynomial.py:

Start line: 1125, End line: 1170

```python
class SplineTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        # ... other code

        if use_sparse:
            # TODO: Remove this conditional error when the minimum supported version of
            # SciPy is 1.9.2
            # `scipy.sparse.hstack` breaks in scipy<1.9.2
            # when `n_features_out_ > max_int32`
            max_int32 = np.iinfo(np.int32).max
            all_int32 = True
            for mat in output_list:
                all_int32 &= mat.indices.dtype == np.int32
            if (
                sp_version < parse_version("1.9.2")
                and self.n_features_out_ > max_int32
                and all_int32
            ):
                raise ValueError(
                    "In scipy versions `<1.9.2`, the function `scipy.sparse.hstack`"
                    " produces negative columns when:\n1. The output shape contains"
                    " `n_cols` too large to be represented by a 32bit signed"
                    " integer.\n. All sub-matrices to be stacked have indices of"
                    " dtype `np.int32`.\nTo avoid this error, either use a version"
                    " of scipy `>=1.9.2` or alter the `SplineTransformer`"
                    " transformer to produce fewer than 2^31 output features"
                )
            XBS = sparse.hstack(output_list, format="csr")
        elif self.sparse_output:
            # TODO: Remove ones scipy 1.10 is the minimum version. See comments above.
            XBS = sparse.csr_matrix(XBS)

        if self.include_bias:
            return XBS
        else:
            # We throw away one spline basis per feature.
            # We chose the last one.
            indices = [j for j in range(XBS.shape[1]) if (j + 1) % n_splines != 0]
            return XBS[:, indices]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimators_pickle": (
                    "Current Scipy implementation of _bsplines does not"
                    "support const memory views."
                ),
            }
        }
```
### 11 - sklearn/preprocessing/_data.py:

Start line: 3178, End line: 3205

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def transform(self, X):
        """Apply the power transform to each feature using the fitted lambdas.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed using a power transformation.

        Returns
        -------
        X_trans : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_positive=True, check_shape=True)

        transform_function = {
            "box-cox": boxcox,
            "yeo-johnson": self._yeo_johnson_transform,
        }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid="ignore"):  # hide NaN warnings
                X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            X = self._scaler.transform(X)

        return X
```
### 12 - sklearn/preprocessing/_data.py:

Start line: 3254, End line: 3284

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _box_cox_inverse_tranform(self, x, lmbda):
        """Return inverse-transformed input x following Box-Cox inverse
        transform with parameter lambda.
        """
        if lmbda == 0:
            x_inv = np.exp(x)
        else:
            x_inv = (x * lmbda + 1) ** (1 / lmbda)

        return x_inv

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        """
        x_inv = np.zeros_like(x)
        pos = x >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-x[~pos])

        return x_inv
```
### 15 - sklearn/preprocessing/_data.py:

Start line: 3207, End line: 3252

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def inverse_transform(self, X):
        """Apply the inverse power transformation using the fitted lambdas.

        The inverse of the Box-Cox transformation is given by::

            if lambda_ == 0:
                X = exp(X_trans)
            else:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_)

        The inverse of the Yeo-Johnson transformation is given by::

            if X >= 0 and lambda_ == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda_ != 0:
                X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
            elif X < 0 and lambda_ != 2:
                X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
            elif X < 0 and lambda_ == 2:
                X = 1 - exp(-X_trans)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The transformed data.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The original data.
        """
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        if self.standardize:
            X = self._scaler.inverse_transform(X)

        inv_fun = {
            "box-cox": self._box_cox_inverse_tranform,
            "yeo-johnson": self._yeo_johnson_inverse_transform,
        }[self.method]
        for i, lmbda in enumerate(self.lambdas_):
            with np.errstate(invalid="ignore"):  # hide NaN warnings
                X[:, i] = inv_fun(X[:, i], lmbda)

        return X
```
### 17 - sklearn/preprocessing/_data.py:

Start line: 3320, End line: 3349

```python
class PowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _yeo_johnson_optimize(self, x):
        """Find and return optimal lambda parameter of the Yeo-Johnson
        transform by MLE, for observed data x.

        Like for Box-Cox, MLE is done via the brent optimizer.
        """
        x_tiny = np.finfo(np.float64).tiny

        def _neg_log_likelihood(lmbda):
            """Return the negative log likelihood of the observed data x as a
            function of lambda."""
            x_trans = self._yeo_johnson_transform(x, lmbda)
            n_samples = x.shape[0]
            x_trans_var = x_trans.var()

            # Reject transformed data that would raise a RuntimeWarning in np.log
            if x_trans_var < x_tiny:
                return np.inf

            log_var = np.log(x_trans_var)
            loglike = -n_samples / 2 * log_var
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # choosing bracket -2, 2 like for boxcox
        return optimize.brent(_neg_log_likelihood, brack=(-2, 2))
```
### 18 - sklearn/preprocessing/_data.py:

Start line: 11, End line: 71

```python
import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy import stats
from scipy import optimize
from scipy.special import boxcox

from ..base import (
    BaseEstimator,
    TransformerMixin,
    OneToOneFeatureMixin,
    ClassNamePrefixFeaturesOutMixin,
)
from ..utils import check_array
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs_fast import (
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)
from ..utils.sparsefuncs import (
    inplace_column_scale,
    mean_variance_axis,
    incr_mean_variance_axis,
    min_max_axis,
)
from ..utils.validation import (
    check_is_fitted,
    check_random_state,
    _check_sample_weight,
    FLOAT_DTYPES,
)

from ._encoders import OneHotEncoder


BOUNDS_THRESHOLD = 1e-7

__all__ = [
    "Binarizer",
    "KernelCenterer",
    "MinMaxScaler",
    "MaxAbsScaler",
    "Normalizer",
    "OneHotEncoder",
    "RobustScaler",
    "StandardScaler",
    "QuantileTransformer",
    "PowerTransformer",
    "add_dummy_feature",
    "binarize",
    "normalize",
    "scale",
    "robust_scale",
    "maxabs_scale",
    "minmax_scale",
    "quantile_transform",
    "power_transform",
]
```
### 37 - sklearn/preprocessing/_data.py:

Start line: 2752, End line: 2774

```python
class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _check_inputs(self, X, in_fit, accept_sparse_negative=False, copy=False):
        """Check inputs before fit and transform."""
        X = self._validate_data(
            X,
            reset=in_fit,
            accept_sparse="csc",
            copy=copy,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if (
                not accept_sparse_negative
                and not self.ignore_implicit_zeros
                and (sparse.issparse(X) and np.any(X.data < 0))
            ):
                raise ValueError(
                    "QuantileTransformer only accepts non-negative sparse matrices."
                )

        return X
```
### 46 - sklearn/preprocessing/_data.py:

Start line: 2855, End line: 2995

```python
@validate_params(
    {"X": ["array-like", "sparse matrix"], "axis": [Options(Integral, {0, 1})]}
)
def quantile_transform(
    X,
    *,
    axis=0,
    n_quantiles=1000,
    output_distribution="uniform",
    ignore_implicit_zeros=False,
    subsample=int(1e5),
    random_state=None,
    copy=True,
):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to transform.

    axis : int, default=0
        Axis used to compute the means and standard deviations along. If 0,
        transform each feature, otherwise (if 1) transform each sample.

    n_quantiles : int, default=1000 or n_samples
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.
        If n_quantiles is larger than the number of samples, n_quantiles is set
        to the number of samples as a larger number of quantiles does not give
        a better approximation of the cumulative distribution function
        estimator.

    output_distribution : {'uniform', 'normal'}, default='uniform'
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, default=False
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, default=1e5
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling and smoothing
        noise.
        Please see ``subsample`` for more details.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    copy : bool, default=True
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array). If True, a copy of `X` is transformed,
        leaving the original `X` unchanged.

        .. versionchanged:: 0.23
            The default value of `copy` changed from False to True in 0.23.

    Returns
    -------
    Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.

    See Also
    --------
    QuantileTransformer : Performs quantile-based scaling using the
        Transformer API (e.g. as part of a preprocessing
        :class:`~sklearn.pipeline.Pipeline`).
    power_transform : Maps data to a normal distribution using a
        power transformation.
    scale : Performs standardization that is faster, but less robust
        to outliers.
    robust_scale : Performs robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    .. warning:: Risk of data leak

        Do not use :func:`~sklearn.preprocessing.quantile_transform` unless
        you know what you are doing. A common mistake is to apply it
        to the entire data *before* splitting into training and
        test sets. This will bias the model evaluation because
        information would have leaked from the test set to the
        training set.
        In general, we recommend using
        :class:`~sklearn.preprocessing.QuantileTransformer` within a
        :ref:`Pipeline <pipeline>` in order to prevent most risks of data
        leaking:`pipe = make_pipeline(QuantileTransformer(),
        LogisticRegression())`.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> quantile_transform(X, n_quantiles=10, random_state=0, copy=True)
    array([...])
    """
    n = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=output_distribution,
        subsample=subsample,
        ignore_implicit_zeros=ignore_implicit_zeros,
        random_state=random_state,
        copy=copy,
    )
    if axis == 0:
        X = n.fit_transform(X)
    else:  # axis == 1
        X = n.fit_transform(X.T).T
    return X
```
### 47 - sklearn/preprocessing/_data.py:

Start line: 902, End line: 987

```python
class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def partial_fit(self, X, y=None, sample_weight=None):
        # ... other code

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            sparse_constructor = (
                sparse.csr_matrix if X.format == "csr" else sparse.csc_matrix
            )

            if self.with_std:
                # First pass
                if not hasattr(self, "scale_"):
                    self.mean_, self.var_, self.n_samples_seen_ = mean_variance_axis(
                        X, axis=0, weights=sample_weight, return_sum_weights=True
                    )
                # Next passes
                else:
                    (
                        self.mean_,
                        self.var_,
                        self.n_samples_seen_,
                    ) = incr_mean_variance_axis(
                        X,
                        axis=0,
                        last_mean=self.mean_,
                        last_var=self.var_,
                        last_n=self.n_samples_seen_,
                        weights=sample_weight,
                    )
                # We force the mean and variance to float64 for large arrays
                # See https://github.com/scikit-learn/scikit-learn/pull/12338
                self.mean_ = self.mean_.astype(np.float64, copy=False)
                self.var_ = self.var_.astype(np.float64, copy=False)
            else:
                self.mean_ = None  # as with_mean must be False for sparse
                self.var_ = None
                weights = _check_sample_weight(sample_weight, X)
                sum_weights_nan = weights @ sparse_constructor(
                    (np.isnan(X.data), X.indices, X.indptr), shape=X.shape
                )
                self.n_samples_seen_ += (np.sum(weights) - sum_weights_nan).astype(
                    dtype
                )
        else:
            # First pass
            if not hasattr(self, "scale_"):
                self.mean_ = 0.0
                if self.with_std:
                    self.var_ = 0.0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - np.isnan(X).sum(axis=0)

            else:
                self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(
                    X,
                    self.mean_,
                    self.var_,
                    self.n_samples_seen_,
                    sample_weight=sample_weight,
                )

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if np.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            # Extract the list of near constant features on the raw variances,
            # before taking the square root.
            constant_mask = _is_constant_feature(
                self.var_, self.mean_, self.n_samples_seen_
            )
            self.scale_ = _handle_zeros_in_scale(
                np.sqrt(self.var_), copy=False, constant_mask=constant_mask
            )
        else:
            self.scale_ = None

        return self
```
### 57 - sklearn/preprocessing/_data.py:

Start line: 2807, End line: 2826

```python
class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The projected data.
        """
        check_is_fitted(self)
        X = self._check_inputs(X, in_fit=False, copy=self.copy)

        return self._transform(X, inverse=False)
```
### 67 - sklearn/preprocessing/_data.py:

Start line: 2592, End line: 2635

```python
class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative. If a sparse matrix is provided,
            it will be converted into a sparse ``csc_matrix``.
        """
        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for feature_idx in range(n_features):
            column_nnz_data = X.data[X.indptr[feature_idx] : X.indptr[feature_idx + 1]]
            if len(column_nnz_data) > self.subsample:
                column_subsample = self.subsample * len(column_nnz_data) // n_samples
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample, dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                column_data[:column_subsample] = random_state.choice(
                    column_nnz_data, size=column_subsample, replace=False
                )
            else:
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data), dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                column_data[: len(column_nnz_data)] = column_nnz_data

            if not column_data.size:
                # if no nnz, an error will be raised for computing the
                # quantiles. Force the quantiles to be zeros.
                self.quantiles_.append([0] * len(references))
            else:
                self.quantiles_.append(np.nanpercentile(column_data, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # due to floating-point precision error in `np.nanpercentile`,
        # make sure the quantiles are monotonically increasing
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)
```
### 69 - sklearn/preprocessing/_data.py:

Start line: 2534, End line: 2558

```python
class QuantileTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "output_distribution": [StrOptions({"uniform", "normal"})],
        "ignore_implicit_zeros": ["boolean"],
        "subsample": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        *,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=10_000,
        random_state=None,
        copy=True,
    ):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
```

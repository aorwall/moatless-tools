# scikit-learn__scikit-learn-12462

| **scikit-learn/scikit-learn** | `9ec5a15823dcb924a5cca322f9f97357f9428345` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 3 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -140,7 +140,12 @@ def _num_samples(x):
         if len(x.shape) == 0:
             raise TypeError("Singleton array %r cannot be considered"
                             " a valid collection." % x)
-        return x.shape[0]
+        # Check that shape is returning an integer or default to len
+        # Dask dataframes may not return numeric shape[0] value
+        if isinstance(x.shape[0], numbers.Integral):
+            return x.shape[0]
+        else:
+            return len(x)
     else:
         return len(x)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/validation.py | 143 | 143 | - | 3 | -


## Problem Statement

```
SkLearn `.score()` method generating error with Dask DataFrames
When using Dask Dataframes with SkLearn, I used to be able to just ask SkLearn for the score of any given algorithm. It would spit out a nice answer and I'd move on. After updating to the newest versions, all metrics that compute based on (y_true, y_predicted) are failing. I've tested `accuracy_score`, `precision_score`, `r2_score`, and `mean_squared_error.` Work-around shown below, but it's not ideal because it requires me to cast from Dask Arrays to numpy arrays which won't work if the data is huge.

I've asked Dask about it here: https://github.com/dask/dask/issues/4137 and they've said it's an issue with the SkLearn `shape` check, and that they won't be addressing it. It seems like it should be not super complicated to add a `try-except` that says "if shape doesn't return a tuple revert to pretending shape didn't exist". If others think that sounds correct, I can attempt a pull-request, but I don't want to attempt to solve it on my own only to find out others don't deem that an acceptable solutions.

Trace, MWE, versions, and workaround all in-line.

MWE:

\`\`\`
import dask.dataframe as dd
from sklearn.linear_model import LinearRegression, SGDRegressor

df = dd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
lr = LinearRegression()
X = df.drop('quality', axis=1)
y = df['quality']

lr.fit(X,y)
lr.score(X,y)
\`\`\`

Output of error:

\`\`\`
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-5-4eafa0e7fc85> in <module>
      8 
      9 lr.fit(X,y)
---> 10 lr.score(X,y)

~/anaconda3/lib/python3.6/site-packages/sklearn/base.py in score(self, X, y, sample_weight)
    327         from .metrics import r2_score
    328         return r2_score(y, self.predict(X), sample_weight=sample_weight,
--> 329                         multioutput='variance_weighted')
    330 
    331 

~/anaconda3/lib/python3.6/site-packages/sklearn/metrics/regression.py in r2_score(y_true, y_pred, sample_weight, multioutput)
    532     """
    533     y_type, y_true, y_pred, multioutput = _check_reg_targets(
--> 534         y_true, y_pred, multioutput)
    535     check_consistent_length(y_true, y_pred, sample_weight)
    536 

~/anaconda3/lib/python3.6/site-packages/sklearn/metrics/regression.py in _check_reg_targets(y_true, y_pred, multioutput)
     73 
     74     """
---> 75     check_consistent_length(y_true, y_pred)
     76     y_true = check_array(y_true, ensure_2d=False)
     77     y_pred = check_array(y_pred, ensure_2d=False)

~/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    225 
    226     lengths = [_num_samples(X) for X in arrays if X is not None]
--> 227     uniques = np.unique(lengths)
    228     if len(uniques) > 1:
    229         raise ValueError("Found input variables with inconsistent numbers of"

~/anaconda3/lib/python3.6/site-packages/numpy/lib/arraysetops.py in unique(ar, return_index, return_inverse, return_counts, axis)
    229 
    230     """
--> 231     ar = np.asanyarray(ar)
    232     if axis is None:
    233         ret = _unique1d(ar, return_index, return_inverse, return_counts)

~/anaconda3/lib/python3.6/site-packages/numpy/core/numeric.py in asanyarray(a, dtype, order)
    551 
    552     """
--> 553     return array(a, dtype, copy=False, order=order, subok=True)
    554 
    555 

TypeError: int() argument must be a string, a bytes-like object or a number, not 'Scalar'
\`\`\`

Problem occurs after upgrading as follows:

Before bug:
\`\`\`
for lib in (sklearn, dask):
    print(f'{lib.__name__} Version: {lib.__version__}')
> sklearn Version: 0.19.1
> dask Version: 0.18.2
\`\`\`

Update from conda, then bug starts:
\`\`\`
for lib in (sklearn, dask):
    print(f'{lib.__name__} Version: {lib.__version__}')
> sklearn Version: 0.20.0
> dask Version: 0.19.4
\`\`\`

Work around:

\`\`\`
from sklearn.metrics import r2_score
preds = lr.predict(X_test)
r2_score(np.array(y_test), np.array(preds))
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 sklearn/utils/fixes.py | 1 | 96| 646 | 646 | 2564 | 
| 2 | 2 sklearn/metrics/__init__.py | 72 | 133| 400 | 1046 | 3492 | 
| 3 | **3 sklearn/utils/validation.py** | 740 | 763| 276 | 1322 | 11705 | 
| 4 | 4 sklearn/utils/estimator_checks.py | 1617 | 1646| 324 | 1646 | 32698 | 
| 5 | **4 sklearn/utils/validation.py** | 507 | 570| 747 | 2393 | 32698 | 
| 6 | 5 sklearn/metrics/scorer.py | 454 | 531| 836 | 3229 | 37076 | 
| 7 | 6 sklearn/metrics/regression.py | 534 | 576| 432 | 3661 | 42305 | 
| 8 | 7 sklearn/__init__.py | 1 | 80| 663 | 4324 | 43101 | 
| 9 | 7 sklearn/utils/estimator_checks.py | 483 | 529| 467 | 4791 | 43101 | 
| 10 | 7 sklearn/utils/estimator_checks.py | 1 | 81| 709 | 5500 | 43101 | 
| 11 | 7 sklearn/utils/estimator_checks.py | 1049 | 1074| 282 | 5782 | 43101 | 
| 12 | **7 sklearn/utils/validation.py** | 571 | 605| 429 | 6211 | 43101 | 
| 13 | 7 sklearn/utils/estimator_checks.py | 606 | 641| 328 | 6539 | 43101 | 
| 14 | **7 sklearn/utils/validation.py** | 451 | 505| 549 | 7088 | 43101 | 
| 15 | 7 sklearn/utils/estimator_checks.py | 1397 | 1494| 1021 | 8109 | 43101 | 
| 16 | 8 sklearn/metrics/classification.py | 44 | 112| 567 | 8676 | 62000 | 
| 17 | 8 sklearn/metrics/__init__.py | 1 | 70| 527 | 9203 | 62000 | 
| 18 | 8 sklearn/metrics/scorer.py | 328 | 385| 538 | 9741 | 62000 | 
| 19 | 9 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 11055 | 63341 | 
| 20 | 9 sklearn/metrics/regression.py | 1 | 41| 148 | 11203 | 63341 | 
| 21 | 9 sklearn/utils/fixes.py | 313 | 335| 208 | 11411 | 63341 | 
| 22 | 9 sklearn/utils/estimator_checks.py | 1015 | 1046| 346 | 11757 | 63341 | 
| 23 | 10 benchmarks/bench_saga.py | 6 | 99| 809 | 12566 | 65366 | 
| 24 | 10 benchmarks/bench_saga.py | 102 | 177| 616 | 13182 | 65366 | 
| 25 | 10 sklearn/utils/estimator_checks.py | 1123 | 1191| 601 | 13783 | 65366 | 
| 26 | 10 sklearn/metrics/regression.py | 449 | 533| 746 | 14529 | 65366 | 
| 27 | 10 sklearn/metrics/regression.py | 44 | 110| 591 | 15120 | 65366 | 
| 28 | 10 sklearn/utils/estimator_checks.py | 1759 | 1801| 474 | 15594 | 65366 | 
| 29 | 11 benchmarks/bench_glmnet.py | 47 | 129| 796 | 16390 | 66453 | 
| 30 | 11 sklearn/utils/estimator_checks.py | 2019 | 2033| 211 | 16601 | 66453 | 
| 31 | 12 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 17112 | 66986 | 
| 32 | 13 examples/preprocessing/plot_scaling_importance.py | 83 | 135| 457 | 17569 | 68213 | 
| 33 | 13 sklearn/utils/estimator_checks.py | 739 | 761| 221 | 17790 | 68213 | 
| 34 | 13 sklearn/utils/estimator_checks.py | 2133 | 2174| 400 | 18190 | 68213 | 
| 35 | 13 sklearn/utils/estimator_checks.py | 938 | 1012| 702 | 18892 | 68213 | 
| 36 | 13 sklearn/utils/estimator_checks.py | 847 | 878| 297 | 19189 | 68213 | 
| 37 | 14 examples/plot_missing_values.py | 34 | 78| 447 | 19636 | 69292 | 
| 38 | 14 examples/preprocessing/plot_scaling_importance.py | 1 | 82| 762 | 20398 | 69292 | 
| 39 | 14 sklearn/utils/estimator_checks.py | 2036 | 2056| 200 | 20598 | 69292 | 
| 40 | 14 sklearn/utils/estimator_checks.py | 1077 | 1097| 236 | 20834 | 69292 | 
| 41 | 15 examples/preprocessing/plot_all_scaling.py | 1 | 107| 792 | 21626 | 72408 | 
| 42 | 15 sklearn/utils/estimator_checks.py | 1828 | 1870| 449 | 22075 | 72408 | 
| 43 | 16 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 22607 | 74353 | 
| 44 | 16 sklearn/utils/estimator_checks.py | 1497 | 1562| 567 | 23174 | 74353 | 
| 45 | 17 sklearn/metrics/cluster/bicluster.py | 1 | 29| 237 | 23411 | 75079 | 
| 46 | 17 sklearn/metrics/scorer.py | 1 | 44| 327 | 23738 | 75079 | 
| 47 | 18 examples/model_selection/plot_precision_recall.py | 100 | 205| 770 | 24508 | 77432 | 
| 48 | 19 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 25141 | 78158 | 
| 49 | 20 sklearn/linear_model/base.py | 169 | 183| 155 | 25296 | 82785 | 
| 50 | 20 sklearn/utils/estimator_checks.py | 1731 | 1756| 277 | 25573 | 82785 | 
| 51 | 21 examples/compose/plot_transformed_target.py | 1 | 94| 759 | 26332 | 84546 | 
| 52 | 22 examples/cluster/plot_adjusted_for_chance_measures.py | 58 | 123| 542 | 26874 | 85539 | 
| 53 | 23 sklearn/linear_model/least_angle.py | 207 | 365| 1584 | 28458 | 99535 | 
| 54 | 23 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 28831 | 99535 | 
| 55 | 24 examples/plot_kernel_ridge_regression.py | 155 | 172| 194 | 29025 | 101262 | 
| 56 | 25 sklearn/model_selection/_validation.py | 483 | 563| 803 | 29828 | 114275 | 
| 57 | 25 sklearn/utils/estimator_checks.py | 2268 | 2296| 282 | 30110 | 114275 | 
| 58 | 26 examples/classification/plot_lda.py | 41 | 72| 325 | 30435 | 114866 | 
| 59 | 27 examples/plot_anomaly_comparison.py | 79 | 151| 763 | 31198 | 116371 | 
| 60 | 28 sklearn/metrics/cluster/unsupervised.py | 240 | 296| 490 | 31688 | 119570 | 
| 61 | 28 examples/preprocessing/plot_all_scaling.py | 221 | 312| 815 | 32503 | 119570 | 
| 62 | 28 sklearn/utils/estimator_checks.py | 819 | 844| 234 | 32737 | 119570 | 
| 63 | 29 examples/model_selection/plot_multi_metric_evaluation.py | 72 | 97| 317 | 33054 | 120451 | 
| 64 | 30 sklearn/metrics/ranking.py | 1 | 38| 177 | 33231 | 128557 | 
| 65 | 30 sklearn/metrics/cluster/unsupervised.py | 299 | 353| 451 | 33682 | 128557 | 
| 66 | 31 examples/text/plot_document_classification_20newsgroups.py | 250 | 326| 662 | 34344 | 131103 | 
| 67 | 32 sklearn/utils/_scipy_sparse_lsqr_backport.py | 349 | 491| 1496 | 35840 | 136262 | 
| 68 | 32 sklearn/metrics/classification.py | 1137 | 1170| 389 | 36229 | 136262 | 
| 69 | 32 sklearn/utils/fixes.py | 99 | 212| 798 | 37027 | 136262 | 
| 70 | 33 benchmarks/bench_multilabel_metrics.py | 136 | 191| 543 | 37570 | 137913 | 
| 71 | 34 sklearn/linear_model/stochastic_gradient.py | 7 | 43| 338 | 37908 | 152307 | 
| 72 | 34 sklearn/linear_model/least_angle.py | 367 | 488| 1218 | 39126 | 152307 | 
| 73 | 34 benchmarks/bench_multilabel_metrics.py | 1 | 37| 239 | 39365 | 152307 | 
| 74 | 34 sklearn/utils/estimator_checks.py | 913 | 935| 248 | 39613 | 152307 | 
| 75 | 35 benchmarks/bench_sparsify.py | 83 | 106| 153 | 39766 | 153214 | 
| 76 | 35 examples/compose/plot_transformed_target.py | 96 | 174| 747 | 40513 | 153214 | 
| 77 | 36 sklearn/metrics/pairwise.py | 60 | 127| 664 | 41177 | 166230 | 
| 78 | 36 sklearn/utils/estimator_checks.py | 1269 | 1327| 600 | 41777 | 166230 | 
| 79 | 37 examples/compose/plot_column_transformer_mixed_types.py | 1 | 106| 807 | 42584 | 167059 | 
| 80 | 37 sklearn/metrics/classification.py | 1527 | 1597| 682 | 43266 | 167059 | 
| 81 | 38 benchmarks/bench_tree.py | 64 | 125| 523 | 43789 | 167929 | 
| 82 | 39 sklearn/externals/_arff.py | 265 | 365| 798 | 44587 | 176192 | 
| 83 | 40 sklearn/linear_model/coordinate_descent.py | 8 | 29| 155 | 44742 | 197295 | 
| 84 | 41 benchmarks/bench_mnist.py | 85 | 106| 314 | 45056 | 199023 | 


### Hint

```
Some context: dask DataFrame doesn't know it's length. Previously, it didn't have a `shape` attribute.

Now dask DataFrame has a shape that returns a `Tuple[Delayed, int]` for the number of rows and columns.

> Work-around shown below, but it's not ideal because it requires me to cast from Dask Arrays to numpy arrays which won't work if the data is huge.

FYI @ZWMiller that's exactly what was occurring previously. Personally, I don't think relying on this is a good idea, for exactly the reason you state.

In `_num_samples` scikit-learn simply checks whether the array-like has a `'shape'` attribute, and then assumes that it's an int from there on. The potential fix would be slightly stricter duck typing. Checking something like `hasattr(x, 'shape') and isinstance(x.shape[0], int)` or `numbers.Integral`.

\`\`\`python
if hasattr(x, 'shape') and isinstance(x.shape[0], int):
    ...
else:
    return len(x) 
\`\`\`
```

## Patch

```diff
diff --git a/sklearn/utils/validation.py b/sklearn/utils/validation.py
--- a/sklearn/utils/validation.py
+++ b/sklearn/utils/validation.py
@@ -140,7 +140,12 @@ def _num_samples(x):
         if len(x.shape) == 0:
             raise TypeError("Singleton array %r cannot be considered"
                             " a valid collection." % x)
-        return x.shape[0]
+        # Check that shape is returning an integer or default to len
+        # Dask dataframes may not return numeric shape[0] value
+        if isinstance(x.shape[0], numbers.Integral):
+            return x.shape[0]
+        else:
+            return len(x)
     else:
         return len(x)
 

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_validation.py b/sklearn/utils/tests/test_validation.py
--- a/sklearn/utils/tests/test_validation.py
+++ b/sklearn/utils/tests/test_validation.py
@@ -41,6 +41,7 @@
     check_memory,
     check_non_negative,
     LARGE_SPARSE_SUPPORTED,
+    _num_samples
 )
 import sklearn
 
@@ -786,3 +787,15 @@ def test_check_X_y_informative_error():
     X = np.ones((2, 2))
     y = None
     assert_raise_message(ValueError, "y cannot be None", check_X_y, X, y)
+
+
+def test_retrieve_samples_from_non_standard_shape():
+    class TestNonNumericShape:
+        def __init__(self):
+            self.shape = ("not numeric",)
+
+        def __len__(self):
+            return len([1, 2, 3])
+
+    X = TestNonNumericShape()
+    assert _num_samples(X) == len(X)

```


## Code snippets

### 1 - sklearn/utils/fixes.py:

Start line: 1, End line: 96

```python
"""Compatibility fixes for older version of python, numpy and scipy

If you add content to this file, please give the version of the package
at which the fixe is no longer needed.
"""

import os
import errno
import sys

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
PY3_OR_LATER = sys.version_info[0] >= 3


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


# boxcox ignore NaN in scipy.special.boxcox after 0.14
if sp_version < (0, 14):
    from scipy import stats

    def boxcox(x, lmbda):
        with np.errstate(invalid='ignore'):
            return stats.boxcox(x, lmbda)
else:
    from scipy.special import boxcox  # noqa


if sp_version < (0, 15):
    # Backport fix for scikit-learn/scikit-learn#2986 / scipy/scipy#4142
    from ._scipy_sparse_lsqr_backport import lsqr as sparse_lsqr
else:
    from scipy.sparse.linalg import lsqr as sparse_lsqr  # noqa


try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa
```
### 2 - sklearn/metrics/__init__.py:

Start line: 72, End line: 133

```python
__all__ = [
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'balanced_accuracy_score',
    'calinski_harabaz_score',
    'calinski_harabasz_score',
    'check_scoring',
    'classification_report',
    'cluster',
    'cohen_kappa_score',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'davies_bouldin_score',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'fowlkes_mallows_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'max_error',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_distances_chunked',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
]
```
### 3 - sklearn/utils/validation.py:

Start line: 740, End line: 763

```python
def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    if y is None:
        raise ValueError("y cannot be None")

    X = check_array(X, accept_sparse=accept_sparse,
                    accept_large_sparse=accept_large_sparse,
                    dtype=dtype, order=order, copy=copy,
                    force_all_finite=force_all_finite,
                    ensure_2d=ensure_2d, allow_nd=allow_nd,
                    ensure_min_samples=ensure_min_samples,
                    ensure_min_features=ensure_min_features,
                    warn_on_dtype=warn_on_dtype,
                    estimator=estimator)
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
### 4 - sklearn/utils/estimator_checks.py:

Start line: 1617, End line: 1646

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_2d(name, estimator_orig):
    if "MultiTask" in name:
        # These only work on 2d, so this test makes no sense
        return
    rnd = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)), estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % (
        ", ".join([str(w_x) for w_x in w]))
    if name not in MULTI_OUTPUT:
        # check that we warned if we don't support multi-output
        assert_greater(len(w), 0, msg)
        assert_true("DataConversionWarning('A column-vector y"
                    " was passed when a 1d array was expected" in msg)
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())
```
### 5 - sklearn/utils/validation.py:

Start line: 507, End line: 570

```python
def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=False, estimator=None):
    # ... other code

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
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

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    # ... other code
```
### 6 - sklearn/metrics/scorer.py:

Start line: 454, End line: 531

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
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

# Score functions that need decision values
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)

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
### 7 - sklearn/metrics/regression.py:

Start line: 534, End line: 576

```python
def r2_score(y_true, y_pred, sample_weight=None,
             multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)

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
    if isinstance(multioutput, string_types):
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
### 8 - sklearn/__init__.py:

Start line: 1, End line: 80

```python
"""
Machine learning module for Python
==================================

sklearn is a Python module integrating classical machine
learning algorithms in the tightly-knit world of scientific Python
packages (numpy, scipy, matplotlib).

It aims to provide simple and efficient solutions to learning problems
that are accessible to everybody and reusable in various contexts:
machine-learning as a versatile tool for science and engineering.

See http://scikit-learn.org for complete documentation.
"""
import sys
import re
import warnings
import logging

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.21.dev0'


try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __SKLEARN_SETUP__
except NameError:
    __SKLEARN_SETUP__ = False

if __SKLEARN_SETUP__:
    sys.stderr.write('Partial import of sklearn during the build process.\n')
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import __check_build
    from .base import clone
    from .utils._show_versions import show_versions

    __check_build  # avoid flakes unused variable error

    __all__ = ['calibration', 'cluster', 'covariance', 'cross_decomposition',
               'datasets', 'decomposition', 'dummy', 'ensemble', 'exceptions',
               'externals', 'feature_extraction', 'feature_selection',
               'gaussian_process', 'isotonic', 'kernel_approximation',
               'kernel_ridge', 'linear_model', 'manifold', 'metrics',
               'mixture', 'model_selection', 'multiclass', 'multioutput',
               'naive_bayes', 'neighbors', 'neural_network', 'pipeline',
               'preprocessing', 'random_projection', 'semi_supervised',
               'svm', 'tree', 'discriminant_analysis', 'impute', 'compose',
               # Non-modules:
               'clone', 'get_config', 'set_config', 'config_context',
               'show_versions']
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 483, End line: 529

```python
def check_estimator_sparse_data(name, estimator_orig):

    rng = np.random.RandomState(0)
    X = rng.rand(40, 10)
    X[X < .8] = 0
    X = pairwise_estimator_convert_X(X, estimator_orig)
    X_csr = sparse.csr_matrix(X)
    y = (4 * rng.rand(40)).astype(np.int)
    # catch deprecation warnings
    with ignore_warnings(category=DeprecationWarning):
        estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    for matrix_format, X in _generate_sparse_matrix(X_csr):
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            if name in ['Scaler', 'StandardScaler']:
                estimator = clone(estimator).set_params(with_mean=False)
            else:
                estimator = clone(estimator)
        # fit and predict
        try:
            with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
                estimator.fit(X, y)
            if hasattr(estimator, "predict"):
                pred = estimator.predict(X)
                assert_equal(pred.shape, (X.shape[0],))
            if hasattr(estimator, 'predict_proba'):
                probs = estimator.predict_proba(X)
                assert_equal(probs.shape, (X.shape[0], 4))
        except (TypeError, ValueError) as e:
            if 'sparse' not in repr(e).lower():
                if "64" in matrix_format:
                    msg = ("Estimator %s doesn't seem to support %s matrix, "
                           "and is not failing gracefully, e.g. by using "
                           "check_array(X, accept_large_sparse=False)")
                    raise AssertionError(msg % (name, matrix_format))
                else:
                    print("Estimator %s doesn't seem to fail gracefully on "
                          "sparse data: error message state explicitly that "
                          "sparse input is not supported if this is not"
                          " the case." % name)
                    raise
        except Exception:
            print("Estimator %s doesn't seem to fail gracefully on "
                  "sparse data: it should raise a TypeError if sparse input "
                  "is explicitly not supported." % name)
            raise
```
### 10 - sklearn/utils/estimator_checks.py:

Start line: 1, End line: 81

```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
from functools import partial

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from sklearn.externals.six.moves import zip
from sklearn.utils import IS_PYPY, _IS_32BIT
from sklearn.utils._joblib import hash, Memory
from sklearn.utils.testing import assert_raises, _get_args
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import META_ESTIMATORS
from sklearn.utils.testing import set_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import SkipTest
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_dict_equal
from sklearn.utils.testing import create_memmap_backed_data
from sklearn.utils import is_scalar_nan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, ClusterMixin,
                          is_classifier, is_regressor, is_outlier_detector)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.fixes import signature
from sklearn.utils.validation import (has_fit_parameter, _num_samples,
                                      LARGE_SPARSE_SUPPORTED)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']

ALLOW_NAN = ['Imputer', 'SimpleImputer', 'MissingIndicator',
             'MaxAbsScaler', 'MinMaxScaler', 'RobustScaler', 'StandardScaler',
             'PowerTransformer', 'QuantileTransformer']
```
### 12 - sklearn/utils/validation.py:

Start line: 571, End line: 605

```python
def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=False, estimator=None):
    # ... other code
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

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        warnings.warn(msg, DataConversionWarning, stacklevel=3)

    return array
```
### 14 - sklearn/utils/validation.py:

Start line: 451, End line: 505

```python
def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=False, estimator=None):
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array, "__array__"):
        dtypes_orig = np.array(array.dtypes)

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

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

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

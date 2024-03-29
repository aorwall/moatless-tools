# scikit-learn__scikit-learn-13584

| **scikit-learn/scikit-learn** | `0e3c1879b06d839171b7d0a607d71bbb19a966a9` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | - |
| **Avg pos** | - |
| **Min pos** | - |
| **Max pos** | - |
| **Top file pos** | 20 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/utils/_pprint.py b/sklearn/utils/_pprint.py
--- a/sklearn/utils/_pprint.py
+++ b/sklearn/utils/_pprint.py
@@ -95,7 +95,7 @@ def _changed_params(estimator):
     init_params = signature(init_func).parameters
     init_params = {name: param.default for name, param in init_params.items()}
     for k, v in params.items():
-        if (v != init_params[k] and
+        if (repr(v) != repr(init_params[k]) and
                 not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
             filtered_params[k] = v
     return filtered_params

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/_pprint.py | 98 | 98 | - | 20 | -


## Problem Statement

```
bug in print_changed_only in new repr: vector values
\`\`\`python
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
sklearn.set_config(print_changed_only=True)
print(LogisticRegressionCV(Cs=np.array([0.1, 1])))
\`\`\`
> ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

ping @NicolasHug 


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/plot_changed_only_pprint_parameter.py | 1 | 31| 209 | 209 | 209 | 
| 2 | 2 sklearn/utils/estimator_checks.py | 2092 | 2106| 211 | 420 | 22106 | 
| 3 | 2 sklearn/utils/estimator_checks.py | 880 | 911| 297 | 717 | 22106 | 
| 4 | 3 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 497 | 1214 | 23135 | 
| 5 | 4 sklearn/utils/validation.py | 693 | 716| 276 | 1490 | 31204 | 
| 6 | 4 sklearn/utils/validation.py | 470 | 539| 810 | 2300 | 31204 | 
| 7 | 4 sklearn/utils/estimator_checks.py | 1151 | 1219| 601 | 2901 | 31204 | 
| 8 | 5 sklearn/metrics/classification.py | 2304 | 2320| 140 | 3041 | 53514 | 
| 9 | 5 sklearn/utils/estimator_checks.py | 660 | 711| 436 | 3477 | 53514 | 
| 10 | 5 sklearn/utils/validation.py | 541 | 567| 356 | 3833 | 53514 | 
| 11 | 5 sklearn/utils/estimator_checks.py | 493 | 541| 491 | 4324 | 53514 | 
| 12 | 5 sklearn/utils/estimator_checks.py | 1423 | 1522| 940 | 5264 | 53514 | 
| 13 | 6 sklearn/utils/multiclass.py | 105 | 152| 403 | 5667 | 57390 | 
| 14 | 6 sklearn/utils/estimator_checks.py | 852 | 877| 234 | 5901 | 57390 | 
| 15 | 7 sklearn/linear_model/coordinate_descent.py | 8 | 27| 149 | 6050 | 78485 | 
| 16 | 7 sklearn/utils/estimator_checks.py | 1718 | 1759| 417 | 6467 | 78485 | 
| 17 | 7 sklearn/utils/estimator_checks.py | 1 | 62| 466 | 6933 | 78485 | 
| 18 | 8 sklearn/linear_model/stochastic_gradient.py | 7 | 45| 361 | 7294 | 92413 | 
| 19 | 8 sklearn/utils/estimator_checks.py | 1800 | 1825| 277 | 7571 | 92413 | 
| 20 | 9 sklearn/model_selection/_search.py | 358 | 375| 158 | 7729 | 105780 | 
| 21 | 10 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 8269 | 107304 | 
| 22 | 11 benchmarks/bench_rcv1_logreg_convergence.py | 6 | 31| 199 | 8468 | 109249 | 
| 23 | 11 sklearn/utils/estimator_checks.py | 140 | 160| 202 | 8670 | 109249 | 
| 24 | 11 sklearn/utils/estimator_checks.py | 1897 | 1941| 472 | 9142 | 109249 | 
| 25 | 11 sklearn/utils/estimator_checks.py | 1873 | 1894| 214 | 9356 | 109249 | 
| 26 | 11 sklearn/utils/validation.py | 423 | 468| 458 | 9814 | 109249 | 
| 27 | 11 sklearn/utils/estimator_checks.py | 1372 | 1403| 258 | 10072 | 109249 | 
| 28 | 12 sklearn/linear_model/logistic.py | 1929 | 1950| 238 | 10310 | 131291 | 
| 29 | 13 benchmarks/bench_20newsgroups.py | 1 | 97| 778 | 11088 | 132069 | 
| 30 | 14 sklearn/utils/fixes.py | 158 | 197| 380 | 11468 | 133909 | 
| 31 | 15 examples/applications/plot_out_of_core_classification.py | 187 | 214| 241 | 11709 | 137214 | 
| 32 | 15 sklearn/utils/estimator_checks.py | 1828 | 1870| 469 | 12178 | 137214 | 
| 33 | 15 sklearn/utils/estimator_checks.py | 1545 | 1617| 655 | 12833 | 137214 | 
| 34 | 15 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 65 | 123| 525 | 13358 | 137214 | 
| 35 | 15 sklearn/utils/estimator_checks.py | 2109 | 2129| 200 | 13558 | 137214 | 
| 36 | 15 sklearn/utils/estimator_checks.py | 2348 | 2377| 294 | 13852 | 137214 | 
| 37 | 15 sklearn/utils/estimator_checks.py | 813 | 849| 352 | 14204 | 137214 | 
| 38 | 16 sklearn/mixture/gaussian_mixture.py | 76 | 94| 180 | 14384 | 143498 | 
| 39 | 17 benchmarks/bench_mnist.py | 84 | 105| 314 | 14698 | 145216 | 
| 40 | 18 sklearn/linear_model/ransac.py | 1 | 19| 111 | 14809 | 149281 | 
| 41 | 18 sklearn/utils/estimator_checks.py | 163 | 180| 164 | 14973 | 149281 | 
| 42 | 18 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 15505 | 149281 | 
| 43 | 18 sklearn/linear_model/logistic.py | 463 | 483| 217 | 15722 | 149281 | 
| 44 | 18 sklearn/utils/estimator_checks.py | 914 | 933| 167 | 15889 | 149281 | 
| 45 | 18 sklearn/linear_model/logistic.py | 2213 | 2228| 240 | 16129 | 149281 | 
| 46 | 19 sklearn/discriminant_analysis.py | 1 | 29| 161 | 16290 | 156047 | 
| 47 | 19 sklearn/linear_model/logistic.py | 2102 | 2134| 401 | 16691 | 156047 | 
| 48 | 19 sklearn/linear_model/logistic.py | 1686 | 1928| 2833 | 19524 | 156047 | 
| 49 | **20 sklearn/utils/_pprint.py** | 403 | 436| 327 | 19851 | 160147 | 
| 50 | 20 sklearn/utils/multiclass.py | 9 | 37| 158 | 20009 | 160147 | 
| 51 | 21 examples/ensemble/plot_bias_variance.py | 69 | 96| 220 | 20229 | 161961 | 
| 52 | 21 sklearn/utils/estimator_checks.py | 2380 | 2424| 427 | 20656 | 161961 | 
| 53 | 22 sklearn/preprocessing/data.py | 11 | 59| 309 | 20965 | 187112 | 
| 54 | 23 examples/plot_anomaly_comparison.py | 81 | 153| 763 | 21728 | 188636 | 
| 55 | 23 sklearn/utils/estimator_checks.py | 2069 | 2089| 227 | 21955 | 188636 | 
| 56 | 23 sklearn/linear_model/logistic.py | 2135 | 2211| 820 | 22775 | 188636 | 
| 57 | 23 sklearn/linear_model/logistic.py | 1 | 39| 240 | 23015 | 188636 | 
| 58 | 24 examples/linear_model/plot_logistic_multinomial.py | 1 | 72| 669 | 23684 | 189333 | 
| 59 | 24 sklearn/linear_model/logistic.py | 2040 | 2101| 680 | 24364 | 189333 | 
| 60 | 24 sklearn/utils/estimator_checks.py | 1105 | 1125| 236 | 24600 | 189333 | 
| 61 | 25 sklearn/ensemble/iforest.py | 6 | 26| 116 | 24716 | 193687 | 
| 62 | 26 benchmarks/bench_plot_incremental_pca.py | 104 | 151| 468 | 25184 | 195043 | 
| 63 | 27 examples/linear_model/plot_logistic.py | 1 | 66| 438 | 25622 | 195489 | 


## Patch

```diff
diff --git a/sklearn/utils/_pprint.py b/sklearn/utils/_pprint.py
--- a/sklearn/utils/_pprint.py
+++ b/sklearn/utils/_pprint.py
@@ -95,7 +95,7 @@ def _changed_params(estimator):
     init_params = signature(init_func).parameters
     init_params = {name: param.default for name, param in init_params.items()}
     for k, v in params.items():
-        if (v != init_params[k] and
+        if (repr(v) != repr(init_params[k]) and
                 not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
             filtered_params[k] = v
     return filtered_params

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_pprint.py b/sklearn/utils/tests/test_pprint.py
--- a/sklearn/utils/tests/test_pprint.py
+++ b/sklearn/utils/tests/test_pprint.py
@@ -4,6 +4,7 @@
 import numpy as np
 
 from sklearn.utils._pprint import _EstimatorPrettyPrinter
+from sklearn.linear_model import LogisticRegressionCV
 from sklearn.pipeline import make_pipeline
 from sklearn.base import BaseEstimator, TransformerMixin
 from sklearn.feature_selection import SelectKBest, chi2
@@ -212,6 +213,9 @@ def test_changed_only():
     expected = """SimpleImputer()"""
     assert imputer.__repr__() == expected
 
+    # make sure array parameters don't throw error (see #13583)
+    repr(LogisticRegressionCV(Cs=np.array([0.1, 1])))
+
     set_config(print_changed_only=False)
 
 

```


## Code snippets

### 1 - examples/plot_changed_only_pprint_parameter.py:

Start line: 1, End line: 31

```python
"""
=================================
Compact estimator representations
=================================

This example illustrates the use of the print_changed_only global parameter.

Setting print_changed_only to True will alterate the representation of
estimators to only show the parameters that have been set to non-default
values. This can be used to have more compact representations.
"""
print(__doc__)

from sklearn.linear_model import LogisticRegression
from sklearn import set_config


lr = LogisticRegression(penalty='l1')
print('Default representation:')
print(lr)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l1',
#                    random_state=None, solver='warn', tol=0.0001, verbose=0,
#                    warm_start=False)

set_config(print_changed_only=True)
print('\nWith changed_only option:')
print(lr)
# LogisticRegression(penalty='l1')
```
### 2 - sklearn/utils/estimator_checks.py:

Start line: 2092, End line: 2106

```python
@ignore_warnings(category=DeprecationWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X = np.array([[3, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 1]])
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = [1, 1, 1, 2, 2, 2]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)


@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = _boston_subset(n_samples=50)
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)
```
### 3 - sklearn/utils/estimator_checks.py:

Start line: 880, End line: 911

```python
@ignore_warnings
def check_fit2d_1feature(name, estimator_orig):
    # check fitting a 2d array with only 1 feature either works or returns
    # informative message
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(10, 1))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1
    # ensure two labels in subsample for RandomizedLogisticRegression
    if name == 'RandomizedLogisticRegression':
        estimator.sample_fraction = 1
    # ensure non skipped trials for RANSACRegressor
    if name == 'RANSACRegressor':
        estimator.residual_threshold = 0.5

    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator, 1)

    msgs = ["1 feature(s)", "n_features = 1", "n_features=1"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e
```
### 4 - examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py:

Start line: 1, End line: 63

```python
"""
=====================================================
Multiclass sparse logisitic regression on newgroups20
=====================================================

Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression
to classify documents from the newgroups20 dataset. Multinomial logistic
regression yields more accurate results and is faster to train on the larger
scale dataset.

Here we use the l1 sparsity that trims the weights of not informative
features to zero. This is good if the goal is to extract the strongly
discriminative vocabulary of each class. If the goal is to get the best
predictive accuracy, it is better to use the non sparsity-inducing l2 penalty
instead.

A more traditional (and possibly better) way to predict on a sparse subset of
input features would be to use univariate feature selection followed by a
traditional (l2-penalised) logistic regression model.
"""
import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

print(__doc__)

warnings.filterwarnings("ignore", category=ConvergenceWarning,
                        module="sklearn")
t0 = timeit.default_timer()

# We use SAGA solver
solver = 'saga'

# Turn down for faster run time
n_samples = 10000

# Memorized fetch_rcv1 for faster access
dataset = fetch_20newsgroups_vectorized('all')
X = dataset.data
y = dataset.target
X = X[:n_samples]
y = y[:n_samples]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, n_features, n_classes))

models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 2, 4]},
          'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}}
```
### 5 - sklearn/utils/validation.py:

Start line: 693, End line: 716

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
### 6 - sklearn/utils/validation.py:

Start line: 470, End line: 539

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

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, array.shape, ensure_min_samples,
                                context))
    # ... other code
```
### 7 - sklearn/utils/estimator_checks.py:

Start line: 1151, End line: 1219

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
### 8 - sklearn/metrics/classification.py:

Start line: 2304, End line: 2320

```python
def _check_binary_probabilistic_predictions(y_true, y_prob):
    """Check that y_true is binary and y_prob contains valid probabilities"""
    check_consistent_length(y_true, y_prob)

    labels = np.unique(y_true)

    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)

    if y_prob.max() > 1:
        raise ValueError("y_prob contains values greater than 1.")

    if y_prob.min() < 0:
        raise ValueError("y_prob contains values less than 0.")

    return label_binarize(y_true, labels)[:, 0]
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 660, End line: 711

```python
def check_complex_data(name, estimator_orig):
    # check that estimators raise an exception on providing complex data
    X = np.random.sample(10) + 1j * np.random.sample(10)
    X = X.reshape(-1, 1)
    y = np.random.sample(10) + 1j * np.random.sample(10)
    estimator = clone(estimator_orig)
    assert_raises_regex(ValueError, "Complex data not supported",
                        estimator.fit, X, y)


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    # this estimator raises
    # ValueError: Found array with 0 feature(s) (shape=(23, 0))
    # while a minimum of 1 is required.
    # error
    if name in ['SpectralCoclustering']:
        return
    rnd = np.random.RandomState(0)
    if name in ['RANSACRegressor']:
        X = 3 * rnd.uniform(size=(20, 3))
    else:
        X = 2 * rnd.uniform(size=(20, 3))

    X = pairwise_estimator_convert_X(X, estimator_orig)

    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function",
                   "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert_dict_equal(estimator.__dict__, dict_before,
                              'Estimator changes __dict__ during %s' % method)


def is_public_parameter(attr):
    return not (attr.startswith('_') or attr.endswith('_'))
```
### 10 - sklearn/utils/validation.py:

Start line: 541, End line: 567

```python
def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=False, estimator=None):
    # ... other code

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, array.shape, ensure_min_features,
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
### 49 - sklearn/utils/_pprint.py:

Start line: 403, End line: 436

```python
def _safe_repr(object, context, maxlevels, level, changed_only=False):
    # ... other code

    if issubclass(typ, BaseEstimator):
        objid = id(object)
        if maxlevels and level >= maxlevels:
            return "{...}", False, objid in context
        if objid in context:
            return pprint._recursion(object), False, True
        context[objid] = 1
        readable = True
        recursive = False
        if changed_only:
            params = _changed_params(object)
        else:
            params = object.get_params(deep=False)
        components = []
        append = components.append
        level += 1
        saferepr = _safe_repr
        items = sorted(params.items(), key=pprint._safe_tuple)
        for k, v in items:
            krepr, kreadable, krecur = saferepr(
                k, context, maxlevels, level, changed_only=changed_only)
            vrepr, vreadable, vrecur = saferepr(
                v, context, maxlevels, level, changed_only=changed_only)
            append("%s=%s" % (krepr.strip("'"), vrepr))
            readable = readable and kreadable and vreadable
            if krecur or vrecur:
                recursive = True
        del context[objid]
        return ("%s(%s)" % (typ.__name__, ", ".join(components)), readable,
                recursive)

    rep = repr(object)
    return rep, (rep and not rep.startswith('<')), False
```

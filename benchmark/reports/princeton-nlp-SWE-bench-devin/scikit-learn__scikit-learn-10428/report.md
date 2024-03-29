# scikit-learn__scikit-learn-10428

| **scikit-learn/scikit-learn** | `db127bd9693068a5b187d49d08738e690c5c7d98` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 27383 |
| **Any found context length** | 21491 |
| **Avg pos** | 159.0 |
| **Min pos** | 69 |
| **Max pos** | 90 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/utils/estimator_checks.py b/sklearn/utils/estimator_checks.py
--- a/sklearn/utils/estimator_checks.py
+++ b/sklearn/utils/estimator_checks.py
@@ -226,6 +226,7 @@ def _yield_all_checks(name, estimator):
         for check in _yield_clustering_checks(name, estimator):
             yield check
     yield check_fit2d_predict1d
+    yield check_methods_subset_invariance
     if name != 'GaussianProcess':  # FIXME
         # XXX GaussianProcess deprecated in 0.20
         yield check_fit2d_1sample
@@ -643,6 +644,58 @@ def check_fit2d_predict1d(name, estimator_orig):
                                  getattr(estimator, method), X[0])
 
 
+def _apply_func(func, X):
+    # apply function on the whole set and on mini batches
+    result_full = func(X)
+    n_features = X.shape[1]
+    result_by_batch = [func(batch.reshape(1, n_features))
+                       for batch in X]
+    # func can output tuple (e.g. score_samples)
+    if type(result_full) == tuple:
+        result_full = result_full[0]
+        result_by_batch = list(map(lambda x: x[0], result_by_batch))
+
+    return np.ravel(result_full), np.ravel(result_by_batch)
+
+
+@ignore_warnings(category=(DeprecationWarning, FutureWarning))
+def check_methods_subset_invariance(name, estimator_orig):
+    # check that method gives invariant results if applied
+    # on mini bathes or the whole set
+    rnd = np.random.RandomState(0)
+    X = 3 * rnd.uniform(size=(20, 3))
+    X = pairwise_estimator_convert_X(X, estimator_orig)
+    y = X[:, 0].astype(np.int)
+    estimator = clone(estimator_orig)
+    y = multioutput_estimator_convert_y_2d(estimator, y)
+
+    if hasattr(estimator, "n_components"):
+        estimator.n_components = 1
+    if hasattr(estimator, "n_clusters"):
+        estimator.n_clusters = 1
+
+    set_random_state(estimator, 1)
+    estimator.fit(X, y)
+
+    for method in ["predict", "transform", "decision_function",
+                   "score_samples", "predict_proba"]:
+
+        msg = ("{method} of {name} is not invariant when applied "
+               "to a subset.").format(method=method, name=name)
+        # TODO remove cases when corrected
+        if (name, method) in [('SVC', 'decision_function'),
+                              ('SparsePCA', 'transform'),
+                              ('MiniBatchSparsePCA', 'transform'),
+                              ('BernoulliRBM', 'score_samples')]:
+            raise SkipTest(msg)
+
+        if hasattr(estimator, method):
+            result_full, result_by_batch = _apply_func(
+                getattr(estimator, method), X)
+            assert_allclose(result_full, result_by_batch,
+                            atol=1e-7, err_msg=msg)
+
+
 @ignore_warnings
 def check_fit2d_1sample(name, estimator_orig):
     # Check that fitting a 2d array with only one sample either works or

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/utils/estimator_checks.py | 229 | 229 | 90 | 1 | 27383
| sklearn/utils/estimator_checks.py | 646 | 646 | 69 | 1 | 21491


## Problem Statement

```
Add common test to ensure all(predict(X[mask]) == predict(X)[mask])
I don't think we currently test that estimator predictions/transformations are invariant whether performed in batch or on subsets of a dataset. For some fitted estimator `est`, data `X` and any boolean mask `mask` of length `X.shape[0]`, we need:

\`\`\`python
all(est.method(X[mask]) == est.method(X)[mask])
\`\`\`
where `method` is any of {`predict`, `predict_proba`, `decision_function`, `score_samples`, `transform`}. Testing that predictions for individual samples match the predictions across the dataset might be sufficient. This should be added to common tests at `sklearn/utils/estimator_checks.py`

Indeed, #9174 reports that this is broken for one-vs-one classification. :'(
  

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/utils/estimator_checks.py** | 239 | 287| 367 | 367 | 17447 | 
| 2 | 2 sklearn/model_selection/_validation.py | 756 | 801| 462 | 829 | 29162 | 
| 3 | **2 sklearn/utils/estimator_checks.py** | 1200 | 1307| 1123 | 1952 | 29162 | 
| 4 | **2 sklearn/utils/estimator_checks.py** | 1729 | 1749| 200 | 2152 | 29162 | 
| 5 | 3 sklearn/utils/mocking.py | 43 | 87| 380 | 2532 | 29790 | 
| 6 | **3 sklearn/utils/estimator_checks.py** | 764 | 838| 702 | 3234 | 29790 | 
| 7 | 4 sklearn/preprocessing/imputation.py | 4 | 32| 154 | 3388 | 32774 | 
| 8 | **4 sklearn/utils/estimator_checks.py** | 1893 | 1926| 233 | 3621 | 32774 | 
| 9 | **4 sklearn/utils/estimator_checks.py** | 1026 | 1063| 328 | 3949 | 32774 | 
| 10 | 4 sklearn/model_selection/_validation.py | 666 | 698| 348 | 4297 | 32774 | 
| 11 | **4 sklearn/utils/estimator_checks.py** | 75 | 109| 264 | 4561 | 32774 | 
| 12 | **4 sklearn/utils/estimator_checks.py** | 878 | 903| 282 | 4843 | 32774 | 
| 13 | **4 sklearn/utils/estimator_checks.py** | 346 | 402| 300 | 5143 | 32774 | 
| 14 | **4 sklearn/utils/estimator_checks.py** | 841 | 875| 372 | 5515 | 32774 | 
| 15 | **4 sklearn/utils/estimator_checks.py** | 1712 | 1726| 211 | 5726 | 32774 | 
| 16 | 5 sklearn/tree/tree.py | 373 | 389| 180 | 5906 | 45740 | 
| 17 | **5 sklearn/utils/estimator_checks.py** | 739 | 761| 248 | 6154 | 45740 | 
| 18 | **5 sklearn/utils/estimator_checks.py** | 955 | 1023| 601 | 6755 | 45740 | 
| 19 | **5 sklearn/utils/estimator_checks.py** | 1149 | 1163| 146 | 6901 | 45740 | 
| 20 | **5 sklearn/utils/estimator_checks.py** | 1929 | 1947| 192 | 7093 | 45740 | 
| 21 | **5 sklearn/utils/estimator_checks.py** | 906 | 929| 269 | 7362 | 45740 | 
| 22 | 6 sklearn/multiclass.py | 110 | 130| 137 | 7499 | 52228 | 
| 23 | **6 sklearn/utils/estimator_checks.py** | 1458 | 1496| 442 | 7941 | 52228 | 
| 24 | **6 sklearn/utils/estimator_checks.py** | 1393 | 1427| 336 | 8277 | 52228 | 
| 25 | 7 sklearn/utils/testing.py | 506 | 543| 386 | 8663 | 59016 | 
| 26 | **7 sklearn/utils/estimator_checks.py** | 621 | 643| 221 | 8884 | 59016 | 
| 27 | **7 sklearn/utils/estimator_checks.py** | 1622 | 1657| 352 | 9236 | 59016 | 
| 28 | 8 sklearn/feature_selection/from_model.py | 132 | 144| 135 | 9371 | 60569 | 
| 29 | 9 sklearn/cross_validation.py | 1401 | 1451| 365 | 9736 | 78059 | 
| 30 | **9 sklearn/utils/estimator_checks.py** | 932 | 952| 255 | 9991 | 78059 | 
| 31 | 9 sklearn/model_selection/_validation.py | 192 | 234| 465 | 10456 | 78059 | 
| 32 | 10 sklearn/utils/validation.py | 753 | 819| 476 | 10932 | 84790 | 
| 33 | 10 sklearn/cross_validation.py | 1381 | 1398| 214 | 11146 | 84790 | 
| 34 | **10 sklearn/utils/estimator_checks.py** | 112 | 138| 258 | 11404 | 84790 | 
| 35 | **10 sklearn/utils/estimator_checks.py** | 570 | 618| 441 | 11845 | 84790 | 
| 36 | **10 sklearn/utils/estimator_checks.py** | 488 | 523| 328 | 12173 | 84790 | 
| 37 | 11 sklearn/ensemble/bagging.py | 970 | 1000| 248 | 12421 | 92599 | 
| 38 | 11 sklearn/model_selection/_validation.py | 701 | 755| 393 | 12814 | 92599 | 
| 39 | 12 sklearn/utils/__init__.py | 77 | 99| 120 | 12934 | 96434 | 
| 40 | **12 sklearn/utils/estimator_checks.py** | 1361 | 1390| 324 | 13258 | 96434 | 
| 41 | **12 sklearn/utils/estimator_checks.py** | 472 | 485| 166 | 13424 | 96434 | 
| 42 | **12 sklearn/utils/estimator_checks.py** | 405 | 446| 427 | 13851 | 96434 | 
| 43 | **12 sklearn/utils/estimator_checks.py** | 1568 | 1585| 174 | 14025 | 96434 | 
| 44 | **12 sklearn/utils/estimator_checks.py** | 1430 | 1455| 277 | 14302 | 96434 | 
| 45 | **12 sklearn/utils/estimator_checks.py** | 708 | 723| 139 | 14441 | 96434 | 
| 46 | **12 sklearn/utils/estimator_checks.py** | 526 | 567| 334 | 14775 | 96434 | 
| 47 | 12 sklearn/model_selection/_validation.py | 1194 | 1212| 238 | 15013 | 96434 | 
| 48 | 13 sklearn/grid_search.py | 442 | 536| 616 | 15629 | 104942 | 
| 49 | **13 sklearn/utils/estimator_checks.py** | 674 | 705| 297 | 15926 | 104942 | 
| 50 | 14 sklearn/preprocessing/_function_transformer.py | 100 | 119| 126 | 16052 | 106330 | 
| 51 | **14 sklearn/utils/estimator_checks.py** | 1329 | 1358| 219 | 16271 | 106330 | 
| 52 | **14 sklearn/utils/estimator_checks.py** | 726 | 736| 142 | 16413 | 106330 | 
| 53 | 15 benchmarks/bench_mnist.py | 85 | 106| 314 | 16727 | 108061 | 
| 54 | **15 sklearn/utils/estimator_checks.py** | 1820 | 1864| 455 | 17182 | 108061 | 
| 55 | 15 sklearn/utils/testing.py | 619 | 644| 260 | 17442 | 108061 | 
| 56 | 15 sklearn/preprocessing/_function_transformer.py | 87 | 98| 127 | 17569 | 108061 | 
| 57 | 15 sklearn/multiclass.py | 84 | 107| 184 | 17753 | 108061 | 
| 58 | **15 sklearn/utils/estimator_checks.py** | 1 | 72| 625 | 18378 | 108061 | 
| 59 | 15 sklearn/model_selection/_validation.py | 932 | 948| 245 | 18623 | 108061 | 
| 60 | **15 sklearn/utils/estimator_checks.py** | 449 | 469| 253 | 18876 | 108061 | 
| 61 | 15 sklearn/cross_validation.py | 1940 | 1953| 217 | 19093 | 108061 | 
| 62 | **15 sklearn/utils/estimator_checks.py** | 1066 | 1089| 185 | 19278 | 108061 | 
| 63 | 16 sklearn/model_selection/_search.py | 684 | 760| 809 | 20087 | 121020 | 
| 64 | **16 sklearn/utils/estimator_checks.py** | 1523 | 1565| 449 | 20536 | 121020 | 
| 65 | **16 sklearn/utils/estimator_checks.py** | 164 | 180| 153 | 20689 | 121020 | 
| 66 | 16 sklearn/ensemble/bagging.py | 124 | 147| 177 | 20866 | 121020 | 
| 67 | 17 sklearn/ensemble/iforest.py | 208 | 227| 188 | 21054 | 123680 | 
| 68 | 17 sklearn/cross_validation.py | 1765 | 1785| 203 | 21257 | 123680 | 
| **-> 69 <-** | **17 sklearn/utils/estimator_checks.py** | 646 | 671| 234 | 21491 | 123680 | 
| 70 | 18 sklearn/multioutput.py | 44 | 60| 123 | 21614 | 129186 | 
| 71 | 18 sklearn/ensemble/bagging.py | 409 | 426| 160 | 21774 | 129186 | 
| 72 | 18 sklearn/cross_validation.py | 1570 | 1582| 179 | 21953 | 129186 | 
| 73 | 18 sklearn/model_selection/_validation.py | 951 | 971| 211 | 22164 | 129186 | 
| 74 | 18 sklearn/model_selection/_validation.py | 448 | 513| 676 | 22840 | 129186 | 
| 75 | 18 sklearn/multiclass.py | 183 | 217| 353 | 23193 | 129186 | 
| 76 | **18 sklearn/utils/estimator_checks.py** | 1310 | 1326| 154 | 23347 | 129186 | 
| 77 | **18 sklearn/utils/estimator_checks.py** | 183 | 200| 220 | 23567 | 129186 | 
| 78 | 18 sklearn/multiclass.py | 478 | 513| 283 | 23850 | 129186 | 
| 79 | 18 sklearn/multiclass.py | 275 | 314| 374 | 24224 | 129186 | 
| 80 | 19 sklearn/feature_extraction/image.py | 64 | 84| 169 | 24393 | 133611 | 
| 81 | 20 sklearn/model_selection/_split.py | 100 | 121| 208 | 24601 | 151612 | 
| 82 | 20 sklearn/ensemble/bagging.py | 175 | 186| 104 | 24705 | 151612 | 
| 83 | 20 sklearn/preprocessing/imputation.py | 250 | 299| 482 | 25187 | 151612 | 
| 84 | 20 sklearn/cross_validation.py | 1671 | 1710| 393 | 25580 | 151612 | 
| 85 | 21 sklearn/dummy.py | 238 | 298| 505 | 26085 | 155272 | 
| 86 | **21 sklearn/utils/estimator_checks.py** | 1166 | 1197| 258 | 26343 | 155272 | 
| 87 | **21 sklearn/utils/estimator_checks.py** | 1867 | 1890| 308 | 26651 | 155272 | 
| 88 | 21 sklearn/ensemble/bagging.py | 580 | 616| 296 | 26947 | 155272 | 
| 89 | 22 sklearn/cluster/birch.py | 540 | 553| 140 | 27087 | 160555 | 
| **-> 90 <-** | **22 sklearn/utils/estimator_checks.py** | 203 | 236| 296 | 27383 | 160555 | 
| 91 | 22 sklearn/cross_validation.py | 62 | 103| 298 | 27681 | 160555 | 
| 92 | 23 sklearn/learning_curve.py | 240 | 258| 238 | 27919 | 163963 | 
| 93 | 23 sklearn/multiclass.py | 756 | 774| 153 | 28072 | 163963 | 
| 94 | 23 sklearn/cross_validation.py | 1585 | 1669| 637 | 28709 | 163963 | 
| 95 | 23 sklearn/utils/__init__.py | 102 | 119| 181 | 28890 | 163963 | 
| 96 | 23 sklearn/model_selection/_validation.py | 571 | 665| 815 | 29705 | 163963 | 
| 97 | **23 sklearn/utils/estimator_checks.py** | 1588 | 1619| 340 | 30045 | 163963 | 
| 98 | 23 sklearn/multiclass.py | 66 | 81| 127 | 30172 | 163963 | 
| 99 | 23 sklearn/multiclass.py | 413 | 435| 238 | 30410 | 163963 | 
| 100 | 23 sklearn/multiclass.py | 707 | 754| 384 | 30794 | 163963 | 
| 101 | 24 sklearn/preprocessing/data.py | 2752 | 2786| 305 | 31099 | 189764 | 
| 102 | 24 sklearn/dummy.py | 163 | 236| 595 | 31694 | 189764 | 
| 103 | **24 sklearn/utils/estimator_checks.py** | 141 | 161| 203 | 31897 | 189764 | 
| 104 | 25 benchmarks/bench_covertype.py | 100 | 110| 151 | 32048 | 191667 | 
| 105 | 26 sklearn/pipeline.py | 339 | 357| 135 | 32183 | 197873 | 


### Hint

```
Hi, could I take this issue ?
sure, it seems right up your alley. thanks!

```

## Patch

```diff
diff --git a/sklearn/utils/estimator_checks.py b/sklearn/utils/estimator_checks.py
--- a/sklearn/utils/estimator_checks.py
+++ b/sklearn/utils/estimator_checks.py
@@ -226,6 +226,7 @@ def _yield_all_checks(name, estimator):
         for check in _yield_clustering_checks(name, estimator):
             yield check
     yield check_fit2d_predict1d
+    yield check_methods_subset_invariance
     if name != 'GaussianProcess':  # FIXME
         # XXX GaussianProcess deprecated in 0.20
         yield check_fit2d_1sample
@@ -643,6 +644,58 @@ def check_fit2d_predict1d(name, estimator_orig):
                                  getattr(estimator, method), X[0])
 
 
+def _apply_func(func, X):
+    # apply function on the whole set and on mini batches
+    result_full = func(X)
+    n_features = X.shape[1]
+    result_by_batch = [func(batch.reshape(1, n_features))
+                       for batch in X]
+    # func can output tuple (e.g. score_samples)
+    if type(result_full) == tuple:
+        result_full = result_full[0]
+        result_by_batch = list(map(lambda x: x[0], result_by_batch))
+
+    return np.ravel(result_full), np.ravel(result_by_batch)
+
+
+@ignore_warnings(category=(DeprecationWarning, FutureWarning))
+def check_methods_subset_invariance(name, estimator_orig):
+    # check that method gives invariant results if applied
+    # on mini bathes or the whole set
+    rnd = np.random.RandomState(0)
+    X = 3 * rnd.uniform(size=(20, 3))
+    X = pairwise_estimator_convert_X(X, estimator_orig)
+    y = X[:, 0].astype(np.int)
+    estimator = clone(estimator_orig)
+    y = multioutput_estimator_convert_y_2d(estimator, y)
+
+    if hasattr(estimator, "n_components"):
+        estimator.n_components = 1
+    if hasattr(estimator, "n_clusters"):
+        estimator.n_clusters = 1
+
+    set_random_state(estimator, 1)
+    estimator.fit(X, y)
+
+    for method in ["predict", "transform", "decision_function",
+                   "score_samples", "predict_proba"]:
+
+        msg = ("{method} of {name} is not invariant when applied "
+               "to a subset.").format(method=method, name=name)
+        # TODO remove cases when corrected
+        if (name, method) in [('SVC', 'decision_function'),
+                              ('SparsePCA', 'transform'),
+                              ('MiniBatchSparsePCA', 'transform'),
+                              ('BernoulliRBM', 'score_samples')]:
+            raise SkipTest(msg)
+
+        if hasattr(estimator, method):
+            result_full, result_by_batch = _apply_func(
+                getattr(estimator, method), X)
+            assert_allclose(result_full, result_by_batch,
+                            atol=1e-7, err_msg=msg)
+
+
 @ignore_warnings
 def check_fit2d_1sample(name, estimator_orig):
     # Check that fitting a 2d array with only one sample either works or

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_estimator_checks.py b/sklearn/utils/tests/test_estimator_checks.py
--- a/sklearn/utils/tests/test_estimator_checks.py
+++ b/sklearn/utils/tests/test_estimator_checks.py
@@ -134,6 +134,23 @@ def predict(self, X):
         return np.ones(X.shape[0])
 
 
+class NotInvariantPredict(BaseEstimator):
+    def fit(self, X, y):
+        # Convert data
+        X, y = check_X_y(X, y,
+                         accept_sparse=("csr", "csc"),
+                         multi_output=True,
+                         y_numeric=True)
+        return self
+
+    def predict(self, X):
+        # return 1 if X has more than one element else return 0
+        X = check_array(X)
+        if X.shape[0] > 1:
+            return np.ones(X.shape[0])
+        return np.zeros(X.shape[0])
+
+
 def test_check_estimator():
     # tests that the estimator actually fails on "bad" estimators.
     # not a complete test of all checks, which are very extensive.
@@ -184,6 +201,13 @@ def test_check_estimator():
            ' with _ but wrong_attribute added')
     assert_raises_regex(AssertionError, msg,
                         check_estimator, SetsWrongAttribute)
+    # check for invariant method
+    name = NotInvariantPredict.__name__
+    method = 'predict'
+    msg = ("{method} of {name} is not invariant when applied "
+           "to a subset.").format(method=method, name=name)
+    assert_raises_regex(AssertionError, msg,
+                        check_estimator, NotInvariantPredict)
     # check for sparse matrix input handling
     name = NoSparseClassifier.__name__
     msg = "Estimator %s doesn't seem to fail gracefully on sparse data" % name

```


## Code snippets

### 1 - sklearn/utils/estimator_checks.py:

Start line: 239, End line: 287

```python
def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions.

    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.

    This test can be applied to classes or instances.
    Classes currently have some additional tests that related to construction,
    while passing instances allows the testing of multiple options.

    Parameters
    ----------
    estimator : estimator object or class
        Estimator to check. Estimator is a class object or instance.

    """
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_all_checks(name, estimator):
        try:
            check(name, estimator)
        except SkipTest as message:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(message, SkipTestWarning)


def _boston_subset(n_samples=200):
    global BOSTON
    if BOSTON is None:
        boston = load_boston()
        X, y = boston.data, boston.target
        X, y = shuffle(X, y, random_state=0)
        X, y = X[:n_samples], y[:n_samples]
        X = StandardScaler().fit_transform(X)
        BOSTON = X, y
    return BOSTON
```
### 2 - sklearn/model_selection/_validation.py:

Start line: 756, End line: 801

```python
def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
                     method):
    # ... other code
    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        n_classes = len(set(y))
        if n_classes != len(estimator.classes_):
            recommendation = (
                'To fix this, use a cross-validation '
                'technique resulting in properly '
                'stratified folds')
            warnings.warn('Number of classes in training fold ({}) does '
                          'not match total number of classes ({}). '
                          'Results may not be appropriate for your use case. '
                          '{}'.format(len(estimator.classes_),
                                      n_classes, recommendation),
                          RuntimeWarning)
            if method == 'decision_function':
                if (predictions.ndim == 2 and
                        predictions.shape[1] != len(estimator.classes_)):
                    # This handles the case when the shape of predictions
                    # does not match the number of classes used to train
                    # it with. This case is found when sklearn.svm.SVC is
                    # set to `decision_function_shape='ovo'`.
                    raise ValueError('Output shape {} of {} does not match '
                                     'number of classes ({}) in fold. '
                                     'Irregular decision_function outputs '
                                     'are not currently supported by '
                                     'cross_val_predict'.format(
                                        predictions.shape, method,
                                        len(estimator.classes_),
                                        recommendation))
                if len(estimator.classes_) <= 2:
                    # In this special case, `predictions` contains a 1D array.
                    raise ValueError('Only {} class/es in training fold, this '
                                     'is not supported for decision_function '
                                     'with imbalanced folds. {}'.format(
                                        len(estimator.classes_),
                                        recommendation))

            float_min = np.finfo(predictions.dtype).min
            default_values = {'decision_function': float_min,
                              'predict_log_proba': float_min,
                              'predict_proba': 0}
            predictions_for_all_classes = np.full((_num_samples(predictions),
                                                   n_classes),
                                                  default_values[method])
            predictions_for_all_classes[:, estimator.classes_] = predictions
            predictions = predictions_for_all_classes
    return predictions, test
```
### 3 - sklearn/utils/estimator_checks.py:

Start line: 1200, End line: 1307

```python
@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig):
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            X -= X.min()
        X = pairwise_estimator_convert_X(X, classifier_orig)
        set_random_state(classifier)
        # raises error on malformed input for fit
        with assert_raises(ValueError, msg="The classifer {} does not"
                           " raise an error when incorrect/malformed input "
                           "data for fit is passed. The number of training "
                           "examples is not the same as the number of labels."
                           " Perhaps use check_X_y in fit.".format(name)):
            classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert_true(hasattr(classifier, "classes_"))
        y_pred = classifier.predict(X)
        assert_equal(y_pred.shape, (n_samples,))
        # training set performance
        if name not in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
            assert_greater(accuracy_score(y, y_pred), 0.83)

        # raises error on malformed input for predict
        if _is_pairwise(classifier):
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when shape of X"
                               "in predict is not equal to (n_test_samples,"
                               "n_training_samples)".format(name)):
                classifier.predict(X.reshape(-1, 1))
        else:
            with assert_raises(ValueError, msg="The classifier {} does not"
                               " raise an error when the number of features "
                               "in predict is different from the number of"
                               " features in fit.".format(name)):
                classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    assert_equal(decision.shape, (n_samples,))
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                if (n_classes == 3 and
                        # 1on1 of LibSVM works differently
                        not isinstance(classifier, BaseLibSVM)):
                    assert_equal(decision.shape, (n_samples, n_classes))
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if _is_pairwise(classifier):
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the  "
                                       "shape of X in decision_function is "
                                       "not equal to (n_test_samples, "
                                       "n_training_samples) in fit."
                                       .format(name)):
                        classifier.decision_function(X.reshape(-1, 1))
                else:
                    with assert_raises(ValueError, msg="The classifier {} does"
                                       " not raise an error when the number "
                                       "of features in decision_function is "
                                       "different from the number of features"
                                       " in fit.".format(name)):
                        classifier.decision_function(X.T)
            except NotImplementedError:
                pass
        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert_equal(y_prob.shape, (n_samples, n_classes))
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_allclose(np.sum(y_prob, axis=1), np.ones(n_samples))
            # raises error on malformed input for predict_proba
            if _is_pairwise(classifier_orig):
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the shape of X"
                                   "in predict_proba is not equal to "
                                   "(n_test_samples, n_training_samples)."
                                   .format(name)):
                    classifier.predict_proba(X.reshape(-1, 1))
            else:
                with assert_raises(ValueError, msg="The classifier {} does not"
                                   " raise an error when the number of "
                                   "features in predict_proba is different "
                                   "from the number of features in fit."
                                   .format(name)):
                    classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))
```
### 4 - sklearn/utils/estimator_checks.py:

Start line: 1729, End line: 1749

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_data_not_an_array(name, estimator_orig, X, y):
    if name in CROSS_DECOMPOSITION:
        raise SkipTest("Skipping check_estimators_data_not_an_array "
                       "for cross decomposition module as estimators "
                       "are not deterministic.")
    # separate estimators to control random seeds
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    y_ = NotAnArray(np.asarray(y))
    X_ = NotAnArray(np.asarray(X))

    # fit
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)
```
### 5 - sklearn/utils/mocking.py:

Start line: 43, End line: 87

```python
class CheckingClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of X and y in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.
    """
    def __init__(self, check_y=None, check_X=None, foo_param=0,
                 expected_fit_params=None):
        self.check_y = check_y
        self.check_X = check_X
        self.foo_param = foo_param
        self.expected_fit_params = expected_fit_params

    def fit(self, X, y, **fit_params):
        assert_true(len(X) == len(y))
        if self.check_X is not None:
            assert_true(self.check_X(X))
        if self.check_y is not None:
            assert_true(self.check_y(y))
        self.classes_ = np.unique(check_array(y, ensure_2d=False,
                                              allow_nd=True))
        if self.expected_fit_params:
            missing = set(self.expected_fit_params) - set(fit_params)
            assert_true(len(missing) == 0, 'Expected fit parameter(s) %s not '
                                           'seen.' % list(missing))
            for key, value in fit_params.items():
                assert_true(len(value) == len(X),
                            'Fit parameter %s has length %d; '
                            'expected %d.' % (key, len(value), len(X)))

        return self

    def predict(self, T):
        if self.check_X is not None:
            assert_true(self.check_X(T))
        return self.classes_[np.zeros(_num_samples(T), dtype=np.int)]

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score
```
### 6 - sklearn/utils/estimator_checks.py:

Start line: 764, End line: 838

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
### 7 - sklearn/preprocessing/imputation.py:

Start line: 4, End line: 32

```python
import warnings

import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array
from ..utils.sparsefuncs import _get_median
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES

from ..externals import six

zip = six.moves.zip
map = six.moves.map

__all__ = [
    'Imputer',
]


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask
```
### 8 - sklearn/utils/estimator_checks.py:

Start line: 1893, End line: 1926

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_get_params_invariance(name, estimator_orig):
    # Checks if get_params(deep=False) is a subset of get_params(deep=True)
    class T(BaseEstimator):
        """Mock classifier
        """

        def __init__(self):
            pass

        def fit(self, X, y):
            return self

        def transform(self, X):
            return X

    e = clone(estimator_orig)

    shallow_params = e.get_params(deep=False)
    deep_params = e.get_params(deep=True)

    assert_true(all(item in deep_params.items() for item in
                    shallow_params.items()))


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_regression_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    boston = load_boston()
    X, y = boston.data, boston.target
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    assert_raises_regex(ValueError, msg, e.fit, X, y)
```
### 9 - sklearn/utils/estimator_checks.py:

Start line: 1026, End line: 1063

```python
@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators"""
    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]

    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)

    # some estimators can't do features less than 0
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)

    estimator = clone(estimator_orig)

    # some estimators only take multioutputs
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)
    estimator.fit(X, y)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    if estimator.__module__.startswith('sklearn.'):
        assert_true(b"version" in pickled_estimator)
    unpickled_estimator = pickle.loads(pickled_estimator)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)
```
### 10 - sklearn/model_selection/_validation.py:

Start line: 666, End line: 698

```python
def cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1,
                      verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                      method='predict'):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
        le = LabelEncoder()
        y = le.fit_transform(y)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    # Check for sparse predictions
    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    else:
        predictions = np.concatenate(predictions)
    return predictions[inv_test_indices]
```
### 11 - sklearn/utils/estimator_checks.py:

Start line: 75, End line: 109

```python
def _yield_non_meta_checks(name, estimator):
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_dtype_object
    yield check_sample_weights_pandas_series
    yield check_sample_weights_list
    yield check_estimators_fit_returns_self
    yield check_complex_data

    # Check that all estimator yield informative messages when
    # trained on empty datasets
    yield check_estimators_empty_data_messages

    if name not in CROSS_DECOMPOSITION + ['SpectralEmbedding']:
        # SpectralEmbedding is non-deterministic,
        # see issue #4236
        # cross-decomposition's "transform" returns X and Y
        yield check_pipeline_consistency

    if name not in ['Imputer']:
        # Test that all estimators check their input for NaN's and infs
        yield check_estimators_nan_inf

    if name not in ['GaussianProcess']:
        # FIXME!
        # in particular GaussianProcess!
        yield check_estimators_overwrite_params
    if hasattr(estimator, 'sparsify'):
        yield check_sparsify_coefficients

    yield check_estimator_sparse_data

    # Test that estimators can be pickled, and once pickled
    # give the same answer as before.
    yield check_estimators_pickle
```
### 12 - sklearn/utils/estimator_checks.py:

Start line: 878, End line: 903

```python
@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    # check that all estimators accept an optional y
    # in fit and score so they can be used in pipelines
    rnd = np.random.RandomState(0)
    X = rnd.uniform(size=(10, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = np.arange(10) % 3
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert_true(args[1] in ["y", "Y"],
                        "Expected y or Y as second argument for method "
                        "%s of %s. Got arguments: %r."
                        % (func_name, type(estimator).__name__, args))
```
### 13 - sklearn/utils/estimator_checks.py:

Start line: 346, End line: 402

```python
class NotAnArray(object):
    " An object that is convertable to an array"

    def __init__(self, data):
        self.data = data

    def __array__(self, dtype=None):
        return self.data


def _is_32bit():
    """Detect if process is 32bit Python."""
    return struct.calcsize('P') * 8 == 32


def _is_pairwise(estimator):
    """Returns True if estimator has a _pairwise attribute set to True.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    return bool(getattr(estimator, "_pairwise", False))


def _is_pairwise_metric(estimator):
    """Returns True if estimator accepts pairwise metric.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if _pairwise is set to True and False otherwise.
    """
    metric = getattr(estimator,  "metric", None)

    return bool(metric == 'precomputed')


def pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):

    if _is_pairwise_metric(estimator):
        return pairwise_distances(X, metric='euclidean')
    if _is_pairwise(estimator):
        return kernel(X, X)

    return X
```
### 14 - sklearn/utils/estimator_checks.py:

Start line: 841, End line: 875

```python
@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if name in ('CCA', 'LocallyLinearEmbedding', 'KernelPCA') and _is_32bit():
        # Those transformers yield non-deterministic output when executed on
        # a 32bit Python. The same transformers are stable on 64bit Python.
        # FIXME: try to isolate a minimalistic reproduction case only depending
        # scipy and/or maybe generate a test dataset that does not
        # cause such unstable behaviors.
        msg = name + ' is non deterministic on 32bit Python'
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)
```
### 15 - sklearn/utils/estimator_checks.py:

Start line: 1712, End line: 1726

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
### 17 - sklearn/utils/estimator_checks.py:

Start line: 739, End line: 761

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_data_not_an_array(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - .1
    this_X = NotAnArray(X)
    this_y = NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformers_unfitted(name, transformer):
    X, y = _boston_subset()

    transformer = clone(transformer)
    with assert_raises((AttributeError, ValueError), msg="The unfitted "
                       "transformer {} does not raise an error when "
                       "transform is called. Perhaps use "
                       "check_is_fitted in transform.".format(name)):
        transformer.transform(X)
```
### 18 - sklearn/utils/estimator_checks.py:

Start line: 955, End line: 1023

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
### 19 - sklearn/utils/estimator_checks.py:

Start line: 1149, End line: 1163

```python
@ignore_warnings(category=DeprecationWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels"""
    X, y = make_blobs(n_samples=20, random_state=0)
    clusterer = clone(clusterer_orig)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        if hasattr(clusterer, "random_state"):
            clusterer.set_params(random_state=0)

        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)
```
### 20 - sklearn/utils/estimator_checks.py:

Start line: 1929, End line: 1947

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_decision_proba_consistency(name, estimator_orig):
    # Check whether an estimator having both decision_function and
    # predict_proba methods has outputs with perfect rank correlation.

    centers = [(2, 2), (4, 4)]
    X, y = make_blobs(n_samples=100, random_state=0, n_features=4,
                      centers=centers, cluster_std=1.0, shuffle=True)
    X_test = np.random.randn(20, 2) + 4
    estimator = clone(estimator_orig)

    if (hasattr(estimator, "decision_function") and
            hasattr(estimator, "predict_proba")):

        estimator.fit(X, y)
        a = estimator.predict_proba(X_test)[:, 1]
        b = estimator.decision_function(X_test)
        assert_array_equal(rankdata(a), rankdata(b))
```
### 21 - sklearn/utils/estimator_checks.py:

Start line: 906, End line: 929

```python
@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    rnd = np.random.RandomState(0)
    X_train_32 = 3 * rnd.uniform(size=(20, 5)).astype(np.float32)
    X_train_32 = pairwise_estimator_convert_X(X_train_32, estimator_orig)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = X_train_int_64[:, 0]
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        if name == 'PowerTransformer':
            # Box-Cox requires positive, non-zero data
            X_train = np.abs(X_train) + 1
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)
```
### 23 - sklearn/utils/estimator_checks.py:

Start line: 1458, End line: 1496

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_train(name, regressor_orig):
    X, y = _boston_subset()
    X = pairwise_estimator_convert_X(X, regressor_orig)
    y = StandardScaler().fit_transform(y.reshape(-1, 1))  # X is already scaled
    y = y.ravel()
    regressor = clone(regressor_orig)
    y = multioutput_estimator_convert_y_2d(regressor, y)
    rnd = np.random.RandomState(0)
    if not hasattr(regressor, 'alphas') and hasattr(regressor, 'alpha'):
        # linear regressors need to set alpha, but not generalized CV ones
        regressor.alpha = 0.01
    if name == 'PassiveAggressiveRegressor':
        regressor.C = 0.01

    # raises error on malformed input for fit
    with assert_raises(ValueError, msg="The classifer {} does not"
                       " raise an error when incorrect/malformed input "
                       "data for fit is passed. The number of training "
                       "examples is not the same as the number of "
                       "labels. Perhaps use check_X_y in fit.".format(name)):
        regressor.fit(X, y[:-1])
    # fit
    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y
    set_random_state(regressor)
    regressor.fit(X, y_)
    regressor.fit(X.tolist(), y_.tolist())
    y_pred = regressor.predict(X)
    assert_equal(y_pred.shape, y_.shape)

    # TODO: find out why PLS and CCA fail. RANSAC is random
    # and furthermore assumes the presence of outliers, hence
    # skipped
    if name not in ('PLSCanonical', 'CCA', 'RANSACRegressor'):
        assert_greater(regressor.score(X, y_), 0.5)
```
### 24 - sklearn/utils/estimator_checks.py:

Start line: 1393, End line: 1427

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_classes(name, classifier_orig):
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - .1
    X = pairwise_estimator_convert_X(X, classifier_orig)
    y_names = np.array(["one", "two", "three"])[y]

    for y_names in [y_names, y_names.astype('O')]:
        if name in ["LabelPropagation", "LabelSpreading"]:
            # TODO some complication with -1 label
            y_ = y
        else:
            y_ = y_names

        classes = np.unique(y_)
        classifier = clone(classifier_orig)
        if name == 'BernoulliNB':
            X = X > X.mean()
        set_random_state(classifier)
        # fit
        classifier.fit(X, y_)

        y_pred = classifier.predict(X)
        # training set performance
        if name != "ComplementNB":
            # This is a pathological data set for ComplementNB.
            assert_array_equal(np.unique(y_), np.unique(y_pred))
        if np.any(classifier.classes_ != classes):
            print("Unexpected classes_ attribute for %r: "
                  "expected %s, got %s" %
                  (classifier, classes, classifier.classes_))
```
### 26 - sklearn/utils/estimator_checks.py:

Start line: 621, End line: 643

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_fit2d_predict1d(name, estimator_orig):
    # check by fitting a 2d array and predicting with a 1d array
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function",
                   "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(ValueError, "Reshape your data",
                                 getattr(estimator, method), X[0])
```
### 27 - sklearn/utils/estimator_checks.py:

Start line: 1622, End line: 1657

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
### 30 - sklearn/utils/estimator_checks.py:

Start line: 932, End line: 952

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_empty_data_messages(name, estimator_orig):
    e = clone(estimator_orig)
    set_random_state(e, 1)

    X_zero_samples = np.empty(0).reshape(0, 3)
    # The precise message can change depending on whether X or y is
    # validated first. Let us test the type of exception only:
    with assert_raises(ValueError, msg="The estimator {} does not"
                       " raise an error when an empty data is used "
                       "to train. Perhaps use "
                       "check_array in train.".format(name)):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape(3, 0)
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = multioutput_estimator_convert_y_2d(e, np.array([1, 0, 1]))
    msg = ("0 feature\(s\) \(shape=\(3, 0\)\) while a minimum of \d* "
           "is required.")
    assert_raises_regex(ValueError, msg, e.fit, X_zero_features, y)
```
### 34 - sklearn/utils/estimator_checks.py:

Start line: 112, End line: 138

```python
def _yield_classifier_checks(name, classifier):
    # test classifiers can handle non-array data
    yield check_classifier_data_not_an_array
    # test classifiers trained on a single label always return this label
    yield check_classifiers_one_label
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    # basic consistency testing
    yield check_classifiers_train
    yield check_classifiers_regression_target
    if (name not in ["MultinomialNB", "ComplementNB", "LabelPropagation",
                     "LabelSpreading"] and
        # TODO some complication with -1 label
            name not in ["DecisionTreeClassifier", "ExtraTreeClassifier"]):
        # We don't raise a warning in these classifiers, as
        # the column y interface is used by the forests.

        yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    # test if NotFittedError is raised
    yield check_estimators_unfitted
    if 'class_weight' in classifier.get_params().keys():
        yield check_class_weight_classifiers

    yield check_non_transformer_estimators_n_iter
    # test if predict_proba is a monotonic transformation of decision_function
    yield check_decision_proba_consistency
```
### 35 - sklearn/utils/estimator_checks.py:

Start line: 570, End line: 618

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
### 36 - sklearn/utils/estimator_checks.py:

Start line: 488, End line: 523

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    # check that estimators treat dtype object as numeric if possible
    rng = np.random.RandomState(0)
    X = pairwise_estimator_convert_X(rng.rand(40, 10), estimator_orig)
    X = X.astype(object)
    y = (X[:, 0] * 4).astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    try:
        estimator.fit(X, y.astype(object))
    except Exception as e:
        if "Unknown label type" not in str(e):
            raise

    X[0, 0] = {'foo': 'bar'}
    msg = "argument must be a string or a number"
    assert_raises_regex(TypeError, msg, estimator.fit, X, y)


def check_complex_data(name, estimator_orig):
    # check that estimators raise an exception on providing complex data
    X = np.random.sample(10) + 1j * np.random.sample(10)
    X = X.reshape(-1, 1)
    y = np.random.sample(10) + 1j * np.random.sample(10)
    estimator = clone(estimator_orig)
    assert_raises_regex(ValueError, "Complex data not supported",
                        estimator.fit, X, y)
```
### 40 - sklearn/utils/estimator_checks.py:

Start line: 1361, End line: 1390

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
### 41 - sklearn/utils/estimator_checks.py:

Start line: 472, End line: 485

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sample_weights_list(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type list in the 'fit' function.
    if has_fit_parameter(estimator_orig, "sample_weight"):
        estimator = clone(estimator_orig)
        rnd = np.random.RandomState(0)
        X = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                         estimator_orig)
        y = np.arange(10) % 3
        y = multioutput_estimator_convert_y_2d(estimator, y)
        sample_weight = [3] * 10
        # Test that estimators don't raise any exception
        estimator.fit(X, y, sample_weight=sample_weight)
```
### 42 - sklearn/utils/estimator_checks.py:

Start line: 405, End line: 446

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
    for sparse_format in ['csr', 'csc', 'dok', 'lil', 'coo', 'dia', 'bsr']:
        X = X_csr.asformat(sparse_format)
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
                print("Estimator %s doesn't seem to fail gracefully on "
                      "sparse data: error message state explicitly that "
                      "sparse input is not supported if this is not the case."
                      % name)
                raise
        except Exception:
            print("Estimator %s doesn't seem to fail gracefully on "
                  "sparse data: it should raise a TypeError if sparse input "
                  "is explicitly not supported." % name)
            raise
```
### 43 - sklearn/utils/estimator_checks.py:

Start line: 1568, End line: 1585

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_balanced_classifiers(name, classifier_orig, X_train,
                                            y_train, X_test, y_test, weights):
    classifier = clone(classifier_orig)
    if hasattr(classifier, "n_iter"):
        classifier.set_params(n_iter=100)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)

    set_random_state(classifier)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    classifier.set_params(class_weight='balanced')
    classifier.fit(X_train, y_train)
    y_pred_balanced = classifier.predict(X_test)
    assert_greater(f1_score(y_test, y_pred_balanced, average='weighted'),
                   f1_score(y_test, y_pred, average='weighted'))
```
### 44 - sklearn/utils/estimator_checks.py:

Start line: 1430, End line: 1455

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_int(name, regressor_orig):
    X, _ = _boston_subset()
    X = pairwise_estimator_convert_X(X[:50], regressor_orig)
    rnd = np.random.RandomState(0)
    y = rnd.randint(3, size=X.shape[0])
    y = multioutput_estimator_convert_y_2d(regressor_orig, y)
    rnd = np.random.RandomState(0)
    # separate estimators to control random seeds
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    if name in CROSS_DECOMPOSITION:
        y_ = np.vstack([y, 2 * y + rnd.randint(2, size=len(y))])
        y_ = y_.T
    else:
        y_ = y

    # fit
    regressor_1.fit(X, y_)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y_.astype(np.float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)
```
### 45 - sklearn/utils/estimator_checks.py:

Start line: 708, End line: 723

```python
@ignore_warnings
def check_fit1d(name, estimator_orig):
    # check fitting 1d X array raises a ValueError
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    assert_raises(ValueError, estimator.fit, X, y)
```
### 46 - sklearn/utils/estimator_checks.py:

Start line: 526, End line: 567

```python
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
### 49 - sklearn/utils/estimator_checks.py:

Start line: 674, End line: 705

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
### 51 - sklearn/utils/estimator_checks.py:

Start line: 1329, End line: 1358

```python
@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise either AttributeError or ValueError.
    The specific exception type NotFittedError inherits from both and can
    therefore be adequately raised for that purpose.
    """

    # Common test for Regressors as well as Classifiers
    X, y = _boston_subset()

    est = clone(estimator_orig)

    msg = "fit"
    if hasattr(est, 'predict'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict, X)

    if hasattr(est, 'decision_function'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.decision_function, X)

    if hasattr(est, 'predict_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_proba, X)

    if hasattr(est, 'predict_log_proba'):
        assert_raise_message((AttributeError, ValueError), msg,
                             est.predict_log_proba, X)
```
### 52 - sklearn/utils/estimator_checks.py:

Start line: 726, End line: 736

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_general(name, transformer):
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X = StandardScaler().fit_transform(X)
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    _check_transformer(name, transformer, X, y)
    _check_transformer(name, transformer, X.tolist(), y.tolist())
```
### 54 - sklearn/utils/estimator_checks.py:

Start line: 1820, End line: 1864

```python
def multioutput_estimator_convert_y_2d(estimator, y):
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if "MultiTask" in estimator.__class__.__name__:
        return np.reshape(y, (-1, 1))
    return y


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = ['Ridge', 'SVR', 'NuSVR', 'NuSVC',
                            'RidgeClassifier', 'SVC', 'RandomizedLasso',
                            'LogisticRegressionCV', 'LinearSVC',
                            'LogisticRegression']

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y_ = multioutput_estimator_convert_y_2d(estimator, y_)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
            estimator.fit(X, y_)

        # HuberRegressor depends on scipy.optimize.fmin_l_bfgs_b
        # which doesn't return a n_iter for old versions of SciPy.
        if not (name == 'HuberRegressor' and estimator.n_iter_ is None):
            assert_greater_equal(estimator.n_iter_, 1)
```
### 58 - sklearn/utils/estimator_checks.py:

Start line: 1, End line: 72

```python
from __future__ import print_function

import types
import warnings
import sys
import traceback
import pickle
from copy import deepcopy
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
import struct

from sklearn.externals.six.moves import zip
from sklearn.externals.joblib import hash, Memory
from sklearn.utils.testing import assert_raises, _get_args
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.base import (clone, TransformerMixin, ClusterMixin,
                          BaseEstimator, is_classifier, is_regressor)

from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score

from sklearn.random_projection import BaseRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.svm.base import BaseLibSVM
from sklearn.linear_model.stochastic_gradient import BaseSGD
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import SkipTestWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import (rbf_kernel, linear_kernel,
                                      pairwise_distances)

from sklearn.utils import shuffle
from sklearn.utils.fixes import signature
from sklearn.utils.validation import has_fit_parameter, _num_samples
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_boston, make_blobs


BOSTON = None
CROSS_DECOMPOSITION = ['PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']
MULTI_OUTPUT = ['CCA', 'DecisionTreeRegressor', 'ElasticNet',
                'ExtraTreeRegressor', 'ExtraTreesRegressor', 'GaussianProcess',
                'GaussianProcessRegressor', 'TransformedTargetRegressor',
                'KNeighborsRegressor', 'KernelRidge', 'Lars', 'Lasso',
                'LassoLars', 'LinearRegression', 'MultiTaskElasticNet',
                'MultiTaskElasticNetCV', 'MultiTaskLasso', 'MultiTaskLassoCV',
                'OrthogonalMatchingPursuit', 'PLSCanonical', 'PLSRegression',
                'RANSACRegressor', 'RadiusNeighborsRegressor',
                'RandomForestRegressor', 'Ridge', 'RidgeCV']
```
### 60 - sklearn/utils/estimator_checks.py:

Start line: 449, End line: 469

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_sample_weights_pandas_series(name, estimator_orig):
    # check that estimators will accept a 'sample_weight' parameter of
    # type pandas.Series in the 'fit' function.
    estimator = clone(estimator_orig)
    if has_fit_parameter(estimator, "sample_weight"):
        try:
            import pandas as pd
            X = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]])
            X = pd.DataFrame(pairwise_estimator_convert_X(X, estimator_orig))
            y = pd.Series([1, 1, 1, 2, 2, 2])
            weights = pd.Series([1] * 6)
            try:
                estimator.fit(X, y, sample_weight=weights)
            except ValueError:
                raise ValueError("Estimator {0} raises error if "
                                 "'sample_weight' parameter is of "
                                 "type pandas.Series".format(name))
        except ImportError:
            raise SkipTest("pandas is not installed: not testing for "
                           "input of type pandas.Series to class weight.")
```
### 62 - sklearn/utils/estimator_checks.py:

Start line: 1066, End line: 1089

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_partial_fit_n_features(name, estimator_orig):
    # check if number of features changes between calls to partial_fit.
    if not hasattr(estimator_orig, 'partial_fit'):
        return
    estimator = clone(estimator_orig)
    X, y = make_blobs(n_samples=50, random_state=1)
    X -= X.min()

    try:
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return

    with assert_raises(ValueError,
                       msg="The estimator {} does not raise an"
                           " error when the number of features"
                           " changes between calls to "
                           "partial_fit.".format(name)):
        estimator.partial_fit(X[:, :-1], y)
```
### 64 - sklearn/utils/estimator_checks.py:

Start line: 1523, End line: 1565

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_classifiers(name, classifier_orig):
    if name == "NuSVC":
        # the sparse version has a parameter that doesn't do anything
        raise SkipTest("Not testing NuSVC class weight as it is ignored.")
    if name.endswith("NB"):
        # NaiveBayes classifiers have a somewhat different interface.
        # FIXME SOON!
        raise SkipTest

    for n_centers in [2, 3]:
        # create a very noisy dataset
        X, y = make_blobs(centers=n_centers, random_state=0, cluster_std=20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                            random_state=0)

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _is_pairwise(classifier_orig):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_centers = len(np.unique(y_train))

        if n_centers == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(
            class_weight=class_weight)
        if hasattr(classifier, "n_iter"):
            classifier.set_params(n_iter=100)
        if hasattr(classifier, "max_iter"):
            classifier.set_params(max_iter=1000)
        if hasattr(classifier, "min_weight_fraction_leaf"):
            classifier.set_params(min_weight_fraction_leaf=0.01)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # XXX: Generally can use 0.89 here. On Windows, LinearSVC gets
        #      0.88 (Issue #9111)
        assert_greater(np.mean(y_pred == 0), 0.87)
```
### 65 - sklearn/utils/estimator_checks.py:

Start line: 164, End line: 180

```python
def _yield_regressor_checks(name, regressor):
    # TODO: test with intercept
    # TODO: test with multiple responses
    # basic testing
    yield check_regressors_train
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    yield check_regressors_no_decision_function
    yield check_supervised_y_2d
    yield check_supervised_y_no_nan
    if name != 'CCA':
        # check that the regressor handles int input
        yield check_regressors_int
    if name != "GaussianProcessRegressor":
        # Test if NotFittedError is raised
        yield check_estimators_unfitted
    yield check_non_transformer_estimators_n_iter
```
### 69 - sklearn/utils/estimator_checks.py:

Start line: 646, End line: 671

```python
@ignore_warnings
def check_fit2d_1sample(name, estimator_orig):
    # Check that fitting a 2d array with only one sample either works or
    # returns an informative message. The error message should either mention
    # the number of samples or the number of classes.
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(1, 10))
    y = X[:, 0].astype(np.int)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)

    msgs = ["1 sample", "n_samples = 1", "n_samples=1", "one sample",
            "1 class", "one class"]

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if all(msg not in repr(e) for msg in msgs):
            raise e
```
### 76 - sklearn/utils/estimator_checks.py:

Start line: 1310, End line: 1326

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_estimators_fit_returns_self(name, estimator_orig):
    """Check if self is returned when calling fit"""
    X, y = make_blobs(random_state=0, n_samples=9, n_features=4)
    # some want non-negative input
    X -= X.min()
    if name == 'PowerTransformer':
        # Box-Cox requires positive, non-zero data
        X += 1
    X = pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)

    set_random_state(estimator)

    assert_true(estimator.fit(X, y) is estimator)
```
### 77 - sklearn/utils/estimator_checks.py:

Start line: 183, End line: 200

```python
def _yield_transformer_checks(name, transformer):
    # All transformers should either deal with sparse data or raise an
    # exception with type TypeError and an intelligible error message
    if name not in ['AdditiveChi2Sampler', 'Binarizer', 'Normalizer',
                    'PLSCanonical', 'PLSRegression', 'CCA', 'PLSSVD']:
        yield check_transformer_data_not_an_array
    # these don't actually fit the data, so don't raise errors
    if name not in ['AdditiveChi2Sampler', 'Binarizer',
                    'FunctionTransformer', 'Normalizer']:
        # basic tests
        yield check_transformer_general
        yield check_transformers_unfitted
    # Dependent on external solvers and hence accessing the iter
    # param is non-trivial.
    external_solver = ['Isomap', 'KernelPCA', 'LocallyLinearEmbedding',
                       'RandomizedLasso', 'LogisticRegressionCV']
    if name not in external_solver:
        yield check_transformer_n_iter
```
### 86 - sklearn/utils/estimator_checks.py:

Start line: 1166, End line: 1197

```python
@ignore_warnings(category=DeprecationWarning)
def check_classifiers_one_label(name, classifier_orig):
    error_string_fit = "Classifier can't train when only one class is present."
    error_string_predict = ("Classifier can't predict when only one class is "
                            "present.")
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 3))
    X_test = rnd.uniform(size=(10, 3))
    y = np.ones(10)
    # catch deprecation warnings
    with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
        classifier = clone(classifier_orig)
        # try to fit
        try:
            classifier.fit(X_train, y)
        except ValueError as e:
            if 'class' not in repr(e):
                print(error_string_fit, classifier, e)
                traceback.print_exc(file=sys.stdout)
                raise e
            else:
                return
        except Exception as exc:
            print(error_string_fit, classifier, exc)
            traceback.print_exc(file=sys.stdout)
            raise exc
        # predict
        try:
            assert_array_equal(classifier.predict(X_test), y)
        except Exception as exc:
            print(error_string_predict, classifier, exc)
            raise exc
```
### 87 - sklearn/utils/estimator_checks.py:

Start line: 1867, End line: 1890

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_transformer_n_iter(name, estimator_orig):
    # Test that transformers with a parameter max_iter, return the
    # attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        if name in CROSS_DECOMPOSITION:
            # Check using default data
            X = [[0., 0., 1.], [1., 0., 0.], [2., 2., 2.], [2., 5., 4.]]
            y_ = [[0.1, -0.2], [0.9, 1.1], [0.1, -0.5], [0.3, -0.2]]

        else:
            X, y_ = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                               random_state=0, n_features=2, cluster_std=0.1)
            X -= X.min() - 0.1
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        # These return a n_iter per component.
        if name in CROSS_DECOMPOSITION:
            for iter_ in estimator.n_iter_:
                assert_greater_equal(iter_, 1)
        else:
            assert_greater_equal(estimator.n_iter_, 1)
```
### 90 - sklearn/utils/estimator_checks.py:

Start line: 203, End line: 236

```python
def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ('WardAgglomeration', "FeatureAgglomeration"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield check_estimators_partial_fit_n_features
    yield check_non_transformer_estimators_n_iter


def _yield_all_checks(name, estimator):
    for check in _yield_non_meta_checks(name, estimator):
        yield check
    if is_classifier(estimator):
        for check in _yield_classifier_checks(name, estimator):
            yield check
    if is_regressor(estimator):
        for check in _yield_regressor_checks(name, estimator):
            yield check
    if isinstance(estimator, TransformerMixin):
        for check in _yield_transformer_checks(name, estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(name, estimator):
            yield check
    yield check_fit2d_predict1d
    if name != 'GaussianProcess':  # FIXME
        # XXX GaussianProcess deprecated in 0.20
        yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_fit1d
    yield check_get_params_invariance
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
```
### 97 - sklearn/utils/estimator_checks.py:

Start line: 1588, End line: 1619

```python
@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_class_weight_balanced_linear_classifier(name, Classifier):
    """Test class weights with non-contiguous class labels."""
    # this is run on classes, not instances, though this should be changed
    X = np.array([[-1.0, -1.0], [-1.0, 0], [-.8, -1.0],
                  [1.0, 1.0], [1.0, 0.0]])
    y = np.array([1, 1, 1, -1, -1])

    classifier = Classifier()

    if hasattr(classifier, "n_iter"):
        # This is a very small dataset, default n_iter are likely to prevent
        # convergence
        classifier.set_params(n_iter=1000)
    if hasattr(classifier, "max_iter"):
        classifier.set_params(max_iter=1000)
    set_random_state(classifier)

    # Let the model compute the class frequencies
    classifier.set_params(class_weight='balanced')
    coef_balanced = classifier.fit(X, y).coef_.copy()

    # Count each label occurrence to reweight manually
    n_samples = len(y)
    n_classes = float(len(np.unique(y)))

    class_weight = {1: n_samples / (np.sum(y == 1) * n_classes),
                    -1: n_samples / (np.sum(y == -1) * n_classes)}
    classifier.set_params(class_weight=class_weight)
    coef_manual = classifier.fit(X, y).coef_.copy()

    assert_allclose(coef_balanced, coef_manual)
```
### 103 - sklearn/utils/estimator_checks.py:

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

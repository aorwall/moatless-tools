# scikit-learn__scikit-learn-10428

* repo: scikit-learn/scikit-learn
* base_commit: `db127bd9693068a5b187d49d08738e690c5c7d98`

## Problem statement

Add common test to ensure all(predict(X[mask]) == predict(X)[mask])
I don't think we currently test that estimator predictions/transformations are invariant whether performed in batch or on subsets of a dataset. For some fitted estimator `est`, data `X` and any boolean mask `mask` of length `X.shape[0]`, we need:

```python
all(est.method(X[mask]) == est.method(X)[mask])
```
where `method` is any of {`predict`, `predict_proba`, `decision_function`, `score_samples`, `transform`}. Testing that predictions for individual samples match the predictions across the dataset might be sufficient. This should be added to common tests at `sklearn/utils/estimator_checks.py`

Indeed, #9174 reports that this is broken for one-vs-one classification. :'(
  


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

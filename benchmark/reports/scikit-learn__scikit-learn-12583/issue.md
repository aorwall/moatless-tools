# scikit-learn__scikit-learn-12583

* repo: scikit-learn/scikit-learn
* base_commit: `e8c6cb151cff869cf1b61bddd3c72841318501ab`

## Problem statement

add_indicator switch in imputers
For whatever imputers we have, but especially [SimpleImputer](http://scikit-learn.org/dev/modules/generated/sklearn.impute.SimpleImputer.html), we should have an `add_indicator` parameter, which simply stacks a [MissingIndicator](http://scikit-learn.org/dev/modules/generated/sklearn.impute.MissingIndicator.html) transform onto the output of the imputer's `transform`.


## Patch

```diff
diff --git a/sklearn/impute.py b/sklearn/impute.py
--- a/sklearn/impute.py
+++ b/sklearn/impute.py
@@ -141,13 +141,26 @@ class SimpleImputer(BaseEstimator, TransformerMixin):
         a new copy will always be made, even if `copy=False`:
 
         - If X is not an array of floating values;
-        - If X is encoded as a CSR matrix.
+        - If X is encoded as a CSR matrix;
+        - If add_indicator=True.
+
+    add_indicator : boolean, optional (default=False)
+        If True, a `MissingIndicator` transform will stack onto output
+        of the imputer's transform. This allows a predictive estimator
+        to account for missingness despite imputation. If a feature has no
+        missing values at fit/train time, the feature won't appear on
+        the missing indicator even if there are missing values at
+        transform/test time.
 
     Attributes
     ----------
     statistics_ : array of shape (n_features,)
         The imputation fill value for each feature.
 
+    indicator_ : :class:`sklearn.impute.MissingIndicator`
+        Indicator used to add binary indicators for missing values.
+        ``None`` if add_indicator is False.
+
     See also
     --------
     IterativeImputer : Multivariate imputation of missing values.
@@ -159,8 +172,8 @@ class SimpleImputer(BaseEstimator, TransformerMixin):
     >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
     >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
     ... # doctest: +NORMALIZE_WHITESPACE
-    SimpleImputer(copy=True, fill_value=None, missing_values=nan,
-           strategy='mean', verbose=0)
+    SimpleImputer(add_indicator=False, copy=True, fill_value=None,
+            missing_values=nan, strategy='mean', verbose=0)
     >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
     >>> print(imp_mean.transform(X))
     ... # doctest: +NORMALIZE_WHITESPACE
@@ -175,12 +188,13 @@ class SimpleImputer(BaseEstimator, TransformerMixin):
 
     """
     def __init__(self, missing_values=np.nan, strategy="mean",
-                 fill_value=None, verbose=0, copy=True):
+                 fill_value=None, verbose=0, copy=True, add_indicator=False):
         self.missing_values = missing_values
         self.strategy = strategy
         self.fill_value = fill_value
         self.verbose = verbose
         self.copy = copy
+        self.add_indicator = add_indicator
 
     def _validate_input(self, X):
         allowed_strategies = ["mean", "median", "most_frequent", "constant"]
@@ -272,6 +286,13 @@ def fit(self, X, y=None):
                                                self.missing_values,
                                                fill_value)
 
+        if self.add_indicator:
+            self.indicator_ = MissingIndicator(
+                missing_values=self.missing_values)
+            self.indicator_.fit(X)
+        else:
+            self.indicator_ = None
+
         return self
 
     def _sparse_fit(self, X, strategy, missing_values, fill_value):
@@ -285,7 +306,6 @@ def _sparse_fit(self, X, strategy, missing_values, fill_value):
             # for constant strategy, self.statistcs_ is used to store
             # fill_value in each column
             statistics.fill(fill_value)
-
         else:
             for i in range(X.shape[1]):
                 column = X.data[X.indptr[i]:X.indptr[i + 1]]
@@ -382,6 +402,9 @@ def transform(self, X):
             raise ValueError("X has %d features per sample, expected %d"
                              % (X.shape[1], self.statistics_.shape[0]))
 
+        if self.add_indicator:
+            X_trans_indicator = self.indicator_.transform(X)
+
         # Delete the invalid columns if strategy is not constant
         if self.strategy == "constant":
             valid_statistics = statistics
@@ -420,6 +443,10 @@ def transform(self, X):
 
             X[coordinates] = values
 
+        if self.add_indicator:
+            hstack = sparse.hstack if sparse.issparse(X) else np.hstack
+            X = hstack((X, X_trans_indicator))
+
         return X
 
     def _more_tags(self):

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_impute.py b/sklearn/tests/test_impute.py
--- a/sklearn/tests/test_impute.py
+++ b/sklearn/tests/test_impute.py
@@ -952,15 +952,15 @@ def test_missing_indicator_error(X_fit, X_trans, params, msg_err):
      ])
 @pytest.mark.parametrize(
     "param_features, n_features, features_indices",
-    [('missing-only', 2, np.array([0, 1])),
+    [('missing-only', 3, np.array([0, 1, 2])),
      ('all', 3, np.array([0, 1, 2]))])
 def test_missing_indicator_new(missing_values, arr_type, dtype, param_features,
                                n_features, features_indices):
     X_fit = np.array([[missing_values, missing_values, 1],
-                      [4, missing_values, 2]])
+                      [4, 2, missing_values]])
     X_trans = np.array([[missing_values, missing_values, 1],
                         [4, 12, 10]])
-    X_fit_expected = np.array([[1, 1, 0], [0, 1, 0]])
+    X_fit_expected = np.array([[1, 1, 0], [0, 0, 1]])
     X_trans_expected = np.array([[1, 1, 0], [0, 0, 0]])
 
     # convert the input to the right array format and right dtype
@@ -1144,3 +1144,54 @@ def test_missing_indicator_sparse_no_explicit_zeros():
     Xt = mi.fit_transform(X)
 
     assert Xt.getnnz() == Xt.sum()
+
+
+@pytest.mark.parametrize("marker", [np.nan, -1, 0])
+def test_imputation_add_indicator(marker):
+    X = np.array([
+        [marker, 1,      5,       marker, 1],
+        [2,      marker, 1,       marker, 2],
+        [6,      3,      marker,  marker, 3],
+        [1,      2,      9,       marker, 4]
+    ])
+    X_true = np.array([
+        [3., 1., 5., 1., 1., 0., 0., 1.],
+        [2., 2., 1., 2., 0., 1., 0., 1.],
+        [6., 3., 5., 3., 0., 0., 1., 1.],
+        [1., 2., 9., 4., 0., 0., 0., 1.]
+    ])
+
+    imputer = SimpleImputer(missing_values=marker, add_indicator=True)
+    X_trans = imputer.fit_transform(X)
+
+    assert_allclose(X_trans, X_true)
+    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))
+
+
+@pytest.mark.parametrize(
+    "arr_type",
+    [
+        sparse.csc_matrix, sparse.csr_matrix, sparse.coo_matrix,
+        sparse.lil_matrix, sparse.bsr_matrix
+    ]
+)
+def test_imputation_add_indicator_sparse_matrix(arr_type):
+    X_sparse = arr_type([
+        [np.nan, 1, 5],
+        [2, np.nan, 1],
+        [6, 3, np.nan],
+        [1, 2, 9]
+    ])
+    X_true = np.array([
+        [3., 1., 5., 1., 0., 0.],
+        [2., 2., 1., 0., 1., 0.],
+        [6., 3., 5., 0., 0., 1.],
+        [1., 2., 9., 0., 0., 0.],
+    ])
+
+    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)
+    X_trans = imputer.fit_transform(X_sparse)
+
+    assert sparse.issparse(X_trans)
+    assert X_trans.shape == X_true.shape
+    assert_allclose(X_trans.toarray(), X_true)

```

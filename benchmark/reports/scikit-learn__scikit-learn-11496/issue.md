# scikit-learn__scikit-learn-11496

* repo: scikit-learn/scikit-learn
* base_commit: `cb0140017740d985960911c4f34820beea915846`

## Problem statement

BUG: SimpleImputer gives wrong result on sparse matrix with explicit zeros
The current implementation of the `SimpleImputer` can't deal with zeros stored explicitly in sparse matrix.
Even when stored explicitly, we'd expect that all zeros are treating equally, right ?
See for example the code below:
```python
import numpy as np
from scipy import sparse
from sklearn.impute import SimpleImputer

X = np.array([[0,0,0],[0,0,0],[1,1,1]])
X = sparse.csc_matrix(X)
X[0] = 0    # explicit zeros in first row

imp = SimpleImputer(missing_values=0, strategy='mean')
imp.fit_transform(X)

>>> array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [1. , 1. , 1. ]])
```
Whereas the expected result would be
```python
>>> array([[1. , 1. , 1. ],
           [1. , 1. , 1. ],
           [1. , 1. , 1. ]])
```


## Patch

```diff
diff --git a/sklearn/impute.py b/sklearn/impute.py
--- a/sklearn/impute.py
+++ b/sklearn/impute.py
@@ -133,7 +133,6 @@ class SimpleImputer(BaseEstimator, TransformerMixin):
         a new copy will always be made, even if `copy=False`:
 
         - If X is not an array of floating values;
-        - If X is sparse and `missing_values=0`;
         - If X is encoded as a CSR matrix.
 
     Attributes
@@ -227,10 +226,17 @@ def fit(self, X, y=None):
                              "data".format(fill_value))
 
         if sparse.issparse(X):
-            self.statistics_ = self._sparse_fit(X,
-                                                self.strategy,
-                                                self.missing_values,
-                                                fill_value)
+            # missing_values = 0 not allowed with sparse data as it would
+            # force densification
+            if self.missing_values == 0:
+                raise ValueError("Imputation not possible when missing_values "
+                                 "== 0 and input is sparse. Provide a dense "
+                                 "array instead.")
+            else:
+                self.statistics_ = self._sparse_fit(X,
+                                                    self.strategy,
+                                                    self.missing_values,
+                                                    fill_value)
         else:
             self.statistics_ = self._dense_fit(X,
                                                self.strategy,
@@ -241,80 +247,41 @@ def fit(self, X, y=None):
 
     def _sparse_fit(self, X, strategy, missing_values, fill_value):
         """Fit the transformer on sparse data."""
-        # Count the zeros
-        if missing_values == 0:
-            n_zeros_axis = np.zeros(X.shape[1], dtype=int)
-        else:
-            n_zeros_axis = X.shape[0] - np.diff(X.indptr)
-
-        # Mean
-        if strategy == "mean":
-            if missing_values != 0:
-                n_non_missing = n_zeros_axis
-
-                # Mask the missing elements
-                mask_missing_values = _get_mask(X.data, missing_values)
-                mask_valids = np.logical_not(mask_missing_values)
-
-                # Sum only the valid elements
-                new_data = X.data.copy()
-                new_data[mask_missing_values] = 0
-                X = sparse.csc_matrix((new_data, X.indices, X.indptr),
-                                      copy=False)
-                sums = X.sum(axis=0)
-
-                # Count the elements != 0
-                mask_non_zeros = sparse.csc_matrix(
-                    (mask_valids.astype(np.float64),
-                     X.indices,
-                     X.indptr), copy=False)
-                s = mask_non_zeros.sum(axis=0)
-                n_non_missing = np.add(n_non_missing, s)
+        mask_data = _get_mask(X.data, missing_values)
+        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)
 
-            else:
-                sums = X.sum(axis=0)
-                n_non_missing = np.diff(X.indptr)
+        statistics = np.empty(X.shape[1])
 
-            # Ignore the error, columns with a np.nan statistics_
-            # are not an error at this point. These columns will
-            # be removed in transform
-            with np.errstate(all="ignore"):
-                return np.ravel(sums) / np.ravel(n_non_missing)
+        if strategy == "constant":
+            # for constant strategy, self.statistcs_ is used to store
+            # fill_value in each column
+            statistics.fill(fill_value)
 
-        # Median + Most frequent + Constant
         else:
-            # Remove the missing values, for each column
-            columns_all = np.hsplit(X.data, X.indptr[1:-1])
-            mask_missing_values = _get_mask(X.data, missing_values)
-            mask_valids = np.hsplit(np.logical_not(mask_missing_values),
-                                    X.indptr[1:-1])
-
-            # astype necessary for bug in numpy.hsplit before v1.9
-            columns = [col[mask.astype(bool, copy=False)]
-                       for col, mask in zip(columns_all, mask_valids)]
-
-            # Median
-            if strategy == "median":
-                median = np.empty(len(columns))
-                for i, column in enumerate(columns):
-                    median[i] = _get_median(column, n_zeros_axis[i])
-
-                return median
-
-            # Most frequent
-            elif strategy == "most_frequent":
-                most_frequent = np.empty(len(columns))
-
-                for i, column in enumerate(columns):
-                    most_frequent[i] = _most_frequent(column,
-                                                      0,
-                                                      n_zeros_axis[i])
-
-                return most_frequent
-
-            # Constant
-            elif strategy == "constant":
-                return np.full(X.shape[1], fill_value)
+            for i in range(X.shape[1]):
+                column = X.data[X.indptr[i]:X.indptr[i+1]]
+                mask_column = mask_data[X.indptr[i]:X.indptr[i+1]]
+                column = column[~mask_column]
+
+                # combine explicit and implicit zeros
+                mask_zeros = _get_mask(column, 0)
+                column = column[~mask_zeros]
+                n_explicit_zeros = mask_zeros.sum()
+                n_zeros = n_implicit_zeros[i] + n_explicit_zeros
+
+                if strategy == "mean":
+                    s = column.size + n_zeros
+                    statistics[i] = np.nan if s == 0 else column.sum() / s
+
+                elif strategy == "median":
+                    statistics[i] = _get_median(column,
+                                                n_zeros)
+
+                elif strategy == "most_frequent":
+                    statistics[i] = _most_frequent(column,
+                                                   0,
+                                                   n_zeros)
+        return statistics
 
     def _dense_fit(self, X, strategy, missing_values, fill_value):
         """Fit the transformer on dense data."""
@@ -364,6 +331,8 @@ def _dense_fit(self, X, strategy, missing_values, fill_value):
 
         # Constant
         elif strategy == "constant":
+            # for constant strategy, self.statistcs_ is used to store
+            # fill_value in each column
             return np.full(X.shape[1], fill_value, dtype=X.dtype)
 
     def transform(self, X):
@@ -402,17 +371,19 @@ def transform(self, X):
                 X = X[:, valid_statistics_indexes]
 
         # Do actual imputation
-        if sparse.issparse(X) and self.missing_values != 0:
-            mask = _get_mask(X.data, self.missing_values)
-            indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
-                                np.diff(X.indptr))[mask]
+        if sparse.issparse(X):
+            if self.missing_values == 0:
+                raise ValueError("Imputation not possible when missing_values "
+                                 "== 0 and input is sparse. Provide a dense "
+                                 "array instead.")
+            else:
+                mask = _get_mask(X.data, self.missing_values)
+                indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
+                                    np.diff(X.indptr))[mask]
 
-            X.data[mask] = valid_statistics[indexes].astype(X.dtype,
-                                                            copy=False)
+                X.data[mask] = valid_statistics[indexes].astype(X.dtype,
+                                                                copy=False)
         else:
-            if sparse.issparse(X):
-                X = X.toarray()
-
             mask = _get_mask(X, self.missing_values)
             n_missing = np.sum(mask, axis=0)
             values = np.repeat(valid_statistics, n_missing)

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_impute.py b/sklearn/tests/test_impute.py
--- a/sklearn/tests/test_impute.py
+++ b/sklearn/tests/test_impute.py
@@ -97,6 +97,23 @@ def test_imputation_deletion_warning(strategy):
         imputer.fit_transform(X)
 
 
+@pytest.mark.parametrize("strategy", ["mean", "median",
+                                      "most_frequent", "constant"])
+def test_imputation_error_sparse_0(strategy):
+    # check that error are raised when missing_values = 0 and input is sparse
+    X = np.ones((3, 5))
+    X[0] = 0
+    X = sparse.csc_matrix(X)
+
+    imputer = SimpleImputer(strategy=strategy, missing_values=0)
+    with pytest.raises(ValueError, match="Provide a dense array"):
+        imputer.fit(X)
+
+    imputer.fit(X.toarray())
+    with pytest.raises(ValueError, match="Provide a dense array"):
+        imputer.transform(X)
+
+
 def safe_median(arr, *args, **kwargs):
     # np.median([]) raises a TypeError for numpy >= 1.10.1
     length = arr.size if hasattr(arr, 'size') else len(arr)
@@ -123,10 +140,8 @@ def test_imputation_mean_median():
     values[4::2] = - values[4::2]
 
     tests = [("mean", np.nan, lambda z, v, p: safe_mean(np.hstack((z, v)))),
-             ("mean", 0, lambda z, v, p: np.mean(v)),
              ("median", np.nan,
-              lambda z, v, p: safe_median(np.hstack((z, v)))),
-             ("median", 0, lambda z, v, p: np.median(v))]
+              lambda z, v, p: safe_median(np.hstack((z, v))))]
 
     for strategy, test_missing_values, true_value_fun in tests:
         X = np.empty(shape)
@@ -427,14 +442,18 @@ def test_imputation_constant_pandas(dtype):
 
 def test_imputation_pipeline_grid_search():
     # Test imputation within a pipeline + gridsearch.
-    pipeline = Pipeline([('imputer', SimpleImputer(missing_values=0)),
-                         ('tree', tree.DecisionTreeRegressor(random_state=0))])
+    X = sparse_random_matrix(100, 100, density=0.10)
+    missing_values = X.data[0]
+
+    pipeline = Pipeline([('imputer',
+                          SimpleImputer(missing_values=missing_values)),
+                         ('tree',
+                          tree.DecisionTreeRegressor(random_state=0))])
 
     parameters = {
         'imputer__strategy': ["mean", "median", "most_frequent"]
     }
 
-    X = sparse_random_matrix(100, 100, density=0.10)
     Y = sparse_random_matrix(100, 1, density=0.10).toarray()
     gs = GridSearchCV(pipeline, parameters)
     gs.fit(X, Y)

```

# scikit-learn__scikit-learn-11315

* repo: scikit-learn/scikit-learn
* base_commit: `bb5110b8e0b70d98eae2f7f8b6d4deaa5d2de038`

## Problem statement

_BaseCompostion._set_params broken where there are no estimators
`_BaseCompostion._set_params` raises an error when the composition has no estimators.

This is a marginal case, but it might be interesting to support alongside #11315.


```py
>>> from sklearn.compose import ColumnTransformer
>>> ColumnTransformer([]).set_params(n_jobs=2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/joel/repos/scikit-learn/sklearn/compose/_column_transformer.py", line 181, in set_params
    self._set_params('_transformers', **kwargs)
  File "/Users/joel/repos/scikit-learn/sklearn/utils/metaestimators.py", line 44, in _set_params
    names, _ = zip(*getattr(self, attr))
ValueError: not enough values to unpack (expected 2, got 0)
```


## Patch

```diff
diff --git a/sklearn/compose/_column_transformer.py b/sklearn/compose/_column_transformer.py
--- a/sklearn/compose/_column_transformer.py
+++ b/sklearn/compose/_column_transformer.py
@@ -6,7 +6,7 @@
 # Author: Andreas Mueller
 #         Joris Van den Bossche
 # License: BSD
-
+from itertools import chain
 
 import numpy as np
 from scipy import sparse
@@ -69,7 +69,7 @@ class ColumnTransformer(_BaseComposition, TransformerMixin):
             ``transformer`` expects X to be a 1d array-like (vector),
             otherwise a 2d array will be passed to the transformer.
 
-    remainder : {'passthrough', 'drop'}, default 'passthrough'
+    remainder : {'passthrough', 'drop'} or estimator, default 'passthrough'
         By default, all remaining columns that were not specified in
         `transformers` will be automatically passed through (default of
         ``'passthrough'``). This subset of columns is concatenated with the
@@ -77,6 +77,9 @@ class ColumnTransformer(_BaseComposition, TransformerMixin):
         By using ``remainder='drop'``, only the specified columns in
         `transformers` are transformed and combined in the output, and the
         non-specified columns are dropped.
+        By setting ``remainder`` to be an estimator, the remaining
+        non-specified columns will use the ``remainder`` estimator. The
+        estimator must support `fit` and `transform`.
 
     n_jobs : int, optional
         Number of jobs to run in parallel (default 1).
@@ -90,7 +93,13 @@ class ColumnTransformer(_BaseComposition, TransformerMixin):
     ----------
     transformers_ : list
         The collection of fitted transformers as tuples of
-        (name, fitted_transformer, column).
+        (name, fitted_transformer, column). `fitted_transformer` can be an
+        estimator, 'drop', or 'passthrough'. If there are remaining columns,
+        the final element is a tuple of the form:
+        ('remainder', transformer, remaining_columns) corresponding to the
+        ``remainder`` parameter. If there are remaining columns, then
+        ``len(transformers_)==len(transformers)+1``, otherwise
+        ``len(transformers_)==len(transformers)``.
 
     named_transformers_ : Bunch object, a dictionary with attribute access
         Read-only attribute to access any transformer by given name.
@@ -188,13 +197,12 @@ def _iter(self, X=None, fitted=False, replace_strings=False):
             transformers = self.transformers_
         else:
             transformers = self.transformers
+            if self._remainder[2] is not None:
+                transformers = chain(transformers, [self._remainder])
         get_weight = (self.transformer_weights or {}).get
 
         for name, trans, column in transformers:
-            if X is None:
-                sub = X
-            else:
-                sub = _get_column(X, column)
+            sub = None if X is None else _get_column(X, column)
 
             if replace_strings:
                 # replace 'passthrough' with identity transformer and
@@ -209,7 +217,10 @@ def _iter(self, X=None, fitted=False, replace_strings=False):
             yield (name, trans, sub, get_weight(name))
 
     def _validate_transformers(self):
-        names, transformers, _, _ = zip(*self._iter())
+        if not self.transformers:
+            return
+
+        names, transformers, _ = zip(*self.transformers)
 
         # validate names
         self._validate_names(names)
@@ -226,24 +237,27 @@ def _validate_transformers(self):
                                 (t, type(t)))
 
     def _validate_remainder(self, X):
-        """Generate list of passthrough columns for 'remainder' case."""
-        if self.remainder not in ('drop', 'passthrough'):
+        """
+        Validates ``remainder`` and defines ``_remainder`` targeting
+        the remaining columns.
+        """
+        is_transformer = ((hasattr(self.remainder, "fit")
+                           or hasattr(self.remainder, "fit_transform"))
+                          and hasattr(self.remainder, "transform"))
+        if (self.remainder not in ('drop', 'passthrough')
+                and not is_transformer):
             raise ValueError(
-                "The remainder keyword needs to be one of 'drop' or "
-                "'passthrough'. {0:r} was passed instead")
+                "The remainder keyword needs to be one of 'drop', "
+                "'passthrough', or estimator. '%s' was passed instead" %
+                self.remainder)
 
         n_columns = X.shape[1]
+        cols = []
+        for _, _, columns in self.transformers:
+            cols.extend(_get_column_indices(X, columns))
+        remaining_idx = sorted(list(set(range(n_columns)) - set(cols))) or None
 
-        if self.remainder == 'passthrough':
-            cols = []
-            for _, _, columns in self.transformers:
-                cols.extend(_get_column_indices(X, columns))
-            self._passthrough = sorted(list(set(range(n_columns)) - set(cols)))
-            if not self._passthrough:
-                # empty list -> no need to select passthrough columns
-                self._passthrough = None
-        else:
-            self._passthrough = None
+        self._remainder = ('remainder', self.remainder, remaining_idx)
 
     @property
     def named_transformers_(self):
@@ -267,12 +281,6 @@ def get_feature_names(self):
             Names of the features produced by transform.
         """
         check_is_fitted(self, 'transformers_')
-        if self._passthrough is not None:
-            raise NotImplementedError(
-                "get_feature_names is not yet supported when having columns"
-                "that are passed through (you specify remainder='drop' to not "
-                "pass through the unspecified columns).")
-
         feature_names = []
         for name, trans, _, _ in self._iter(fitted=True):
             if trans == 'drop':
@@ -294,7 +302,11 @@ def _update_fitted_transformers(self, transformers):
         transformers = iter(transformers)
         transformers_ = []
 
-        for name, old, column in self.transformers:
+        transformer_iter = self.transformers
+        if self._remainder[2] is not None:
+            transformer_iter = chain(transformer_iter, [self._remainder])
+
+        for name, old, column in transformer_iter:
             if old == 'drop':
                 trans = 'drop'
             elif old == 'passthrough':
@@ -304,7 +316,6 @@ def _update_fitted_transformers(self, transformers):
                 trans = 'passthrough'
             else:
                 trans = next(transformers)
-
             transformers_.append((name, trans, column))
 
         # sanity check that transformers is exhausted
@@ -335,7 +346,7 @@ def _fit_transform(self, X, y, func, fitted=False):
             return Parallel(n_jobs=self.n_jobs)(
                 delayed(func)(clone(trans) if not fitted else trans,
                               X_sel, y, weight)
-                for name, trans, X_sel, weight in self._iter(
+                for _, trans, X_sel, weight in self._iter(
                     X=X, fitted=fitted, replace_strings=True))
         except ValueError as e:
             if "Expected 2D array, got 1D array instead" in str(e):
@@ -361,12 +372,12 @@ def fit(self, X, y=None):
             This estimator
 
         """
-        self._validate_transformers()
         self._validate_remainder(X)
+        self._validate_transformers()
 
         transformers = self._fit_transform(X, y, _fit_one_transformer)
-
         self._update_fitted_transformers(transformers)
+
         return self
 
     def fit_transform(self, X, y=None):
@@ -390,31 +401,21 @@ def fit_transform(self, X, y=None):
             sparse matrices.
 
         """
-        self._validate_transformers()
         self._validate_remainder(X)
+        self._validate_transformers()
 
         result = self._fit_transform(X, y, _fit_transform_one)
 
         if not result:
             # All transformers are None
-            if self._passthrough is None:
-                return np.zeros((X.shape[0], 0))
-            else:
-                return _get_column(X, self._passthrough)
+            return np.zeros((X.shape[0], 0))
 
         Xs, transformers = zip(*result)
 
         self._update_fitted_transformers(transformers)
         self._validate_output(Xs)
 
-        if self._passthrough is not None:
-            Xs = list(Xs) + [_get_column(X, self._passthrough)]
-
-        if any(sparse.issparse(f) for f in Xs):
-            Xs = sparse.hstack(Xs).tocsr()
-        else:
-            Xs = np.hstack(Xs)
-        return Xs
+        return _hstack(list(Xs))
 
     def transform(self, X):
         """Transform X separately by each transformer, concatenate results.
@@ -440,19 +441,9 @@ def transform(self, X):
 
         if not Xs:
             # All transformers are None
-            if self._passthrough is None:
-                return np.zeros((X.shape[0], 0))
-            else:
-                return _get_column(X, self._passthrough)
-
-        if self._passthrough is not None:
-            Xs = list(Xs) + [_get_column(X, self._passthrough)]
+            return np.zeros((X.shape[0], 0))
 
-        if any(sparse.issparse(f) for f in Xs):
-            Xs = sparse.hstack(Xs).tocsr()
-        else:
-            Xs = np.hstack(Xs)
-        return Xs
+        return _hstack(list(Xs))
 
 
 def _check_key_type(key, superclass):
@@ -486,6 +477,19 @@ def _check_key_type(key, superclass):
     return False
 
 
+def _hstack(X):
+    """
+    Stacks X horizontally.
+
+    Supports input types (X): list of
+        numpy arrays, sparse arrays and DataFrames
+    """
+    if any(sparse.issparse(f) for f in X):
+        return sparse.hstack(X).tocsr()
+    else:
+        return np.hstack(X)
+
+
 def _get_column(X, key):
     """
     Get feature column(s) from input data X.
@@ -612,7 +616,7 @@ def make_column_transformer(*transformers, **kwargs):
     ----------
     *transformers : tuples of column selections and transformers
 
-    remainder : {'passthrough', 'drop'}, default 'passthrough'
+    remainder : {'passthrough', 'drop'} or estimator, default 'passthrough'
         By default, all remaining columns that were not specified in
         `transformers` will be automatically passed through (default of
         ``'passthrough'``). This subset of columns is concatenated with the
@@ -620,6 +624,9 @@ def make_column_transformer(*transformers, **kwargs):
         By using ``remainder='drop'``, only the specified columns in
         `transformers` are transformed and combined in the output, and the
         non-specified columns are dropped.
+        By setting ``remainder`` to be an estimator, the remaining
+        non-specified columns will use the ``remainder`` estimator. The
+        estimator must support `fit` and `transform`.
 
     n_jobs : int, optional
         Number of jobs to run in parallel (default 1).
diff --git a/sklearn/utils/metaestimators.py b/sklearn/utils/metaestimators.py
--- a/sklearn/utils/metaestimators.py
+++ b/sklearn/utils/metaestimators.py
@@ -23,7 +23,7 @@ def __init__(self):
         pass
 
     def _get_params(self, attr, deep=True):
-        out = super(_BaseComposition, self).get_params(deep=False)
+        out = super(_BaseComposition, self).get_params(deep=deep)
         if not deep:
             return out
         estimators = getattr(self, attr)

```

## Test Patch

```diff
diff --git a/sklearn/compose/tests/test_column_transformer.py b/sklearn/compose/tests/test_column_transformer.py
--- a/sklearn/compose/tests/test_column_transformer.py
+++ b/sklearn/compose/tests/test_column_transformer.py
@@ -37,6 +37,14 @@ def transform(self, X, y=None):
         return X
 
 
+class DoubleTrans(BaseEstimator):
+    def fit(self, X, y=None):
+        return self
+
+    def transform(self, X):
+        return 2*X
+
+
 class SparseMatrixTrans(BaseEstimator):
     def fit(self, X, y=None):
         return self
@@ -46,6 +54,23 @@ def transform(self, X, y=None):
         return sparse.eye(n_samples, n_samples).tocsr()
 
 
+class TransNo2D(BaseEstimator):
+    def fit(self, X, y=None):
+        return self
+
+    def transform(self, X, y=None):
+        return X
+
+
+class TransRaise(BaseEstimator):
+
+    def fit(self, X, y=None):
+        raise ValueError("specific message")
+
+    def transform(self, X, y=None):
+        raise ValueError("specific message")
+
+
 def test_column_transformer():
     X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
 
@@ -78,6 +103,7 @@ def test_column_transformer():
                             ('trans2', Trans(), [1])])
     assert_array_equal(ct.fit_transform(X_array), X_res_both)
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
+    assert len(ct.transformers_) == 2
 
     # test with transformer_weights
     transformer_weights = {'trans1': .1, 'trans2': 10}
@@ -88,11 +114,13 @@ def test_column_transformer():
                      transformer_weights['trans2'] * X_res_second1D]).T
     assert_array_equal(both.fit_transform(X_array), res)
     assert_array_equal(both.fit(X_array).transform(X_array), res)
+    assert len(both.transformers_) == 2
 
     both = ColumnTransformer([('trans', Trans(), [0, 1])],
                              transformer_weights={'trans': .1})
     assert_array_equal(both.fit_transform(X_array), 0.1 * X_res_both)
     assert_array_equal(both.fit(X_array).transform(X_array), 0.1 * X_res_both)
+    assert len(both.transformers_) == 1
 
 
 def test_column_transformer_dataframe():
@@ -142,11 +170,15 @@ def test_column_transformer_dataframe():
                             ('trans2', Trans(), ['second'])])
     assert_array_equal(ct.fit_transform(X_df), X_res_both)
     assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     ct = ColumnTransformer([('trans1', Trans(), [0]),
                             ('trans2', Trans(), [1])])
     assert_array_equal(ct.fit_transform(X_df), X_res_both)
     assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # test with transformer_weights
     transformer_weights = {'trans1': .1, 'trans2': 10}
@@ -157,17 +189,23 @@ def test_column_transformer_dataframe():
                      transformer_weights['trans2'] * X_df['second']]).T
     assert_array_equal(both.fit_transform(X_df), res)
     assert_array_equal(both.fit(X_df).transform(X_df), res)
+    assert len(both.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # test multiple columns
     both = ColumnTransformer([('trans', Trans(), ['first', 'second'])],
                              transformer_weights={'trans': .1})
     assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
     assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
+    assert len(both.transformers_) == 1
+    assert ct.transformers_[-1][0] != 'remainder'
 
     both = ColumnTransformer([('trans', Trans(), [0, 1])],
                              transformer_weights={'trans': .1})
     assert_array_equal(both.fit_transform(X_df), 0.1 * X_res_both)
     assert_array_equal(both.fit(X_df).transform(X_df), 0.1 * X_res_both)
+    assert len(both.transformers_) == 1
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # ensure pandas object is passes through
 
@@ -195,6 +233,11 @@ def transform(self, X, y=None):
     assert_array_equal(ct.fit_transform(X_df), X_res_first)
     assert_array_equal(ct.fit(X_df).transform(X_df), X_res_first)
 
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'drop'
+    assert_array_equal(ct.transformers_[-1][2], [1])
+
 
 def test_column_transformer_sparse_array():
     X_sparse = sparse.eye(3, 2).tocsr()
@@ -230,6 +273,8 @@ def test_column_transformer_sparse_stacking():
     assert_true(sparse.issparse(X_trans))
     assert_equal(X_trans.shape, (X_trans.shape[0], X_trans.shape[0] + 1))
     assert_array_equal(X_trans.toarray()[:, 1:], np.eye(X_trans.shape[0]))
+    assert len(col_trans.transformers_) == 2
+    assert col_trans.transformers_[-1][0] != 'remainder'
 
 
 def test_column_transformer_error_msg_1D():
@@ -241,28 +286,12 @@ def test_column_transformer_error_msg_1D():
     assert_raise_message(ValueError, "1D data passed to a transformer",
                          col_trans.fit_transform, X_array)
 
-    class TransRaise(BaseEstimator):
-
-        def fit(self, X, y=None):
-            raise ValueError("specific message")
-
-        def transform(self, X, y=None):
-            raise ValueError("specific message")
-
     col_trans = ColumnTransformer([('trans', TransRaise(), 0)])
     for func in [col_trans.fit, col_trans.fit_transform]:
         assert_raise_message(ValueError, "specific message", func, X_array)
 
 
 def test_2D_transformer_output():
-
-    class TransNo2D(BaseEstimator):
-        def fit(self, X, y=None):
-            return self
-
-        def transform(self, X, y=None):
-            return X
-
     X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
 
     # if one transformer is dropped, test that name is still correct
@@ -278,13 +307,6 @@ def transform(self, X, y=None):
 def test_2D_transformer_output_pandas():
     pd = pytest.importorskip('pandas')
 
-    class TransNo2D(BaseEstimator):
-        def fit(self, X, y=None):
-            return self
-
-        def transform(self, X, y=None):
-            return X
-
     X_array = np.array([[0, 1, 2], [2, 4, 6]]).T
     X_df = pd.DataFrame(X_array, columns=['col1', 'col2'])
 
@@ -344,10 +366,8 @@ def test_make_column_transformer_kwargs():
     norm = Normalizer()
     ct = make_column_transformer(('first', scaler), (['second'], norm),
                                  n_jobs=3, remainder='drop')
-    assert_equal(
-        ct.transformers,
-        make_column_transformer(('first', scaler),
-                                (['second'], norm)).transformers)
+    assert_equal(ct.transformers, make_column_transformer(
+        ('first', scaler), (['second'], norm)).transformers)
     assert_equal(ct.n_jobs, 3)
     assert_equal(ct.remainder, 'drop')
     # invalid keyword parameters should raise an error message
@@ -359,6 +379,15 @@ def test_make_column_transformer_kwargs():
     )
 
 
+def test_make_column_transformer_remainder_transformer():
+    scaler = StandardScaler()
+    norm = Normalizer()
+    remainder = StandardScaler()
+    ct = make_column_transformer(('first', scaler), (['second'], norm),
+                                 remainder=remainder)
+    assert ct.remainder == remainder
+
+
 def test_column_transformer_get_set_params():
     ct = ColumnTransformer([('trans1', StandardScaler(), [0]),
                             ('trans2', StandardScaler(), [1])])
@@ -473,12 +502,16 @@ def test_column_transformer_special_strings():
     exp = np.array([[0.], [1.], [2.]])
     assert_array_equal(ct.fit_transform(X_array), exp)
     assert_array_equal(ct.fit(X_array).transform(X_array), exp)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # all 'drop' -> return shape 0 array
     ct = ColumnTransformer(
         [('trans1', 'drop', [0]), ('trans2', 'drop', [1])])
     assert_array_equal(ct.fit(X_array).transform(X_array).shape, (3, 0))
     assert_array_equal(ct.fit_transform(X_array).shape, (3, 0))
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # 'passthrough'
     X_array = np.array([[0., 1., 2.], [2., 4., 6.]]).T
@@ -487,6 +520,8 @@ def test_column_transformer_special_strings():
     exp = X_array
     assert_array_equal(ct.fit_transform(X_array), exp)
     assert_array_equal(ct.fit(X_array).transform(X_array), exp)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] != 'remainder'
 
     # None itself / other string is not valid
     for val in [None, 'other']:
@@ -509,35 +544,51 @@ def test_column_transformer_remainder():
     ct = ColumnTransformer([('trans', Trans(), [0])])
     assert_array_equal(ct.fit_transform(X_array), X_res_both)
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'passthrough'
+    assert_array_equal(ct.transformers_[-1][2], [1])
 
     # specify to drop remaining columns
     ct = ColumnTransformer([('trans1', Trans(), [0])],
                            remainder='drop')
     assert_array_equal(ct.fit_transform(X_array), X_res_first)
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_first)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'drop'
+    assert_array_equal(ct.transformers_[-1][2], [1])
 
     # column order is not preserved (passed through added to end)
     ct = ColumnTransformer([('trans1', Trans(), [1])],
                            remainder='passthrough')
     assert_array_equal(ct.fit_transform(X_array), X_res_both[:, ::-1])
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both[:, ::-1])
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'passthrough'
+    assert_array_equal(ct.transformers_[-1][2], [0])
 
     # passthrough when all actual transformers are skipped
     ct = ColumnTransformer([('trans1', 'drop', [0])],
                            remainder='passthrough')
     assert_array_equal(ct.fit_transform(X_array), X_res_second)
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_second)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'passthrough'
+    assert_array_equal(ct.transformers_[-1][2], [1])
 
     # error on invalid arg
     ct = ColumnTransformer([('trans1', Trans(), [0])], remainder=1)
     assert_raise_message(
         ValueError,
-        "remainder keyword needs to be one of \'drop\' or \'passthrough\'",
-        ct.fit, X_array)
+        "remainder keyword needs to be one of \'drop\', \'passthrough\', "
+        "or estimator.", ct.fit, X_array)
     assert_raise_message(
         ValueError,
-        "remainder keyword needs to be one of \'drop\' or \'passthrough\'",
-        ct.fit_transform, X_array)
+        "remainder keyword needs to be one of \'drop\', \'passthrough\', "
+        "or estimator.", ct.fit_transform, X_array)
 
 
 @pytest.mark.parametrize("key", [[0], np.array([0]), slice(0, 1),
@@ -551,6 +602,10 @@ def test_column_transformer_remainder_numpy(key):
                            remainder='passthrough')
     assert_array_equal(ct.fit_transform(X_array), X_res_both)
     assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'passthrough'
+    assert_array_equal(ct.transformers_[-1][2], [1])
 
 
 @pytest.mark.parametrize(
@@ -571,3 +626,154 @@ def test_column_transformer_remainder_pandas(key):
                            remainder='passthrough')
     assert_array_equal(ct.fit_transform(X_df), X_res_both)
     assert_array_equal(ct.fit(X_df).transform(X_df), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][1] == 'passthrough'
+    assert_array_equal(ct.transformers_[-1][2], [1])
+
+
+@pytest.mark.parametrize("key", [[0], np.array([0]), slice(0, 1),
+                                 np.array([True, False, False])])
+def test_column_transformer_remainder_transformer(key):
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).T
+    X_res_both = X_array.copy()
+
+    # second and third columns are doubled when remainder = DoubleTrans
+    X_res_both[:, 1:3] *= 2
+
+    ct = ColumnTransformer([('trans1', Trans(), key)],
+                           remainder=DoubleTrans())
+
+    assert_array_equal(ct.fit_transform(X_array), X_res_both)
+    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
+    assert_array_equal(ct.transformers_[-1][2], [1, 2])
+
+
+def test_column_transformer_no_remaining_remainder_transformer():
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).T
+
+    ct = ColumnTransformer([('trans1', Trans(), [0, 1, 2])],
+                           remainder=DoubleTrans())
+
+    assert_array_equal(ct.fit_transform(X_array), X_array)
+    assert_array_equal(ct.fit(X_array).transform(X_array), X_array)
+    assert len(ct.transformers_) == 1
+    assert ct.transformers_[-1][0] != 'remainder'
+
+
+def test_column_transformer_drops_all_remainder_transformer():
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).T
+
+    # columns are doubled when remainder = DoubleTrans
+    X_res_both = 2 * X_array.copy()[:, 1:3]
+
+    ct = ColumnTransformer([('trans1', 'drop', [0])],
+                           remainder=DoubleTrans())
+
+    assert_array_equal(ct.fit_transform(X_array), X_res_both)
+    assert_array_equal(ct.fit(X_array).transform(X_array), X_res_both)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert isinstance(ct.transformers_[-1][1], DoubleTrans)
+    assert_array_equal(ct.transformers_[-1][2], [1, 2])
+
+
+def test_column_transformer_sparse_remainder_transformer():
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).T
+
+    ct = ColumnTransformer([('trans1', Trans(), [0])],
+                           remainder=SparseMatrixTrans())
+
+    X_trans = ct.fit_transform(X_array)
+    assert sparse.issparse(X_trans)
+    # SparseMatrixTrans creates 3 features for each column. There is
+    # one column in ``transformers``, thus:
+    assert X_trans.shape == (3, 3 + 1)
+
+    exp_array = np.hstack(
+        (X_array[:, 0].reshape(-1, 1), np.eye(3)))
+    assert_array_equal(X_trans.toarray(), exp_array)
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)
+    assert_array_equal(ct.transformers_[-1][2], [1, 2])
+
+
+def test_column_transformer_drop_all_sparse_remainder_transformer():
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).T
+    ct = ColumnTransformer([('trans1', 'drop', [0])],
+                           remainder=SparseMatrixTrans())
+
+    X_trans = ct.fit_transform(X_array)
+    assert sparse.issparse(X_trans)
+
+    #  SparseMatrixTrans creates 3 features for each column, thus:
+    assert X_trans.shape == (3, 3)
+    assert_array_equal(X_trans.toarray(), np.eye(3))
+    assert len(ct.transformers_) == 2
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert isinstance(ct.transformers_[-1][1], SparseMatrixTrans)
+    assert_array_equal(ct.transformers_[-1][2], [1, 2])
+
+
+def test_column_transformer_get_set_params_with_remainder():
+    ct = ColumnTransformer([('trans1', StandardScaler(), [0])],
+                           remainder=StandardScaler())
+
+    exp = {'n_jobs': 1,
+           'remainder': ct.remainder,
+           'remainder__copy': True,
+           'remainder__with_mean': True,
+           'remainder__with_std': True,
+           'trans1': ct.transformers[0][1],
+           'trans1__copy': True,
+           'trans1__with_mean': True,
+           'trans1__with_std': True,
+           'transformers': ct.transformers,
+           'transformer_weights': None}
+
+    assert ct.get_params() == exp
+
+    ct.set_params(remainder__with_std=False)
+    assert not ct.get_params()['remainder__with_std']
+
+    ct.set_params(trans1='passthrough')
+    exp = {'n_jobs': 1,
+           'remainder': ct.remainder,
+           'remainder__copy': True,
+           'remainder__with_mean': True,
+           'remainder__with_std': False,
+           'trans1': 'passthrough',
+           'transformers': ct.transformers,
+           'transformer_weights': None}
+
+    assert ct.get_params() == exp
+
+
+def test_column_transformer_no_estimators():
+    X_array = np.array([[0, 1, 2],
+                        [2, 4, 6],
+                        [8, 6, 4]]).astype('float').T
+    ct = ColumnTransformer([], remainder=StandardScaler())
+
+    params = ct.get_params()
+    assert params['remainder__with_mean']
+
+    X_trans = ct.fit_transform(X_array)
+    assert X_trans.shape == X_array.shape
+    assert len(ct.transformers_) == 1
+    assert ct.transformers_[-1][0] == 'remainder'
+    assert ct.transformers_[-1][2] == [0, 1, 2]

```

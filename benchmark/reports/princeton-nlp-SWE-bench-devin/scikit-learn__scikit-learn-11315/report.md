# scikit-learn__scikit-learn-11315

| **scikit-learn/scikit-learn** | `bb5110b8e0b70d98eae2f7f8b6d4deaa5d2de038` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 1258 |
| **Any found context length** | 725 |
| **Avg pos** | 141.5 |
| **Min pos** | 3 |
| **Max pos** | 61 |
| **Top file pos** | 1 |
| **Missing snippets** | 18 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/compose/_column_transformer.py | 9 | 9 | 7 | 1 | 1448
| sklearn/compose/_column_transformer.py | 72 | 72 | 9 | 1 | 2566
| sklearn/compose/_column_transformer.py | 80 | 80 | 9 | 1 | 2566
| sklearn/compose/_column_transformer.py | 93 | 93 | 9 | 1 | 2566
| sklearn/compose/_column_transformer.py | 191 | 194 | 37 | 1 | 12547
| sklearn/compose/_column_transformer.py | 212 | 212 | 3 | 1 | 725
| sklearn/compose/_column_transformer.py | 229 | 246 | 12 | 1 | 3493
| sklearn/compose/_column_transformer.py | 270 | 275 | 61 | 1 | 20569
| sklearn/compose/_column_transformer.py | 297 | 297 | 5 | 1 | 1035
| sklearn/compose/_column_transformer.py | 307 | 307 | 5 | 1 | 1035
| sklearn/compose/_column_transformer.py | 338 | 338 | 25 | 1 | 8170
| sklearn/compose/_column_transformer.py | 364 | 368 | 19 | 1 | 6116
| sklearn/compose/_column_transformer.py | 393 | 417 | 40 | 1 | 13717
| sklearn/compose/_column_transformer.py | 443 | 455 | - | 1 | -
| sklearn/compose/_column_transformer.py | 489 | 489 | - | 1 | -
| sklearn/compose/_column_transformer.py | 615 | 615 | 18 | 1 | 5966
| sklearn/compose/_column_transformer.py | 623 | 623 | 18 | 1 | 5966
| sklearn/utils/metaestimators.py | 26 | 26 | 6 | 2 | 1258


## Problem Statement

```
_BaseCompostion._set_params broken where there are no estimators
`_BaseCompostion._set_params` raises an error when the composition has no estimators.

This is a marginal case, but it might be interesting to support alongside #11315.


\`\`\`py
>>> from sklearn.compose import ColumnTransformer
>>> ColumnTransformer([]).set_params(n_jobs=2)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/joel/repos/scikit-learn/sklearn/compose/_column_transformer.py", line 181, in set_params
    self._set_params('_transformers', **kwargs)
  File "/Users/joel/repos/scikit-learn/sklearn/utils/metaestimators.py", line 44, in _set_params
    names, _ = zip(*getattr(self, attr))
ValueError: not enough values to unpack (expected 2, got 0)
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/compose/_column_transformer.py** | 133 | 182| 339 | 339 | 5017 | 
| 2 | **2 sklearn/utils/metaestimators.py** | 38 | 59| 233 | 572 | 6669 | 
| **-> 3 <-** | **2 sklearn/compose/_column_transformer.py** | 211 | 226| 153 | 725 | 6669 | 
| 4 | **2 sklearn/utils/metaestimators.py** | 61 | 72| 151 | 876 | 6669 | 
| **-> 5 <-** | **2 sklearn/compose/_column_transformer.py** | 292 | 312| 159 | 1035 | 6669 | 
| **-> 6 <-** | **2 sklearn/utils/metaestimators.py** | 1 | 36| 223 | 1258 | 6669 | 
| **-> 7 <-** | **2 sklearn/compose/_column_transformer.py** | 1 | 30| 190 | 1448 | 6669 | 
| 8 | 3 sklearn/ensemble/base.py | 100 | 117| 181 | 1629 | 7816 | 
| **-> 9 <-** | **3 sklearn/compose/_column_transformer.py** | 33 | 131| 937 | 2566 | 7816 | 
| 10 | 4 sklearn/utils/estimator_checks.py | 1842 | 1877| 352 | 2918 | 27495 | 
| 11 | 4 sklearn/utils/estimator_checks.py | 2090 | 2113| 308 | 3226 | 27495 | 
| **-> 12 <-** | **4 sklearn/compose/_column_transformer.py** | 228 | 259| 267 | 3493 | 27495 | 
| 13 | 5 sklearn/decomposition/online_lda.py | 288 | 312| 220 | 3713 | 34153 | 
| 14 | 6 sklearn/base.py | 236 | 271| 264 | 3977 | 38446 | 
| 15 | 7 sklearn/ensemble/voting_classifier.py | 312 | 335| 193 | 4170 | 41319 | 
| 16 | 7 sklearn/utils/estimator_checks.py | 323 | 375| 509 | 4679 | 41319 | 
| 17 | 8 sklearn/ensemble/gradient_boosting.py | 810 | 893| 806 | 5485 | 59583 | 
| **-> 18 <-** | **8 sklearn/compose/_column_transformer.py** | 603 | 663| 481 | 5966 | 59583 | 
| **-> 19 <-** | **8 sklearn/compose/_column_transformer.py** | 346 | 370| 150 | 6116 | 59583 | 
| 20 | 9 sklearn/compose/_target.py | 104 | 140| 411 | 6527 | 61493 | 
| 21 | 9 sklearn/utils/estimator_checks.py | 1047 | 1115| 601 | 7128 | 61493 | 
| 22 | 9 sklearn/utils/estimator_checks.py | 1972 | 2043| 578 | 7706 | 61493 | 
| 23 | 10 sklearn/decomposition/nmf.py | 1190 | 1204| 151 | 7857 | 73187 | 
| 24 | **10 sklearn/compose/_column_transformer.py** | 314 | 324| 137 | 7994 | 73187 | 
| **-> 25 <-** | **10 sklearn/compose/_column_transformer.py** | 326 | 344| 176 | 8170 | 73187 | 
| 26 | 11 sklearn/pipeline.py | 150 | 188| 288 | 8458 | 79780 | 
| 27 | 11 sklearn/utils/estimator_checks.py | 856 | 930| 702 | 9160 | 79780 | 
| 28 | 11 sklearn/ensemble/gradient_boosting.py | 926 | 955| 321 | 9481 | 79780 | 
| 29 | 11 sklearn/base.py | 175 | 211| 314 | 9795 | 79780 | 
| 30 | 11 sklearn/utils/estimator_checks.py | 195 | 213| 233 | 10028 | 79780 | 
| 31 | 11 sklearn/utils/estimator_checks.py | 831 | 853| 248 | 10276 | 79780 | 
| 32 | 11 sklearn/utils/estimator_checks.py | 2046 | 2087| 400 | 10676 | 79780 | 
| 33 | 11 sklearn/utils/estimator_checks.py | 603 | 651| 441 | 11117 | 79780 | 
| 34 | 11 sklearn/utils/estimator_checks.py | 1158 | 1181| 185 | 11302 | 79780 | 
| 35 | 11 sklearn/ensemble/voting_classifier.py | 124 | 203| 684 | 11986 | 79780 | 
| 36 | 11 sklearn/ensemble/base.py | 1 | 57| 375 | 12361 | 79780 | 
| **-> 37 <-** | **11 sklearn/compose/_column_transformer.py** | 184 | 209| 186 | 12547 | 79780 | 
| 38 | 12 sklearn/ensemble/bagging.py | 63 | 120| 446 | 12993 | 87712 | 
| 39 | 13 examples/compose/plot_column_transformer.py | 1 | 55| 376 | 13369 | 88691 | 
| **-> 40 <-** | **13 sklearn/compose/_column_transformer.py** | 372 | 417| 348 | 13717 | 88691 | 
| 41 | 13 sklearn/utils/estimator_checks.py | 1880 | 1906| 252 | 13969 | 88691 | 
| 42 | 13 sklearn/ensemble/base.py | 60 | 98| 279 | 14248 | 88691 | 
| 43 | 13 sklearn/ensemble/base.py | 150 | 162| 122 | 14370 | 88691 | 
| 44 | 14 examples/compose/plot_column_transformer_mixed_types.py | 1 | 108| 816 | 15186 | 89529 | 
| 45 | 15 benchmarks/bench_plot_lasso_path.py | 1 | 82| 625 | 15811 | 90502 | 
| 46 | 16 sklearn/impute.py | 242 | 317| 604 | 16415 | 98389 | 
| 47 | 16 examples/compose/plot_column_transformer.py | 88 | 137| 358 | 16773 | 98389 | 
| 48 | 16 sklearn/pipeline.py | 115 | 148| 187 | 16960 | 98389 | 
| 49 | 17 sklearn/preprocessing/imputation.py | 173 | 251| 675 | 17635 | 101331 | 
| 50 | 17 sklearn/impute.py | 1 | 40| 206 | 17841 | 101331 | 
| 51 | 17 sklearn/utils/estimator_checks.py | 84 | 119| 278 | 18119 | 101331 | 
| 52 | 17 sklearn/utils/estimator_checks.py | 1949 | 1969| 200 | 18319 | 101331 | 
| 53 | 17 sklearn/utils/estimator_checks.py | 933 | 967| 372 | 18691 | 101331 | 
| 54 | 18 sklearn/utils/testing.py | 638 | 663| 260 | 18951 | 108213 | 
| 55 | 19 benchmarks/bench_mnist.py | 85 | 106| 314 | 19265 | 109944 | 
| 56 | 19 sklearn/impute.py | 533 | 556| 183 | 19448 | 109944 | 
| 57 | 19 sklearn/pipeline.py | 688 | 711| 195 | 19643 | 109944 | 
| 58 | 19 sklearn/utils/estimator_checks.py | 272 | 320| 368 | 20011 | 109944 | 
| 59 | 19 sklearn/pipeline.py | 515 | 533| 142 | 20153 | 109944 | 
| 60 | 19 sklearn/utils/estimator_checks.py | 814 | 828| 169 | 20322 | 109944 | 
| **-> 61 <-** | **19 sklearn/compose/_column_transformer.py** | 261 | 290| 247 | 20569 | 109944 | 
| 62 | 20 sklearn/preprocessing/data.py | 10 | 66| 347 | 20916 | 136127 | 
| 63 | 20 sklearn/utils/estimator_checks.py | 2116 | 2149| 233 | 21149 | 136127 | 
| 64 | 20 sklearn/base.py | 67 | 121| 486 | 21635 | 136127 | 
| 65 | 20 sklearn/utils/estimator_checks.py | 696 | 731| 343 | 21978 | 136127 | 
| 66 | 20 sklearn/ensemble/base.py | 119 | 147| 224 | 22202 | 136127 | 
| 67 | 20 sklearn/utils/estimator_checks.py | 1024 | 1044| 257 | 22459 | 136127 | 
| 68 | 21 sklearn/preprocessing/_function_transformer.py | 167 | 185| 170 | 22629 | 137515 | 
| 69 | 21 sklearn/utils/estimator_checks.py | 762 | 793| 297 | 22926 | 137515 | 
| 70 | **21 sklearn/compose/_column_transformer.py** | 541 | 600| 431 | 23357 | 137515 | 
| 71 | 21 sklearn/utils/estimator_checks.py | 1118 | 1155| 328 | 23685 | 137515 | 
| 72 | 21 sklearn/ensemble/bagging.py | 551 | 580| 197 | 23882 | 137515 | 
| 73 | 21 sklearn/pipeline.py | 190 | 225| 341 | 24223 | 137515 | 
| 74 | 21 sklearn/utils/estimator_checks.py | 437 | 478| 427 | 24650 | 137515 | 
| 75 | 21 sklearn/pipeline.py | 1 | 26| 129 | 24779 | 137515 | 
| 76 | 22 sklearn/kernel_approximation.py | 519 | 538| 175 | 24954 | 141639 | 
| 77 | 22 sklearn/base.py | 289 | 302| 151 | 25105 | 141639 | 
| 78 | 23 sklearn/cross_decomposition/pls_.py | 221 | 233| 150 | 25255 | 149650 | 
| 79 | 23 sklearn/utils/estimator_checks.py | 998 | 1021| 269 | 25524 | 149650 | 
| 80 | 23 sklearn/utils/estimator_checks.py | 521 | 556| 328 | 25852 | 149650 | 
| 81 | 23 sklearn/kernel_approximation.py | 263 | 279| 166 | 26018 | 149650 | 
| 82 | 23 sklearn/decomposition/online_lda.py | 264 | 286| 264 | 26282 | 149650 | 
| 83 | 24 benchmarks/bench_covertype.py | 100 | 110| 151 | 26433 | 151553 | 
| 84 | 24 sklearn/ensemble/bagging.py | 984 | 1014| 248 | 26681 | 151553 | 
| 85 | 24 sklearn/utils/estimator_checks.py | 734 | 759| 234 | 26915 | 151553 | 
| 86 | 24 sklearn/utils/estimator_checks.py | 1909 | 1929| 233 | 27148 | 151553 | 
| 87 | 25 sklearn/model_selection/_split.py | 2058 | 2092| 319 | 27467 | 169572 | 
| 88 | 25 sklearn/pipeline.py | 611 | 686| 592 | 28059 | 169572 | 
| 89 | 26 sklearn/ensemble/weight_boosting.py | 414 | 430| 188 | 28247 | 178350 | 
| 90 | 27 sklearn/multiclass.py | 183 | 217| 352 | 28599 | 184837 | 
| 91 | 27 sklearn/utils/estimator_checks.py | 1 | 81| 694 | 29293 | 184837 | 
| 92 | 27 sklearn/utils/estimator_checks.py | 559 | 600| 334 | 29627 | 184837 | 
| 93 | 27 sklearn/utils/estimator_checks.py | 152 | 172| 203 | 29830 | 184837 | 
| 94 | 27 sklearn/ensemble/bagging.py | 342 | 386| 377 | 30207 | 184837 | 
| 95 | 28 sklearn/metrics/scorer.py | 348 | 405| 538 | 30745 | 189757 | 
| 96 | 29 sklearn/compose/__init__.py | 1 | 17| 0 | 30745 | 189843 | 
| 97 | 29 sklearn/compose/_target.py | 5 | 103| 815 | 31560 | 189843 | 
| 98 | 30 sklearn/mixture/base.py | 88 | 122| 289 | 31849 | 193414 | 
| 99 | 31 sklearn/isotonic.py | 234 | 250| 145 | 31994 | 196792 | 


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


## Code snippets

### 1 - sklearn/compose/_column_transformer.py:

Start line: 133, End line: 182

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def __init__(self, transformers, remainder='passthrough', n_jobs=1,
                 transformer_weights=None):
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights

    @property
    def _transformers(self):
        """
        Internal list of transformer only containing the name and
        transformers, dropping the columns. This is for the implementation
        of get_params via BaseComposition._get_params which expects lists
        of tuples of len 2.
        """
        return [(name, trans) for name, trans, _ in self.transformers]

    @_transformers.setter
    def _transformers(self, value):
        self.transformers = [
            (name, trans, col) for ((name, trans), (_, _, col))
            in zip(value, self.transformers)]

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('_transformers', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params('_transformers', **kwargs)
        return self
```
### 2 - sklearn/utils/metaestimators.py:

Start line: 38, End line: 59

```python
class _BaseComposition(six.with_metaclass(ABCMeta, BaseEstimator)):

    def _set_params(self, attr, **params):
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        names, _ = zip(*getattr(self, attr))
        for name in list(six.iterkeys(params)):
            if '__' not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super(_BaseComposition, self).set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)
```
### 3 - sklearn/compose/_column_transformer.py:

Start line: 211, End line: 226

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _validate_transformers(self):
        names, transformers, _, _ = zip(*self._iter())

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All estimators should implement fit and "
                                "transform, or can be 'drop' or 'passthrough' "
                                "specifiers. '%s' (type %s) doesn't." %
                                (t, type(t)))
```
### 4 - sklearn/utils/metaestimators.py:

Start line: 61, End line: 72

```python
class _BaseComposition(six.with_metaclass(ABCMeta, BaseEstimator)):

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))
```
### 5 - sklearn/compose/_column_transformer.py:

Start line: 292, End line: 312

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _update_fitted_transformers(self, transformers):
        # transformers are fitted; excludes 'drop' cases
        transformers = iter(transformers)
        transformers_ = []

        for name, old, column in self.transformers:
            if old == 'drop':
                trans = 'drop'
            elif old == 'passthrough':
                # FunctionTransformer is present in list of transformers,
                # so get next transformer, but save original string
                next(transformers)
                trans = 'passthrough'
            else:
                trans = next(transformers)

            transformers_.append((name, trans, column))

        # sanity check that transformers is exhausted
        assert not list(transformers)
        self.transformers_ = transformers_
```
### 6 - sklearn/utils/metaestimators.py:

Start line: 1, End line: 36

```python
"""Utilities for meta-estimators"""

from abc import ABCMeta, abstractmethod
from operator import attrgetter
from functools import update_wrapper
import numpy as np

from ..utils import safe_indexing
from ..externals import six
from ..base import BaseEstimator

__all__ = ['if_delegate_has_method']


class _BaseComposition(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Handles parameter management for classifiers composed of named estimators.
    """
    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super(_BaseComposition, self).get_params(deep=False)
        if not deep:
            return out
        estimators = getattr(self, attr)
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in six.iteritems(
                        estimator.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
        return out
```
### 7 - sklearn/compose/_column_transformer.py:

Start line: 1, End line: 30

```python
"""
The :mod:`sklearn.compose._column_transformer` module implements utilities
to work with heterogeneous data and to apply different transformers to
different columns.
"""


import numpy as np
from scipy import sparse

from ..base import clone, TransformerMixin
from ..externals.joblib import Parallel, delayed
from ..externals import six
from ..pipeline import (
    _fit_one_transformer, _fit_transform_one, _transform_one, _name_estimators)
from ..preprocessing import FunctionTransformer
from ..utils import Bunch
from ..utils.metaestimators import _BaseComposition
from ..utils.validation import check_is_fitted


__all__ = ['ColumnTransformer', 'make_column_transformer']


_ERR_MSG_1DCOLUMN = ("1D data passed to a transformer that expects 2D data. "
                     "Try to specify the column selection as a list of one "
                     "item instead of a scalar.")
```
### 8 - sklearn/ensemble/base.py:

Start line: 100, End line: 117

```python
class BaseEnsemble(six.with_metaclass(ABCMeta, BaseEstimator,
                                      MetaEstimatorMixin)):

    def _validate_estimator(self, default=None):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError("n_estimators must be an integer, "
                             "got {0}.".format(type(self.n_estimators)))

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))

        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")
```
### 9 - sklearn/compose/_column_transformer.py:

Start line: 33, End line: 131

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):
    """Applies transformers to columns of an array or pandas DataFrame.

    EXPERIMENTAL: some behaviors may change between releases without
    deprecation.

    This estimator allows different columns or column subsets of the input
    to be transformed separately and the results combined into a single
    feature space.
    This is useful for heterogeneous or columnar data, to combine several
    feature extraction mechanisms or transformations into a single transformer.

    Read more in the :ref:`User Guide <column_transformer>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, column(s)) tuples specifying the
        transformer objects to be applied to subsets of the data.

        name : string
            Like in Pipeline and FeatureUnion, this allows the transformer and
            its parameters to be set using ``set_params`` and searched in grid
            search.
        transformer : estimator or {'passthrough', 'drop'}
            Estimator must support `fit` and `transform`. Special-cased
            strings 'drop' and 'passthrough' are accepted as well, to
            indicate to drop the columns or to pass them through untransformed,
            respectively.
        column(s) : string or int, array-like of string or int, slice or \
    ean mask array
            Indexes the data on its second axis. Integers are interpreted as
            positional columns, while strings can reference DataFrame columns
            by name.  A scalar string or int should be used where
            ``transformer`` expects X to be a 1d array-like (vector),
            otherwise a 2d array will be passed to the transformer.

    remainder : {'passthrough', 'drop'}, default 'passthrough'
        By default, all remaining columns that were not specified in
        `transformers` will be automatically passed through (default of
        ``'passthrough'``). This subset of columns is concatenated with the
        output of the transformers.
        By using ``remainder='drop'``, only the specified columns in
        `transformers` are transformed and combined in the output, and the
        non-specified columns are dropped.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    Attributes
    ----------
    transformers_ : list
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column).

    named_transformers_ : Bunch object, a dictionary with attribute access
        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the `transformers` list.
    Columns of the original feature matrix that are not specified are
    dropped from the resulting transformed feature matrix, unless specified
    in the `passthrough` keyword. Those columns specified with `passthrough`
    are added at the right to the output of the transformers.

    See also
    --------
    sklearn.compose.make_column_transformer : convenience function for
        combining the outputs of multiple transformer objects applied to
        column subsets of the original feature space.

    Examples
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import Normalizer
    >>> ct = ColumnTransformer(
    ...     [("norm1", Normalizer(norm='l1'), [0, 1]),
    ...      ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    >>> X = np.array([[0., 1., 2., 2.],
    ...               [1., 1., 0., 1.]])
    >>> # Normalizer scales each row of X to unit norm. A separate scaling
    >>> # is applied for the two first and two last elements of each
    >>> # row independently.
    >>> ct.fit_transform(X)    # doctest: +NORMALIZE_WHITESPACE
    array([[0. , 1. , 0.5, 0.5],
           [0.5, 0.5, 0. , 1. ]])

    """
```
### 10 - sklearn/utils/estimator_checks.py:

Start line: 1842, End line: 1877

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
### 12 - sklearn/compose/_column_transformer.py:

Start line: 228, End line: 259

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _validate_remainder(self, X):
        """Generate list of passthrough columns for 'remainder' case."""
        if self.remainder not in ('drop', 'passthrough'):
            raise ValueError(
                "The remainder keyword needs to be one of 'drop' or "
                "'passthrough'. {0:r} was passed instead")

        n_columns = X.shape[1]

        if self.remainder == 'passthrough':
            cols = []
            for _, _, columns in self.transformers:
                cols.extend(_get_column_indices(X, columns))
            self._passthrough = sorted(list(set(range(n_columns)) - set(cols)))
            if not self._passthrough:
                # empty list -> no need to select passthrough columns
                self._passthrough = None
        else:
            self._passthrough = None

    @property
    def named_transformers_(self):
        """Access the fitted transformer by name.

        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.

        """
        # Use Bunch object to improve autocomplete
        return Bunch(**dict([(name, trans) for name, trans, _
                             in self.transformers_]))
```
### 18 - sklearn/compose/_column_transformer.py:

Start line: 603, End line: 663

```python
def make_column_transformer(*transformers, **kwargs):
    """Construct a ColumnTransformer from the given transformers.

    This is a shorthand for the ColumnTransformer constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting.

    Parameters
    ----------
    *transformers : tuples of column selections and transformers

    remainder : {'passthrough', 'drop'}, default 'passthrough'
        By default, all remaining columns that were not specified in
        `transformers` will be automatically passed through (default of
        ``'passthrough'``). This subset of columns is concatenated with the
        output of the transformers.
        By using ``remainder='drop'``, only the specified columns in
        `transformers` are transformed and combined in the output, and the
        non-specified columns are dropped.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    Returns
    -------
    ct : ColumnTransformer

    See also
    --------
    sklearn.compose.ColumnTransformer : Class that allows combining the
        outputs of multiple transformer objects used on column subsets
        of the data into a single feature space.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler, CategoricalEncoder
    >>> from sklearn.compose import make_column_transformer
    >>> make_column_transformer(
    ...     (['numerical_column'], StandardScaler()),
    ...     (['categorical_column'], CategoricalEncoder()))
    ...     # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ColumnTransformer(n_jobs=1, remainder='passthrough',
             transformer_weights=None,
             transformers=[('standardscaler',
                            StandardScaler(...),
                            ['numerical_column']),
                           ('categoricalencoder',
                            CategoricalEncoder(...),
                            ['categorical_column'])])

    """
    n_jobs = kwargs.pop('n_jobs', 1)
    remainder = kwargs.pop('remainder', 'passthrough')
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs,
                             remainder=remainder)
```
### 19 - sklearn/compose/_column_transformer.py:

Start line: 346, End line: 370

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : array-like or DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : ColumnTransformer
            This estimator

        """
        self._validate_transformers()
        self._validate_remainder(X)

        transformers = self._fit_transform(X, y, _fit_one_transformer)

        self._update_fitted_transformers(transformers)
        return self
```
### 24 - sklearn/compose/_column_transformer.py:

Start line: 314, End line: 324

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _validate_output(self, result):
        """
        Ensure that the output of each transformer is 2D. Otherwise
        hstack can raise an error or produce incorrect results.
        """
        names = [name for name, _, _, _ in self._iter(replace_strings=True)]
        for Xs, name in zip(result, names):
            if not getattr(Xs, 'ndim', 0) == 2:
                raise ValueError(
                    "The output of the '{0}' transformer should be 2D (scipy "
                    "matrix, array, or pandas DataFrame).".format(name))
```
### 25 - sklearn/compose/_column_transformer.py:

Start line: 326, End line: 344

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _fit_transform(self, X, y, func, fitted=False):
        """
        Private function to fit and/or transform on demand.

        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.
        """
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(clone(trans) if not fitted else trans,
                              X_sel, y, weight)
                for name, trans, X_sel, weight in self._iter(
                    X=X, fitted=fitted, replace_strings=True))
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN)
            else:
                raise
```
### 37 - sklearn/compose/_column_transformer.py:

Start line: 184, End line: 209

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def _iter(self, X=None, fitted=False, replace_strings=False):
        """Generate (name, trans, column, weight) tuples
        """
        if fitted:
            transformers = self.transformers_
        else:
            transformers = self.transformers
        get_weight = (self.transformer_weights or {}).get

        for name, trans, column in transformers:
            if X is None:
                sub = X
            else:
                sub = _get_column(X, column)

            if replace_strings:
                # replace 'passthrough' with identity transformer and
                # skip in case of 'drop'
                if trans == 'passthrough':
                    trans = FunctionTransformer(
                        validate=False, accept_sparse=True,
                        check_inverse=False)
                elif trans == 'drop':
                    continue

            yield (name, trans, sub, get_weight(name))
```
### 40 - sklearn/compose/_column_transformer.py:

Start line: 372, End line: 417

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def fit_transform(self, X, y=None):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : array-like or DataFrame of shape [n_samples, n_features]
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        self._validate_transformers()
        self._validate_remainder(X)

        result = self._fit_transform(X, y, _fit_transform_one)

        if not result:
            # All transformers are None
            if self._passthrough is None:
                return np.zeros((X.shape[0], 0))
            else:
                return _get_column(X, self._passthrough)

        Xs, transformers = zip(*result)

        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)

        if self._passthrough is not None:
            Xs = list(Xs) + [_get_column(X, self._passthrough)]

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs
```
### 61 - sklearn/compose/_column_transformer.py:

Start line: 261, End line: 290

```python
class ColumnTransformer(_BaseComposition, TransformerMixin):

    def get_feature_names(self):
        """Get feature names from all transformers.

        Returns
        -------
        feature_names : list of strings
            Names of the features produced by transform.
        """
        check_is_fitted(self, 'transformers_')
        if self._passthrough is not None:
            raise NotImplementedError(
                "get_feature_names is not yet supported when having columns"
                "that are passed through (you specify remainder='drop' to not "
                "pass through the unspecified columns).")

        feature_names = []
        for name, trans, _, _ in self._iter(fitted=True):
            if trans == 'drop':
                continue
            elif trans == 'passthrough':
                raise NotImplementedError(
                    "get_feature_names is not yet supported when using "
                    "a 'passthrough' transformer.")
            elif not hasattr(trans, 'get_feature_names'):
                raise AttributeError("Transformer %s (type %s) does not "
                                     "provide get_feature_names."
                                     % (str(name), type(trans).__name__))
            feature_names.extend([name + "__" + f for f in
                                  trans.get_feature_names()])
        return feature_names
```
### 70 - sklearn/compose/_column_transformer.py:

Start line: 541, End line: 600

```python
def _get_column_indices(X, key):
    """
    Get feature column indices for input data X and key.

    For accepted values of `key`, see the docstring of _get_column

    """
    n_columns = X.shape[1]

    if _check_key_type(key, int):
        if isinstance(key, int):
            return [key]
        elif isinstance(key, slice):
            return list(range(n_columns)[key])
        else:
            return list(key)

    elif _check_key_type(key, six.string_types):
        try:
            all_columns = list(X.columns)
        except AttributeError:
            raise ValueError("Specifying the columns using strings is only "
                             "supported for pandas DataFrames")
        if isinstance(key, six.string_types):
            columns = [key]
        elif isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is not None:
                start = all_columns.index(start)
            if stop is not None:
                # pandas indexing with strings is endpoint included
                stop = all_columns.index(stop) + 1
            else:
                stop = n_columns + 1
            return list(range(n_columns)[slice(start, stop)])
        else:
            columns = list(key)

        return [all_columns.index(col) for col in columns]

    elif hasattr(key, 'dtype') and np.issubdtype(key.dtype, np.bool_):
        # boolean mask
        return list(np.arange(n_columns)[key])
    else:
        raise ValueError("No valid specification of the columns. Only a "
                         "scalar, list or slice of all integers or all "
                         "strings, or boolean mask is allowed")


def _get_transformer_list(estimators):
    """
    Construct (name, trans, column) tuples from list

    """
    transformers = [trans[1] for trans in estimators]
    columns = [trans[0] for trans in estimators]
    names = [trans[0] for trans in _name_estimators(transformers)]

    transformer_list = list(zip(names, transformers, columns))
    return transformer_list
```

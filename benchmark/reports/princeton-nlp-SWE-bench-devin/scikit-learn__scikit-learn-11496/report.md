# scikit-learn__scikit-learn-11496

| **scikit-learn/scikit-learn** | `cb0140017740d985960911c4f34820beea915846` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 5975 |
| **Any found context length** | 765 |
| **Avg pos** | 31.0 |
| **Min pos** | 2 |
| **Max pos** | 13 |
| **Top file pos** | 1 |
| **Missing snippets** | 5 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/impute.py | 136 | 136 | 13 | 1 | 5975
| sklearn/impute.py | 230 | 233 | 5 | 1 | 2342
| sklearn/impute.py | 244 | 317 | 4 | 1 | 2019
| sklearn/impute.py | 367 | 367 | 7 | 1 | 3453
| sklearn/impute.py | 405 | 415 | 2 | 1 | 765


## Problem Statement

```
BUG: SimpleImputer gives wrong result on sparse matrix with explicit zeros
The current implementation of the `SimpleImputer` can't deal with zeros stored explicitly in sparse matrix.
Even when stored explicitly, we'd expect that all zeros are treating equally, right ?
See for example the code below:
\`\`\`python
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
\`\`\`
Whereas the expected result would be
\`\`\`python
>>> array([[1. , 1. , 1. ],
           [1. , 1. , 1. ],
           [1. , 1. , 1. ]])
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/impute.py** | 158 | 194| 331 | 331 | 7910 | 
| **-> 2 <-** | **1 sklearn/impute.py** | 369 | 423| 434 | 765 | 7910 | 
| 3 | 2 sklearn/preprocessing/imputation.py | 298 | 374| 650 | 1415 | 10852 | 
| **-> 4 <-** | **2 sklearn/impute.py** | 242 | 317| 604 | 2019 | 10852 | 
| **-> 5 <-** | **2 sklearn/impute.py** | 196 | 240| 323 | 2342 | 10852 | 
| 6 | 2 sklearn/preprocessing/imputation.py | 173 | 251| 675 | 3017 | 10852 | 
| **-> 7 <-** | **2 sklearn/impute.py** | 319 | 367| 436 | 3453 | 10852 | 
| 8 | 2 sklearn/preprocessing/imputation.py | 129 | 171| 382 | 3835 | 10852 | 
| 9 | 2 sklearn/preprocessing/imputation.py | 253 | 296| 438 | 4273 | 10852 | 
| 10 | **2 sklearn/impute.py** | 762 | 806| 386 | 4659 | 10852 | 
| 11 | **2 sklearn/impute.py** | 891 | 968| 623 | 5282 | 10852 | 
| 12 | 2 sklearn/preprocessing/imputation.py | 4 | 33| 160 | 5442 | 10852 | 
| **-> 13 <-** | **2 sklearn/impute.py** | 95 | 156| 533 | 5975 | 10852 | 
| 14 | **2 sklearn/impute.py** | 43 | 58| 142 | 6117 | 10852 | 
| 15 | 2 sklearn/preprocessing/imputation.py | 64 | 127| 564 | 6681 | 10852 | 
| 16 | **2 sklearn/impute.py** | 558 | 641| 660 | 7341 | 10852 | 
| 17 | **2 sklearn/impute.py** | 426 | 531| 1007 | 8348 | 10852 | 
| 18 | **2 sklearn/impute.py** | 1 | 40| 206 | 8554 | 10852 | 
| 19 | **2 sklearn/impute.py** | 808 | 889| 736 | 9290 | 10852 | 
| 20 | 3 examples/plot_missing_values.py | 84 | 129| 391 | 9681 | 11980 | 
| 21 | 3 examples/plot_missing_values.py | 1 | 28| 209 | 9890 | 11980 | 
| 22 | 4 sklearn/preprocessing/data.py | 69 | 84| 129 | 10019 | 33837 | 
| 23 | **4 sklearn/impute.py** | 683 | 725| 428 | 10447 | 33837 | 
| 24 | **4 sklearn/impute.py** | 727 | 760| 350 | 10797 | 33837 | 
| 25 | 5 sklearn/utils/sparsefuncs.py | 6 | 25| 173 | 10970 | 37758 | 
| 26 | **5 sklearn/impute.py** | 533 | 556| 183 | 11153 | 37758 | 
| 27 | 6 sklearn/cluster/birch.py | 22 | 37| 142 | 11295 | 43049 | 
| 28 | 7 sklearn/utils/estimator_checks.py | 1083 | 1151| 601 | 11896 | 63034 | 
| 29 | 7 examples/plot_missing_values.py | 31 | 81| 527 | 12423 | 63034 | 
| 30 | **7 sklearn/impute.py** | 61 | 92| 296 | 12719 | 63034 | 
| 31 | 7 sklearn/preprocessing/data.py | 145 | 198| 596 | 13315 | 63034 | 
| 32 | 7 sklearn/utils/estimator_checks.py | 468 | 514| 469 | 13784 | 63034 | 
| 33 | 8 sklearn/utils/fixes.py | 99 | 203| 776 | 14560 | 65445 | 
| 34 | 8 sklearn/utils/sparsefuncs.py | 435 | 480| 382 | 14942 | 65445 | 
| 35 | 9 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 15453 | 65978 | 
| 36 | 10 sklearn/utils/validation.py | 256 | 351| 799 | 16252 | 74055 | 
| 37 | 10 sklearn/preprocessing/data.py | 2017 | 2060| 428 | 16680 | 74055 | 
| 38 | 11 sklearn/linear_model/base.py | 301 | 328| 257 | 16937 | 78279 | 
| 39 | 11 sklearn/preprocessing/data.py | 1638 | 1679| 386 | 17323 | 78279 | 
| 40 | 11 sklearn/utils/validation.py | 734 | 754| 261 | 17584 | 78279 | 
| 41 | 11 sklearn/utils/estimator_checks.py | 1945 | 1965| 233 | 17817 | 78279 | 
| 42 | 11 sklearn/utils/sparsefuncs.py | 371 | 397| 238 | 18055 | 78279 | 
| 43 | 12 sklearn/utils/__init__.py | 347 | 372| 167 | 18222 | 82670 | 
| 44 | 12 sklearn/preprocessing/data.py | 654 | 718| 568 | 18790 | 82670 | 
| 45 | 12 sklearn/utils/estimator_checks.py | 434 | 465| 302 | 19092 | 82670 | 
| 46 | 12 sklearn/utils/__init__.py | 103 | 120| 181 | 19273 | 82670 | 
| 47 | 13 examples/linear_model/plot_lasso_dense_vs_sparse_data.py | 1 | 67| 511 | 19784 | 83181 | 
| 48 | 14 sklearn/utils/linear_assignment_.py | 220 | 267| 488 | 20272 | 85610 | 
| 49 | 14 sklearn/utils/sparsefuncs.py | 483 | 508| 228 | 20500 | 85610 | 
| 50 | 15 sklearn/preprocessing/_encoders.py | 76 | 108| 297 | 20797 | 92049 | 
| 51 | 16 sklearn/decomposition/dict_learning.py | 275 | 331| 493 | 21290 | 103347 | 
| 52 | 16 sklearn/utils/sparsefuncs.py | 348 | 368| 228 | 21518 | 103347 | 
| 53 | 17 benchmarks/bench_sparsify.py | 1 | 82| 754 | 22272 | 104254 | 
| 54 | 17 sklearn/preprocessing/data.py | 1614 | 1635| 229 | 22501 | 104254 | 
| 55 | 17 sklearn/preprocessing/imputation.py | 36 | 61| 240 | 22741 | 104254 | 
| 56 | 18 sklearn/ensemble/gradient_boosting.py | 152 | 170| 142 | 22883 | 122532 | 
| 57 | 19 sklearn/manifold/isomap.py | 102 | 124| 215 | 23098 | 124221 | 
| 58 | 19 sklearn/preprocessing/data.py | 1520 | 1550| 278 | 23376 | 124221 | 
| 59 | 19 sklearn/utils/validation.py | 603 | 623| 216 | 23592 | 124221 | 
| 60 | 19 sklearn/utils/linear_assignment_.py | 183 | 217| 360 | 23952 | 124221 | 
| 61 | 19 sklearn/utils/estimator_checks.py | 557 | 592| 328 | 24280 | 124221 | 
| 62 | 19 sklearn/utils/validation.py | 40 | 68| 267 | 24547 | 124221 | 
| 63 | 19 sklearn/utils/validation.py | 504 | 572| 799 | 25346 | 124221 | 
| 64 | 19 sklearn/preprocessing/_encoders.py | 485 | 523| 410 | 25756 | 124221 | 
| 65 | 19 sklearn/preprocessing/data.py | 1682 | 1737| 445 | 26201 | 124221 | 
| 66 | 19 sklearn/preprocessing/data.py | 1739 | 1760| 210 | 26411 | 124221 | 
| 67 | 19 sklearn/linear_model/base.py | 70 | 136| 629 | 27040 | 124221 | 
| 68 | 19 benchmarks/bench_sparsify.py | 83 | 106| 153 | 27193 | 124221 | 
| 69 | 19 sklearn/preprocessing/_encoders.py | 575 | 640| 515 | 27708 | 124221 | 
| 70 | 19 sklearn/decomposition/dict_learning.py | 113 | 181| 692 | 28400 | 124221 | 
| 71 | 20 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 1 | 80| 698 | 29098 | 124973 | 
| 72 | 21 sklearn/kernel_approximation.py | 338 | 364| 269 | 29367 | 129097 | 
| 73 | 22 sklearn/preprocessing/_discretization.py | 229 | 267| 373 | 29740 | 131751 | 
| 74 | 22 sklearn/decomposition/dict_learning.py | 866 | 902| 267 | 30007 | 131751 | 
| 75 | 22 sklearn/utils/validation.py | 448 | 502| 549 | 30556 | 131751 | 
| 76 | 23 sklearn/feature_extraction/dict_vectorizer.py | 137 | 211| 606 | 31162 | 134469 | 
| 77 | 24 sklearn/decomposition/nmf.py | 1 | 66| 366 | 31528 | 146209 | 
| 78 | 25 sklearn/random_projection.py | 264 | 292| 292 | 31820 | 151182 | 
| 79 | 26 sklearn/utils/testing.py | 232 | 246| 131 | 31951 | 158063 | 
| 80 | 27 sklearn/externals/_pilutil.py | 69 | 140| 678 | 32629 | 162668 | 
| 81 | 27 sklearn/preprocessing/data.py | 953 | 1009| 463 | 33092 | 162668 | 
| 82 | 28 sklearn/ensemble/iforest.py | 238 | 258| 203 | 33295 | 166032 | 
| 83 | 28 sklearn/preprocessing/_encoders.py | 405 | 458| 562 | 33857 | 166032 | 
| 84 | 28 sklearn/utils/estimator_checks.py | 151 | 171| 203 | 34060 | 166032 | 
| 85 | 28 sklearn/utils/sparsefuncs.py | 101 | 161| 469 | 34529 | 166032 | 
| 86 | 29 sklearn/naive_bayes.py | 711 | 732| 227 | 34756 | 174441 | 
| 87 | 29 sklearn/utils/__init__.py | 78 | 100| 120 | 34876 | 174441 | 
| 88 | 29 sklearn/random_projection.py | 330 | 394| 488 | 35364 | 174441 | 
| 89 | 29 sklearn/utils/estimator_checks.py | 1985 | 2005| 200 | 35564 | 174441 | 
| 90 | 29 sklearn/utils/estimator_checks.py | 1968 | 1982| 211 | 35775 | 174441 | 
| 91 | 29 sklearn/preprocessing/data.py | 1833 | 1889| 546 | 36321 | 174441 | 
| 92 | **29 sklearn/impute.py** | 643 | 681| 330 | 36651 | 174441 | 
| 93 | 30 sklearn/feature_extraction/text.py | 848 | 908| 467 | 37118 | 186689 | 
| 94 | 31 sklearn/utils/extmath.py | 622 | 647| 194 | 37312 | 193019 | 
| 95 | 31 sklearn/utils/sparsefuncs.py | 511 | 541| 205 | 37517 | 193019 | 
| 96 | 31 sklearn/utils/extmath.py | 114 | 139| 197 | 37714 | 193019 | 
| 97 | 32 sklearn/feature_extraction/hashing.py | 134 | 172| 347 | 38061 | 194549 | 


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


## Code snippets

### 1 - sklearn/impute.py:

Start line: 158, End line: 194

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def _validate_input(self, X):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.strategy in ("most_frequent", "constant"):
            dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        try:
            X = check_array(X, accept_sparse='csc', dtype=dtype,
                            force_all_finite=force_all_finite, copy=self.copy)
        except ValueError as ve:
            if "could not convert" in str(ve):
                raise ValueError("Cannot use {0} strategy with non-numeric "
                                 "data. Received datatype :{1}."
                                 "".format(self.strategy, X.dtype.kind))
            else:
                raise ve

        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("SimpleImputer does not support data with dtype "
                             "{0}. Please provide either a numeric array (with"
                             " a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        return X
```
### 2 - sklearn/impute.py:

Start line: 369, End line: 423

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        check_is_fitted(self, 'statistics_')

        X = self._validate_input(X)

        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[0]))

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
        else:
            # same as np.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, np.nan)
            valid_mask = np.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = np.flatnonzero(valid_mask)

            if invalid_mask.any():
                missing = np.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn("Deleting features without "
                                  "observed values: %s" % missing)
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sparse.issparse(X) and self.missing_values != 0:
            mask = _get_mask(X.data, self.missing_values)
            indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                np.diff(X.indptr))[mask]

            X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                            copy=False)
        else:
            if sparse.issparse(X):
                X = X.toarray()

            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask.transpose())[::-1]

            X[coordinates] = values

        return X
```
### 3 - sklearn/preprocessing/imputation.py:

Start line: 298, End line: 374

```python
@deprecated("Imputer was deprecated in version 0.20 and will be "
            "removed in 0.22. Import impute.SimpleImputer from "
            "sklearn instead.")
class Imputer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The input data to complete.
        """
        if self.axis == 0:
            check_is_fitted(self, 'statistics_')
            X = check_array(X, accept_sparse='csc', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)
            statistics = self.statistics_
            if X.shape[1] != statistics.shape[0]:
                raise ValueError("X has %d features per sample, expected %d"
                                 % (X.shape[1], self.statistics_.shape[0]))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data need to be recomputed
        # when the imputation is done per sample
        else:
            X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES,
                            force_all_finite=False, copy=self.copy)

            if sparse.issparse(X):
                statistics = self._sparse_fit(X,
                                              self.strategy,
                                              self.missing_values,
                                              self.axis)

            else:
                statistics = self._dense_fit(X,
                                             self.strategy,
                                             self.missing_values,
                                             self.axis)

        # Delete the invalid rows/columns
        invalid_mask = np.isnan(statistics)
        valid_mask = np.logical_not(invalid_mask)
        valid_statistics = statistics[valid_mask]
        valid_statistics_indexes = np.where(valid_mask)[0]
        missing = np.arange(X.shape[not self.axis])[invalid_mask]

        if self.axis == 0 and invalid_mask.any():
            if self.verbose:
                warnings.warn("Deleting features without "
                              "observed values: %s" % missing)
            X = X[:, valid_statistics_indexes]
        elif self.axis == 1 and invalid_mask.any():
            raise ValueError("Some rows only contain "
                             "missing values: %s" % missing)

        # Do actual imputation
        if sparse.issparse(X) and self.missing_values != 0:
            mask = _get_mask(X.data, self.missing_values)
            indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                np.diff(X.indptr))[mask]

            X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                            copy=False)
        else:
            if sparse.issparse(X):
                X = X.toarray()

            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=self.axis)
            values = np.repeat(valid_statistics, n_missing)

            if self.axis == 0:
                coordinates = np.where(mask.transpose())[::-1]
            else:
                coordinates = mask

            X[coordinates] = values

        return X
```
### 4 - sklearn/impute.py:

Start line: 242, End line: 317

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        # Count the zeros
        if missing_values == 0:
            n_zeros_axis = np.zeros(X.shape[1], dtype=int)
        else:
            n_zeros_axis = X.shape[0] - np.diff(X.indptr)

        # Mean
        if strategy == "mean":
            if missing_values != 0:
                n_non_missing = n_zeros_axis

                # Mask the missing elements
                mask_missing_values = _get_mask(X.data, missing_values)
                mask_valids = np.logical_not(mask_missing_values)

                # Sum only the valid elements
                new_data = X.data.copy()
                new_data[mask_missing_values] = 0
                X = sparse.csc_matrix((new_data, X.indices, X.indptr),
                                      copy=False)
                sums = X.sum(axis=0)

                # Count the elements != 0
                mask_non_zeros = sparse.csc_matrix(
                    (mask_valids.astype(np.float64),
                     X.indices,
                     X.indptr), copy=False)
                s = mask_non_zeros.sum(axis=0)
                n_non_missing = np.add(n_non_missing, s)

            else:
                sums = X.sum(axis=0)
                n_non_missing = np.diff(X.indptr)

            # Ignore the error, columns with a np.nan statistics_
            # are not an error at this point. These columns will
            # be removed in transform
            with np.errstate(all="ignore"):
                return np.ravel(sums) / np.ravel(n_non_missing)

        # Median + Most frequent + Constant
        else:
            # Remove the missing values, for each column
            columns_all = np.hsplit(X.data, X.indptr[1:-1])
            mask_missing_values = _get_mask(X.data, missing_values)
            mask_valids = np.hsplit(np.logical_not(mask_missing_values),
                                    X.indptr[1:-1])

            # astype necessary for bug in numpy.hsplit before v1.9
            columns = [col[mask.astype(bool, copy=False)]
                       for col, mask in zip(columns_all, mask_valids)]

            # Median
            if strategy == "median":
                median = np.empty(len(columns))
                for i, column in enumerate(columns):
                    median[i] = _get_median(column, n_zeros_axis[i])

                return median

            # Most frequent
            elif strategy == "most_frequent":
                most_frequent = np.empty(len(columns))

                for i, column in enumerate(columns):
                    most_frequent[i] = _most_frequent(column,
                                                      0,
                                                      n_zeros_axis[i])

                return most_frequent

            # Constant
            elif strategy == "constant":
                return np.full(X.shape[1], fill_value)
```
### 5 - sklearn/impute.py:

Start line: 196, End line: 240

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : SimpleImputer
        """
        X = self._validate_input(X)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # fill_value should be numerical in case of numerical input
        if (self.strategy == "constant" and
                X.dtype.kind in ("i", "u", "f") and
                not isinstance(fill_value, numbers.Real)):
            raise ValueError("'fill_value'={0} is invalid. Expected a "
                             "numerical value when imputing numerical "
                             "data".format(fill_value))

        if sparse.issparse(X):
            self.statistics_ = self._sparse_fit(X,
                                                self.strategy,
                                                self.missing_values,
                                                fill_value)
        else:
            self.statistics_ = self._dense_fit(X,
                                               self.strategy,
                                               self.missing_values,
                                               fill_value)

        return self
```
### 6 - sklearn/preprocessing/imputation.py:

Start line: 173, End line: 251

```python
@deprecated("Imputer was deprecated in version 0.20 and will be "
            "removed in 0.22. Import impute.SimpleImputer from "
            "sklearn instead.")
class Imputer(BaseEstimator, TransformerMixin):

    def _sparse_fit(self, X, strategy, missing_values, axis):
        """Fit the transformer on sparse data."""
        # Imputation is done "by column", so if we want to do it
        # by row we only need to convert the matrix to csr format.
        if axis == 1:
            X = X.tocsr()
        else:
            X = X.tocsc()

        # Count the zeros
        if missing_values == 0:
            n_zeros_axis = np.zeros(X.shape[not axis], dtype=int)
        else:
            n_zeros_axis = X.shape[axis] - np.diff(X.indptr)

        # Mean
        if strategy == "mean":
            if missing_values != 0:
                n_non_missing = n_zeros_axis

                # Mask the missing elements
                mask_missing_values = _get_mask(X.data, missing_values)
                mask_valids = np.logical_not(mask_missing_values)

                # Sum only the valid elements
                new_data = X.data.copy()
                new_data[mask_missing_values] = 0
                X = sparse.csc_matrix((new_data, X.indices, X.indptr),
                                      copy=False)
                sums = X.sum(axis=0)

                # Count the elements != 0
                mask_non_zeros = sparse.csc_matrix(
                    (mask_valids.astype(np.float64),
                     X.indices,
                     X.indptr), copy=False)
                s = mask_non_zeros.sum(axis=0)
                n_non_missing = np.add(n_non_missing, s)

            else:
                sums = X.sum(axis=axis)
                n_non_missing = np.diff(X.indptr)

            # Ignore the error, columns with a np.nan statistics_
            # are not an error at this point. These columns will
            # be removed in transform
            with np.errstate(all="ignore"):
                return np.ravel(sums) / np.ravel(n_non_missing)

        # Median + Most frequent
        else:
            # Remove the missing values, for each column
            columns_all = np.hsplit(X.data, X.indptr[1:-1])
            mask_missing_values = _get_mask(X.data, missing_values)
            mask_valids = np.hsplit(np.logical_not(mask_missing_values),
                                    X.indptr[1:-1])

            # astype necessary for bug in numpy.hsplit before v1.9
            columns = [col[mask.astype(bool, copy=False)]
                       for col, mask in zip(columns_all, mask_valids)]

            # Median
            if strategy == "median":
                median = np.empty(len(columns))
                for i, column in enumerate(columns):
                    median[i] = _get_median(column, n_zeros_axis[i])

                return median

            # Most frequent
            elif strategy == "most_frequent":
                most_frequent = np.empty(len(columns))

                for i, column in enumerate(columns):
                    most_frequent[i] = _most_frequent(column,
                                                      0,
                                                      n_zeros_axis[i])

                return most_frequent
```
### 7 - sklearn/impute.py:

Start line: 319, End line: 367

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=0)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # scipy.stats.mstats.mode cannot be used because it will no work
            # properly if the first element is masked and if its frequency
            # is equal to the frequency of the most frequent valid element
            # See https://github.com/scipy/scipy/issues/2636

            # To be able access the elements by columns
            X = X.transpose()
            mask = mask.transpose()

            if X.dtype.kind == "O":
                most_frequent = np.empty(X.shape[0], dtype=object)
            else:
                most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(np.bool)
                row = row[row_mask]
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent

        # Constant
        elif strategy == "constant":
            return np.full(X.shape[1], fill_value, dtype=X.dtype)
```
### 8 - sklearn/preprocessing/imputation.py:

Start line: 129, End line: 171

```python
@deprecated("Imputer was deprecated in version 0.20 and will be "
            "removed in 0.22. Import impute.SimpleImputer from "
            "sklearn instead.")
class Imputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : Imputer
        """
        # Check parameters
        allowed_strategies = ["mean", "median", "most_frequent"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.axis not in [0, 1]:
            raise ValueError("Can only impute missing values on axis 0 and 1, "
                             " got axis={0}".format(self.axis))

        # Since two different arrays can be provided in fit(X) and
        # transform(X), the imputation data will be computed in transform()
        # when the imputation is done per sample (i.e., when axis=1).
        if self.axis == 0:
            X = check_array(X, accept_sparse='csc', dtype=np.float64,
                            force_all_finite=False)

            if sparse.issparse(X):
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    self.axis)
            else:
                self.statistics_ = self._dense_fit(X,
                                                   self.strategy,
                                                   self.missing_values,
                                                   self.axis)

        return self
```
### 9 - sklearn/preprocessing/imputation.py:

Start line: 253, End line: 296

```python
@deprecated("Imputer was deprecated in version 0.20 and will be "
            "removed in 0.22. Import impute.SimpleImputer from "
            "sklearn instead.")
class Imputer(BaseEstimator, TransformerMixin):

    def _dense_fit(self, X, strategy, missing_values, axis):
        """Fit the transformer on dense data."""
        X = check_array(X, force_all_finite=False)
        mask = _get_mask(X, missing_values)
        masked_X = ma.masked_array(X, mask=mask)

        # Mean
        if strategy == "mean":
            mean_masked = np.ma.mean(masked_X, axis=axis)
            # Avoid the warning "Warning: converting a masked element to nan."
            mean = np.ma.getdata(mean_masked)
            mean[np.ma.getmask(mean_masked)] = np.nan

            return mean

        # Median
        elif strategy == "median":
            median_masked = np.ma.median(masked_X, axis=axis)
            # Avoid the warning "Warning: converting a masked element to nan."
            median = np.ma.getdata(median_masked)
            median[np.ma.getmaskarray(median_masked)] = np.nan

            return median

        # Most frequent
        elif strategy == "most_frequent":
            # scipy.stats.mstats.mode cannot be used because it will no work
            # properly if the first element is masked and if its frequency
            # is equal to the frequency of the most frequent valid element
            # See https://github.com/scipy/scipy/issues/2636

            # To be able access the elements by columns
            if axis == 0:
                X = X.transpose()
                mask = mask.transpose()

            most_frequent = np.empty(X.shape[0])

            for i, (row, row_mask) in enumerate(zip(X[:], mask[:])):
                row_mask = np.logical_not(row_mask).astype(np.bool)
                row = row[row_mask]
                most_frequent[i] = _most_frequent(row, np.nan, 0)

            return most_frequent
```
### 10 - sklearn/impute.py:

Start line: 762, End line: 806

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def _initial_imputation(self, X):
        """Perform initial imputation for input X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        mask_missing_values : ndarray, shape (n_samples, n_features)
            Input data's missing indicator matrix, where "n_samples" is the
            number of samples and "n_features" is the number of features.
        """
        if is_scalar_nan(self.missing_values):
            force_all_finite = "allow-nan"
        else:
            force_all_finite = True

        X = check_array(X, dtype=FLOAT_DTYPES, order="F",
                        force_all_finite=force_all_finite)

        mask_missing_values = _get_mask(X, self.missing_values)
        if self.initial_imputer_ is None:
            self.initial_imputer_ = SimpleImputer(
                                            missing_values=self.missing_values,
                                            strategy=self.initial_strategy)
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)

        valid_mask = np.flatnonzero(np.logical_not(
            np.isnan(self.initial_imputer_.statistics_)))
        Xt = X[:, valid_mask]
        mask_missing_values = mask_missing_values[:, valid_mask]

        return Xt, X_filled, mask_missing_values
```
### 11 - sklearn/impute.py:

Start line: 891, End line: 968

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Imputes all missing values in X.

        Note that this is stochastic, and that if random_state is not fixed,
        repeated calls, or permuted input, will yield different results.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        check_is_fitted(self, 'initial_imputer_')

        X, X_filled, mask_missing_values = self._initial_imputation(X)

        # edge case: in case the user specifies 0 for n_imputations,
        # then there is no need to do burn in and the result should be
        # just the initial imputation (before clipping)
        if self.n_imputations < 1:
            return X_filled

        X_filled = np.clip(X_filled, self._min_value, self._max_value)

        n_rounds = self.n_burn_in + self.n_imputations
        n_imputations = len(self.imputation_sequence_)
        imputations_per_round = n_imputations // n_rounds
        i_rnd = 0
        Xt = np.zeros(X.shape, dtype=X.dtype)
        if self.verbose > 0:
            print("[ChainedImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        for it, predictor_triplet in enumerate(self.imputation_sequence_):
            X_filled, _ = self._impute_one_feature(
                X_filled,
                mask_missing_values,
                predictor_triplet.feat_idx,
                predictor_triplet.neighbor_feat_idx,
                predictor=predictor_triplet.predictor,
                fit_mode=False
            )
            if not (it + 1) % imputations_per_round:
                if i_rnd >= self.n_burn_in:
                    Xt += X_filled
                if self.verbose > 1:
                    print('[ChainedImputer] Ending imputation round '
                          '%d/%d, elapsed time %0.2f'
                          % (i_rnd + 1, n_rounds, time() - start_t))
                i_rnd += 1

        Xt /= self.n_imputations
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt

    def fit(self, X, y=None):
        """Fits the imputer on X and return self.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored

        Returns
        -------
        self : object
            Returns self.
        """
        self.fit_transform(X)
        return self
```
### 13 - sklearn/impute.py:

Start line: 95, End line: 156

```python
class SimpleImputer(BaseEstimator, TransformerMixin):
    """Imputation transformer for completing missing values.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    strategy : string, optional (default="mean")
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        .. versionadded:: 0.20
           strategy="constant" for fixed value imputation.

    fill_value : string or numerical value, optional (default=None)
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If X is encoded as a CSR matrix.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.

    Notes
    -----
    Columns which only contained missing values at `fit` are discarded upon
    `transform` if strategy is not "constant".

    """
    def __init__(self, missing_values=np.nan, strategy="mean",
                 fill_value=None, verbose=0, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
```
### 14 - sklearn/impute.py:

Start line: 43, End line: 58

```python
def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask is np.nan:
        if X.dtype.kind == "f":
            return np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            return np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            return _object_dtype_isnan(X)

    else:
        # X == value_to_mask with object dytpes does not always perform
        # element-wise for old versions of numpy
        return np.equal(X, value_to_mask)
```
### 16 - sklearn/impute.py:

Start line: 558, End line: 641

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def _impute_one_feature(self,
                            X_filled,
                            mask_missing_values,
                            feat_idx,
                            neighbor_feat_idx,
                            predictor=None,
                            fit_mode=True):
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The ``predictor`` must
        support ``return_std=True`` in its ``predict`` method for this function
        to work.

        Parameters
        ----------
        X_filled : ndarray
            Input data with the most recent imputations.

        mask_missing_values : ndarray
            Input data's missing indicator matrix.

        feat_idx : int
            Index of the feature currently being imputed.

        neighbor_feat_idx : ndarray
            Indices of the features to be used in imputing ``feat_idx``.

        predictor : object
            The predictor to use at this step of the round-robin imputation.
            It must support ``return_std`` in its ``predict`` method.
            If None, it will be cloned from self._predictor.

        fit_mode : boolean, default=True
            Whether to fit and predict with the predictor or just predict.

        Returns
        -------
        X_filled : ndarray
            Input data with ``X_filled[missing_row_mask, feat_idx]`` updated.

        predictor : predictor with sklearn API
            The fitted predictor used to impute
            ``X_filled[missing_row_mask, feat_idx]``.
        """

        # if nothing is missing, just return the default
        # (should not happen at fit time because feat_ids would be excluded)
        missing_row_mask = mask_missing_values[:, feat_idx]
        if not np.any(missing_row_mask):
            return X_filled, predictor

        if predictor is None and fit_mode is False:
            raise ValueError("If fit_mode is False, then an already-fitted "
                             "predictor should be passed in.")

        if predictor is None:
            predictor = clone(self._predictor)

        if fit_mode:
            X_train = safe_indexing(X_filled[:, neighbor_feat_idx],
                                    ~missing_row_mask)
            y_train = safe_indexing(X_filled[:, feat_idx],
                                    ~missing_row_mask)
            predictor.fit(X_train, y_train)

        # get posterior samples
        X_test = safe_indexing(X_filled[:, neighbor_feat_idx],
                               missing_row_mask)
        mus, sigmas = predictor.predict(X_test, return_std=True)
        good_sigmas = sigmas > 0
        imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
        imputed_values[~good_sigmas] = mus[~good_sigmas]
        imputed_values[good_sigmas] = self.random_state_.normal(
            loc=mus[good_sigmas], scale=sigmas[good_sigmas])

        # clip the values
        imputed_values = np.clip(imputed_values,
                                 self._min_value,
                                 self._max_value)

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, predictor
```
### 17 - sklearn/impute.py:

Start line: 426, End line: 531

```python
class ChainedImputer(BaseEstimator, TransformerMixin):
    """Chained imputer transformer to impute missing values.

    Basic implementation of chained imputer from MICE (Multivariate
    Imputations by Chained Equations) package from R. This version assumes all
    of the features are Gaussian.

    Read more in the :ref:`User Guide <mice>`.

    Parameters
    ----------
    missing_values : int, np.nan, optional (default=np.nan)
        The placeholder for the missing values. All occurrences of
        ``missing_values`` will be imputed.

    imputation_order : str, optional (default="ascending")
        The order in which the features will be imputed. Possible values:

        "ascending"
            From features with fewest missing values to most.
        "descending"
            From features with most missing values to fewest.
        "roman"
            Left to right.
        "arabic"
            Right to left.
        "random"
            A random order for each round.

    n_imputations : int, optional (default=100)
        Number of chained imputation rounds to perform, the results of which
        will be used in the final average.

    n_burn_in : int, optional (default=10)
        Number of initial imputation rounds to perform the results of which
        will not be returned.

    predictor : estimator object, default=BayesianRidge()
        The predictor to use at each step of the round-robin imputation.
        It must support ``return_std`` in its ``predict`` method.

    n_nearest_features : int, optional (default=None)
        Number of other features to use to estimate the missing values of
        the each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). Can provide significant speed-up when the number
        of features is huge. If ``None``, all features will be used.

    initial_strategy : str, optional (default="mean")
        Which strategy to use to initialize the missing values. Same as the
        ``strategy`` parameter in :class:`sklearn.impute.SimpleImputer`
        Valid values: {"mean", "median", "most_frequent", or "constant"}.

    min_value : float, optional (default=None)
        Minimum possible imputed value. Default of ``None`` will set minimum
        to negative infinity.

    max_value : float, optional (default=None)
        Maximum possible imputed value. Default of ``None`` will set maximum
        to positive infinity.

    verbose : int, optional (default=0)
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated. The higher, the more verbose. Can be 0, 1,
        or 2.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by ``np.random``.

    Attributes
    ----------
    initial_imputer_ : object of class :class:`sklearn.preprocessing.Imputer`'
        The imputer used to initialize the missing values.

    imputation_sequence_ : list of tuples
        Each tuple has ``(feat_idx, neighbor_feat_idx, predictor)``, where
        ``feat_idx`` is the current feature to be imputed,
        ``neighbor_feat_idx`` is the array of other features used to impute the
        current feature, and ``predictor`` is the trained predictor used for
        the imputation.

    Notes
    -----
    The R version of MICE does not have inductive functionality, i.e. first
    fitting on ``X_train`` and then transforming any ``X_test`` without
    additional fitting. We do this by storing each feature's predictor during
    the round-robin ``fit`` phase, and predicting without refitting (in order)
    during the ``transform`` phase.

    Features which contain all missing values at ``fit`` are discarded upon
    ``transform``.

    Features with missing values in transform which did not have any missing
    values in fit will be imputed with the initial imputation method only.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_
    """
```
### 18 - sklearn/impute.py:

Start line: 1, End line: 40

```python
"""Transformers for missing value imputation"""

from __future__ import division

import warnings
from time import time
import numbers

import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats
from collections import namedtuple

from .base import BaseEstimator, TransformerMixin
from .base import clone
from .preprocessing import normalize
from .utils import check_array, check_random_state, safe_indexing
from .utils.sparsefuncs import _get_median
from .utils.validation import check_is_fitted
from .utils.validation import FLOAT_DTYPES
from .utils.fixes import _object_dtype_isnan
from .utils import is_scalar_nan

from .externals import six

zip = six.moves.zip
map = six.moves.map

ImputerTriplet = namedtuple('ImputerTriplet', ['feat_idx',
                                               'neighbor_feat_idx',
                                               'predictor'])

__all__ = [
    'SimpleImputer',
    'ChainedImputer',
]
```
### 19 - sklearn/impute.py:

Start line: 808, End line: 889

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None):
        """Fits the imputer on X and return the transformed X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data, where "n_samples" is the number of samples and
            "n_features" is the number of features.

        y : ignored.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
             The imputed input data.
        """
        self.random_state_ = getattr(self, "random_state_",
                                     check_random_state(self.random_state))

        if self.predictor is None:
            from .linear_model import BayesianRidge
            self._predictor = BayesianRidge()
        else:
            self._predictor = clone(self.predictor)

        self._min_value = np.nan if self.min_value is None else self.min_value
        self._max_value = np.nan if self.max_value is None else self.max_value

        self.initial_imputer_ = None
        X, X_filled, mask_missing_values = self._initial_imputation(X)

        # edge case: in case the user specifies 0 for n_imputations,
        # then there is no need to do burn in and the result should be
        # just the initial imputation (before clipping)
        if self.n_imputations < 1:
            return X_filled

        X_filled = np.clip(X_filled, self._min_value, self._max_value)

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)

        abs_corr_mat = self._get_abs_corr_mat(X_filled)

        # impute data
        n_rounds = self.n_burn_in + self.n_imputations
        n_samples, n_features = X_filled.shape
        Xt = np.zeros((n_samples, n_features), dtype=X.dtype)
        self.imputation_sequence_ = []
        if self.verbose > 0:
            print("[ChainedImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        for i_rnd in range(n_rounds):
            if self.imputation_order == 'random':
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(n_features,
                                                                feat_idx,
                                                                abs_corr_mat)
                X_filled, predictor = self._impute_one_feature(
                    X_filled, mask_missing_values, feat_idx, neighbor_feat_idx,
                    predictor=None, fit_mode=True)
                predictor_triplet = ImputerTriplet(feat_idx,
                                                   neighbor_feat_idx,
                                                   predictor)
                self.imputation_sequence_.append(predictor_triplet)

            if i_rnd >= self.n_burn_in:
                Xt += X_filled
            if self.verbose > 0:
                print('[ChainedImputer] Ending imputation round '
                      '%d/%d, elapsed time %0.2f'
                      % (i_rnd + 1, n_rounds, time() - start_t))

        Xt /= self.n_imputations
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt
```
### 23 - sklearn/impute.py:

Start line: 683, End line: 725

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def _get_ordered_idx(self, mask_missing_values):
        """Decide in what order we will update the features.

        As a homage to the MICE R package, we will have 4 main options of
        how to order the updates, and use a random order if anything else
        is specified.

        Also, this function skips features which have no missing values.

        Parameters
        ----------
        mask_missing_values : array-like, shape (n_samples, n_features)
            Input data's missing indicator matrix, where "n_samples" is the
            number of samples and "n_features" is the number of features.

        Returns
        -------
        ordered_idx : ndarray, shape (n_features,)
            The order in which to impute the features.
        """
        frac_of_missing_values = mask_missing_values.mean(axis=0)
        missing_values_idx = np.nonzero(frac_of_missing_values)[0]
        if self.imputation_order == 'roman':
            ordered_idx = missing_values_idx
        elif self.imputation_order == 'arabic':
            ordered_idx = missing_values_idx[::-1]
        elif self.imputation_order == 'ascending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values,
                                     kind='mergesort')[n:][::-1]
        elif self.imputation_order == 'descending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values,
                                     kind='mergesort')[n:]
        elif self.imputation_order == 'random':
            ordered_idx = missing_values_idx
            self.random_state_.shuffle(ordered_idx)
        else:
            raise ValueError("Got an invalid imputation order: '{0}'. It must "
                             "be one of the following: 'roman', 'arabic', "
                             "'ascending', 'descending', or "
                             "'random'.".format(self.imputation_order))
        return ordered_idx
```
### 24 - sklearn/impute.py:

Start line: 727, End line: 760

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def _get_abs_corr_mat(self, X_filled, tolerance=1e-6):
        """Get absolute correlation matrix between features.

        Parameters
        ----------
        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        tolerance : float, optional (default=1e-6)
            ``abs_corr_mat`` can have nans, which will be replaced
            with ``tolerance``.

        Returns
        -------
        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of ``X`` at the beginning of the
            current round. The diagonal has been zeroed out and each feature's
            absolute correlations with all others have been normalized to sum
            to 1.
        """
        n_features = X_filled.shape[1]
        if (self.n_nearest_features is None or
                self.n_nearest_features >= n_features):
            return None
        abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
        # np.corrcoef is not defined for features with zero std
        abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
        # ensures exploration, i.e. at least some probability of sampling
        np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
        # features are not their own neighbors
        np.fill_diagonal(abs_corr_mat, 0)
        # needs to sum to 1 for np.random.choice sampling
        abs_corr_mat = normalize(abs_corr_mat, norm='l1', axis=0, copy=False)
        return abs_corr_mat
```
### 26 - sklearn/impute.py:

Start line: 533, End line: 556

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 missing_values=np.nan,
                 imputation_order='ascending',
                 n_imputations=100,
                 n_burn_in=10,
                 predictor=None,
                 n_nearest_features=None,
                 initial_strategy="mean",
                 min_value=None,
                 max_value=None,
                 verbose=False,
                 random_state=None):

        self.missing_values = missing_values
        self.imputation_order = imputation_order
        self.n_imputations = n_imputations
        self.n_burn_in = n_burn_in
        self.predictor = predictor
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
```
### 30 - sklearn/impute.py:

Start line: 61, End line: 92

```python
def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
       [extra_value] * n_repeat, where extra_value is assumed to be not part
       of the array."""
    # Compute the most frequent value in array only
    if array.size > 0:
        with warnings.catch_warnings():
            # stats.mode raises a warning when input array contains objects due
            # to incapacity to detect NaNs. Irrelevant here since input array
            # has already been NaN-masked.
            warnings.simplefilter("ignore", RuntimeWarning)
            mode = stats.mode(array)

        most_frequent_value = mode[0][0]
        most_frequent_count = mode[1][0]
    else:
        most_frequent_value = 0
        most_frequent_count = 0

    # Compare to array + [extra_value] * n_repeat
    if most_frequent_count == 0 and n_repeat == 0:
        return np.nan
    elif most_frequent_count < n_repeat:
        return extra_value
    elif most_frequent_count > n_repeat:
        return most_frequent_value
    elif most_frequent_count == n_repeat:
        # Ties the breaks. Copy the behaviour of scipy.stats.mode
        if most_frequent_value < extra_value:
            return most_frequent_value
        else:
            return extra_value
```
### 92 - sklearn/impute.py:

Start line: 643, End line: 681

```python
class ChainedImputer(BaseEstimator, TransformerMixin):

    def _get_neighbor_feat_idx(self,
                               n_features,
                               feat_idx,
                               abs_corr_mat):
        """Get a list of other features to predict ``feat_idx``.

        If self.n_nearest_features is less than or equal to the total
        number of features, then use a probability proportional to the absolute
        correlation between ``feat_idx`` and each other feature to randomly
        choose a subsample of the other features (without replacement).

        Parameters
        ----------
        n_features : int
            Number of features in ``X``.

        feat_idx : int
            Index of the feature currently being imputed.

        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of ``X``. The diagonal has been zeroed
            out and each feature has been normalized to sum to 1. Can be None.

        Returns
        -------
        neighbor_feat_idx : array-like
            The features to use to impute ``feat_idx``.
        """
        if (self.n_nearest_features is not None and
                self.n_nearest_features < n_features):
            p = abs_corr_mat[:, feat_idx]
            neighbor_feat_idx = self.random_state_.choice(
                np.arange(n_features), self.n_nearest_features, replace=False,
                p=p)
        else:
            inds_left = np.arange(feat_idx)
            inds_right = np.arange(feat_idx + 1, n_features)
            neighbor_feat_idx = np.concatenate((inds_left, inds_right))
        return neighbor_feat_idx
```

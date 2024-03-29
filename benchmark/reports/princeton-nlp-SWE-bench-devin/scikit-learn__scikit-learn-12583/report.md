# scikit-learn__scikit-learn-12583

| **scikit-learn/scikit-learn** | `e8c6cb151cff869cf1b61bddd3c72841318501ab` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 10418 |
| **Any found context length** | 2842 |
| **Avg pos** | 102.0 |
| **Min pos** | 7 |
| **Max pos** | 23 |
| **Top file pos** | 1 |
| **Missing snippets** | 7 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/impute.py | 144 | 144 | 7 | 1 | 2842
| sklearn/impute.py | 162 | 163 | 7 | 1 | 2842
| sklearn/impute.py | 178 | 178 | 7 | 1 | 2842
| sklearn/impute.py | 275 | 275 | 16 | 1 | 7614
| sklearn/impute.py | 288 | 288 | 23 | 1 | 10418
| sklearn/impute.py | 385 | 385 | 21 | 1 | 9436
| sklearn/impute.py | 423 | 423 | 21 | 1 | 9436


## Problem Statement

```
add_indicator switch in imputers
For whatever imputers we have, but especially [SimpleImputer](http://scikit-learn.org/dev/modules/generated/sklearn.impute.SimpleImputer.html), we should have an `add_indicator` parameter, which simply stacks a [MissingIndicator](http://scikit-learn.org/dev/modules/generated/sklearn.impute.MissingIndicator.html) transform onto the output of the imputer's `transform`.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/impute.py** | 1044 | 1115| 650 | 650 | 10656 | 
| 2 | **1 sklearn/impute.py** | 1172 | 1195| 254 | 904 | 10656 | 
| 3 | **1 sklearn/impute.py** | 1227 | 1261| 272 | 1176 | 10656 | 
| 4 | **1 sklearn/impute.py** | 1263 | 1283| 145 | 1321 | 10656 | 
| 5 | **1 sklearn/impute.py** | 1117 | 1170| 429 | 1750 | 10656 | 
| 6 | **1 sklearn/impute.py** | 185 | 222| 343 | 2093 | 10656 | 
| **-> 7 <-** | **1 sklearn/impute.py** | 103 | 183| 749 | 2842 | 10656 | 
| 8 | **1 sklearn/impute.py** | 661 | 700| 499 | 3341 | 10656 | 
| 9 | **1 sklearn/impute.py** | 429 | 562| 1326 | 4667 | 10656 | 
| 10 | **1 sklearn/impute.py** | 1 | 49| 298 | 4965 | 10656 | 
| 11 | **1 sklearn/impute.py** | 1197 | 1225| 228 | 5193 | 10656 | 
| 12 | 2 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 84| 741 | 5934 | 11781 | 
| 13 | 2 examples/impute/plot_iterative_imputer_variants_comparison.py | 85 | 131| 384 | 6318 | 11781 | 
| 14 | **2 sklearn/impute.py** | 936 | 971| 344 | 6662 | 11781 | 
| 15 | 3 sklearn/preprocessing/imputation.py | 60 | 123| 564 | 7226 | 14715 | 
| **-> 16 <-** | **3 sklearn/impute.py** | 224 | 275| 388 | 7614 | 14715 | 
| 17 | 4 examples/impute/plot_missing_values.py | 1 | 46| 371 | 7985 | 15919 | 
| 18 | **4 sklearn/impute.py** | 564 | 589| 189 | 8174 | 15919 | 
| 19 | **4 sklearn/impute.py** | 742 | 784| 428 | 8602 | 15919 | 
| 20 | 5 sklearn/preprocessing/label.py | 957 | 993| 366 | 8968 | 23555 | 
| **-> 21 <-** | **5 sklearn/impute.py** | 367 | 426| 468 | 9436 | 23555 | 
| 22 | 5 sklearn/preprocessing/imputation.py | 294 | 373| 666 | 10102 | 23555 | 
| **-> 23 <-** | **5 sklearn/impute.py** | 277 | 313| 316 | 10418 | 23555 | 
| 24 | 5 sklearn/preprocessing/imputation.py | 169 | 247| 675 | 11093 | 23555 | 
| 25 | 5 sklearn/preprocessing/imputation.py | 125 | 167| 382 | 11475 | 23555 | 
| 26 | **5 sklearn/impute.py** | 821 | 866| 398 | 11873 | 23555 | 
| 27 | **5 sklearn/impute.py** | 973 | 1019| 393 | 12266 | 23555 | 
| 28 | 5 sklearn/preprocessing/imputation.py | 249 | 292| 434 | 12700 | 23555 | 
| 29 | **5 sklearn/impute.py** | 315 | 365| 460 | 13160 | 23555 | 
| 30 | **5 sklearn/impute.py** | 1021 | 1041| 127 | 13287 | 23555 | 
| 31 | **5 sklearn/impute.py** | 868 | 935| 556 | 13843 | 23555 | 
| 32 | 5 examples/impute/plot_missing_values.py | 96 | 141| 391 | 14234 | 23555 | 
| 33 | **5 sklearn/impute.py** | 591 | 660| 526 | 14760 | 23555 | 
| 34 | 6 sklearn/base.py | 572 | 636| 373 | 15133 | 28194 | 
| 35 | 7 sklearn/preprocessing/data.py | 1963 | 2019| 543 | 15676 | 53345 | 
| 36 | 8 sklearn/kernel_approximation.py | 410 | 439| 284 | 15960 | 58577 | 
| 37 | 8 sklearn/preprocessing/imputation.py | 4 | 29| 141 | 16101 | 58577 | 
| 38 | 9 sklearn/preprocessing/__init__.py | 1 | 71| 419 | 16520 | 58996 | 
| 39 | 9 sklearn/preprocessing/label.py | 890 | 921| 233 | 16753 | 58996 | 
| 40 | 10 sklearn/ensemble/gradient_boosting.py | 1368 | 1393| 296 | 17049 | 80289 | 
| 41 | 10 examples/impute/plot_missing_values.py | 49 | 93| 440 | 17489 | 80289 | 
| 42 | 11 sklearn/feature_extraction/text.py | 1293 | 1644| 157 | 17646 | 93527 | 
| 43 | 12 sklearn/multioutput.py | 62 | 121| 485 | 18131 | 99090 | 
| 44 | 12 sklearn/multioutput.py | 1 | 40| 264 | 18395 | 99090 | 
| 45 | 13 examples/compose/plot_column_transformer_mixed_types.py | 1 | 104| 799 | 19194 | 99911 | 
| 46 | **13 sklearn/impute.py** | 702 | 740| 330 | 19524 | 99911 | 
| 47 | 14 sklearn/feature_extraction/dict_vectorizer.py | 135 | 209| 603 | 20127 | 102632 | 
| 48 | 14 sklearn/base.py | 420 | 451| 221 | 20348 | 102632 | 
| 49 | 15 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 21163 | 105740 | 
| 50 | 15 sklearn/preprocessing/label.py | 923 | 955| 237 | 21400 | 105740 | 
| 51 | 15 sklearn/multioutput.py | 234 | 259| 189 | 21589 | 105740 | 
| 52 | 15 sklearn/kernel_approximation.py | 384 | 408| 231 | 21820 | 105740 | 
| 53 | 15 sklearn/ensemble/gradient_boosting.py | 1764 | 2012| 2533 | 24353 | 105740 | 
| 54 | 15 sklearn/ensemble/gradient_boosting.py | 2228 | 2471| 2515 | 26868 | 105740 | 
| 55 | 15 sklearn/preprocessing/label.py | 289 | 378| 741 | 27609 | 105740 | 
| 56 | 16 sklearn/utils/estimator_checks.py | 1151 | 1219| 601 | 28210 | 127637 | 
| 57 | 16 sklearn/preprocessing/data.py | 2350 | 2381| 227 | 28437 | 127637 | 
| 58 | 17 sklearn/pipeline.py | 208 | 242| 284 | 28721 | 134905 | 
| 59 | 17 sklearn/preprocessing/data.py | 2250 | 2317| 677 | 29398 | 134905 | 
| 60 | 17 sklearn/preprocessing/label.py | 847 | 888| 346 | 29744 | 134905 | 
| 61 | 18 sklearn/compose/_column_transformer.py | 502 | 535| 283 | 30027 | 141362 | 
| 62 | 18 sklearn/utils/estimator_checks.py | 2211 | 2252| 399 | 30426 | 141362 | 
| 63 | 18 sklearn/pipeline.py | 29 | 124| 1039 | 31465 | 141362 | 
| 64 | 19 sklearn/utils/testing.py | 665 | 692| 280 | 31745 | 148684 | 
| 65 | 19 sklearn/preprocessing/data.py | 2319 | 2338| 215 | 31960 | 148684 | 
| 66 | 20 sklearn/ensemble/weight_boosting.py | 550 | 596| 397 | 32357 | 157800 | 
| 67 | 21 sklearn/preprocessing/_encoders.py | 107 | 852| 334 | 32691 | 166330 | 
| 68 | 21 sklearn/preprocessing/_encoders.py | 72 | 105| 291 | 32982 | 166330 | 
| 69 | 21 sklearn/preprocessing/_encoders.py | 676 | 710| 326 | 33308 | 166330 | 
| 70 | 21 sklearn/ensemble/gradient_boosting.py | 1136 | 1163| 299 | 33607 | 166330 | 
| 71 | 22 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 50 | 116| 672 | 34279 | 167343 | 
| 72 | 22 sklearn/ensemble/weight_boosting.py | 492 | 548| 534 | 34813 | 167343 | 
| 73 | 22 sklearn/ensemble/weight_boosting.py | 1003 | 1093| 694 | 35507 | 167343 | 
| 74 | 23 sklearn/feature_selection/mutual_info_.py | 271 | 292| 246 | 35753 | 171339 | 
| 75 | 23 sklearn/preprocessing/data.py | 11 | 59| 309 | 36062 | 171339 | 
| 76 | 23 sklearn/utils/estimator_checks.py | 950 | 972| 248 | 36310 | 171339 | 
| 77 | 23 sklearn/preprocessing/label.py | 770 | 817| 394 | 36704 | 171339 | 
| 78 | 23 sklearn/preprocessing/_encoders.py | 144 | 316| 1798 | 38502 | 171339 | 
| 79 | 24 sklearn/metrics/cluster/supervised.py | 728 | 764| 456 | 38958 | 180109 | 
| 80 | 25 sklearn/preprocessing/_discretization.py | 1 | 115| 1026 | 39984 | 182937 | 
| 81 | 26 sklearn/feature_selection/from_model.py | 142 | 162| 207 | 40191 | 184737 | 
| 82 | 26 sklearn/utils/estimator_checks.py | 321 | 397| 714 | 40905 | 184737 | 
| 83 | 27 sklearn/feature_selection/base.py | 1 | 47| 299 | 41204 | 185641 | 
| 84 | 27 sklearn/ensemble/weight_boosting.py | 447 | 490| 323 | 41527 | 185641 | 
| 85 | 27 sklearn/preprocessing/data.py | 2676 | 2703| 293 | 41820 | 185641 | 
| 86 | 27 sklearn/preprocessing/data.py | 2340 | 2348| 119 | 41939 | 185641 | 
| 87 | 28 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 90| 749 | 42688 | 186607 | 
| 88 | 28 sklearn/ensemble/gradient_boosting.py | 1472 | 1550| 742 | 43430 | 186607 | 
| 89 | 28 sklearn/preprocessing/label.py | 726 | 767| 342 | 43772 | 186607 | 
| 90 | 29 benchmarks/bench_mnist.py | 84 | 105| 314 | 44086 | 188325 | 
| 91 | **29 sklearn/impute.py** | 52 | 66| 142 | 44228 | 188325 | 
| 92 | 30 sklearn/multiclass.py | 414 | 436| 238 | 44466 | 194764 | 
| 93 | 30 sklearn/preprocessing/data.py | 1785 | 1854| 580 | 45046 | 194764 | 
| 94 | 31 examples/compose/plot_column_transformer.py | 1 | 54| 370 | 45416 | 195734 | 
| 95 | 31 sklearn/feature_extraction/dict_vectorizer.py | 101 | 133| 208 | 45624 | 195734 | 
| 96 | 32 sklearn/preprocessing/_function_transformer.py | 138 | 184| 254 | 45878 | 197000 | 
| 97 | 33 examples/cluster/plot_inductive_clustering.py | 38 | 56| 149 | 46027 | 197909 | 
| 98 | 33 sklearn/preprocessing/data.py | 2383 | 2402| 161 | 46188 | 197909 | 
| 99 | 33 sklearn/feature_extraction/dict_vectorizer.py | 211 | 229| 144 | 46332 | 197909 | 
| 100 | 33 sklearn/preprocessing/_function_transformer.py | 105 | 116| 127 | 46459 | 197909 | 
| 101 | 33 sklearn/preprocessing/_discretization.py | 117 | 207| 870 | 47329 | 197909 | 
| 102 | 33 sklearn/kernel_approximation.py | 353 | 382| 265 | 47594 | 197909 | 
| 103 | 33 sklearn/multiclass.py | 638 | 711| 690 | 48284 | 197909 | 
| 104 | 33 sklearn/preprocessing/data.py | 677 | 741| 571 | 48855 | 197909 | 
| 105 | 33 sklearn/pipeline.py | 471 | 498| 190 | 49045 | 197909 | 
| 106 | 33 sklearn/feature_selection/from_model.py | 4 | 34| 228 | 49273 | 197909 | 
| 107 | 33 sklearn/preprocessing/_encoders.py | 532 | 544| 147 | 49420 | 197909 | 


### Hint

```
This allows downstream models to adjust for the fact that a value was imputed, rather than observed.
Can I  take this up if no  one else is working on it yet @jnothman ?
Go for it
@prathusha94 are you still working on this?
```

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


## Code snippets

### 1 - sklearn/impute.py:

Start line: 1044, End line: 1115

```python
class MissingIndicator(BaseEstimator, TransformerMixin):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Read more in the :ref:`User Guide <impute>`.

    Parameters
    ----------
    missing_values : number, string, np.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be indicated (True in the output array), the
        other values will be marked as False.

    features : str, optional
        Whether the imputer mask should represent all or a subset of
        features.

        - If "missing-only" (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If "all", the imputer mask will represent all features.

    sparse : boolean or "auto", optional
        Whether the imputer mask format should be sparse or dense.

        - If "auto" (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : boolean, optional
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features="missing-only"``.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)  # doctest: +NORMALIZE_WHITESPACE
    MissingIndicator(error_on_new=True, features='missing-only',
             missing_values=nan, sparse='auto')
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

    """

    def __init__(self, missing_values=np.nan, features="missing-only",
                 sparse="auto", error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new
```
### 2 - sklearn/impute.py:

Start line: 1172, End line: 1195

```python
class MissingIndicator(BaseEstimator, TransformerMixin):

    def _validate_input(self, X):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = check_array(X, accept_sparse=('csc', 'csr'), dtype=None,
                        force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("MissingIndicator does not support data with "
                             "dtype {0}. Please provide either a numeric array"
                             " (with a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        if sparse.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError("Sparse input with missing_values=0 is "
                             "not supported. Provide a dense "
                             "array instead.")

        return X
```
### 3 - sklearn/impute.py:

Start line: 1227, End line: 1261

```python
class MissingIndicator(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self, "features_")
        X = self._validate_input(X)

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = np.setdiff1d(features, self.features_)
            if (self.error_on_new and features_diff_fit_trans.size > 0):
                raise ValueError("The features {} have missing values "
                                 "in transform but have no missing values "
                                 "in fit.".format(features_diff_fit_trans))

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask
```
### 4 - sklearn/impute.py:

Start line: 1263, End line: 1283

```python
class MissingIndicator(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {'allow_nan': True,
                'X_types': ['2darray', 'str']}
```
### 5 - sklearn/impute.py:

Start line: 1117, End line: 1170

```python
class MissingIndicator(BaseEstimator, TransformerMixin):

    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The input data with missing values. Note that ``X`` has been
            checked in ``fit`` and ``transform`` before to call this function.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape \
        es, n_features) or (n_samples, n_features_with_missing)
            The imputer mask of the original data.

        features_with_missing : ndarray, shape (n_features_with_missing)
            The features containing missing values.

        """
        if sparse.issparse(X):
            mask = _get_mask(X.data, self.missing_values)

            # The imputer mask will be constructed with the same sparse format
            # as X.
            sparse_constructor = (sparse.csr_matrix if X.format == 'csr'
                                  else sparse.csc_matrix)
            imputer_mask = sparse_constructor(
                (mask, X.indices.copy(), X.indptr.copy()),
                shape=X.shape, dtype=bool)
            imputer_mask.eliminate_zeros()

            if self.features == 'missing-only':
                n_missing = imputer_mask.getnnz(axis=0)

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            imputer_mask = _get_mask(X, self.missing_values)

            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is True:
                imputer_mask = sparse.csc_matrix(imputer_mask)

        if self.features == 'all':
            features_indices = np.arange(X.shape[1])
        else:
            features_indices = np.flatnonzero(n_missing)

        return imputer_mask, features_indices
```
### 6 - sklearn/impute.py:

Start line: 185, End line: 222

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

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("SimpleImputer does not support data with dtype "
                             "{0}. Please provide either a numeric array (with"
                             " a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        return X
```
### 7 - sklearn/impute.py:

Start line: 103, End line: 183

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
        - If X is encoded as a CSR matrix.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.

    See also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    ... # doctest: +NORMALIZE_WHITESPACE
    SimpleImputer(copy=True, fill_value=None, missing_values=nan,
           strategy='mean', verbose=0)
    >>> X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
    >>> print(imp_mean.transform(X))
    ... # doctest: +NORMALIZE_WHITESPACE
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

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
### 8 - sklearn/impute.py:

Start line: 661, End line: 700

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

    def _impute_one_feature(self,
                            X_filled,
                            mask_missing_values,
                            feat_idx,
                            neighbor_feat_idx,
                            estimator=None,
                            fit_mode=True):
        # ... other code
        if self.sample_posterior:
            mus, sigmas = estimator.predict(X_test, return_std=True)
            imputed_values = np.zeros(mus.shape, dtype=X_filled.dtype)
            # two types of problems: (1) non-positive sigmas, (2) mus outside
            # legal range of min_value and max_value (results in inf sample)
            positive_sigmas = sigmas > 0
            imputed_values[~positive_sigmas] = mus[~positive_sigmas]
            mus_too_low = mus < self._min_value
            imputed_values[mus_too_low] = self._min_value
            mus_too_high = mus > self._max_value
            imputed_values[mus_too_high] = self._max_value
            # the rest can be sampled without statistical issues
            inrange_mask = positive_sigmas & ~mus_too_low & ~mus_too_high
            mus = mus[inrange_mask]
            sigmas = sigmas[inrange_mask]
            a = (self._min_value - mus) / sigmas
            b = (self._max_value - mus) / sigmas

            if scipy.__version__ < LooseVersion('0.18'):
                # bug with vector-valued `a` in old scipy
                imputed_values[inrange_mask] = [
                    stats.truncnorm(a=a_, b=b_,
                                    loc=loc_, scale=scale_).rvs(
                                        random_state=self.random_state_)
                    for a_, b_, loc_, scale_
                    in zip(a, b, mus, sigmas)]
            else:
                truncated_normal = stats.truncnorm(a=a, b=b,
                                                   loc=mus, scale=sigmas)
                imputed_values[inrange_mask] = truncated_normal.rvs(
                    random_state=self.random_state_)
        else:
            imputed_values = estimator.predict(X_test)
            imputed_values = np.clip(imputed_values,
                                     self._min_value,
                                     self._max_value)

        # update the feature
        X_filled[missing_row_mask, feat_idx] = imputed_values
        return X_filled, estimator
```
### 9 - sklearn/impute.py:

Start line: 429, End line: 562

```python
class IterativeImputer(BaseEstimator, TransformerMixin):
    """Multivariate imputer that estimates each feature from all the others.

    A strategy for imputing missing values by modeling each feature with
    missing values as a function of other features in a round-robin fashion.

    Read more in the :ref:`User Guide <iterative_imputer>`.

    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use at each step of the round-robin imputation.
        If ``sample_posterior`` is True, the estimator must support
        ``return_std`` in its ``predict`` method.

    missing_values : int, np.nan, optional (default=np.nan)
        The placeholder for the missing values. All occurrences of
        ``missing_values`` will be imputed.

    sample_posterior : boolean, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. Estimator must support
        ``return_std`` in its ``predict`` method if set to ``True``. Set to
        ``True`` if using ``IterativeImputer`` for multiple imputations.

    max_iter : int, optional (default=10)
        Maximum number of imputation rounds to perform before returning the
        imputations computed during the final round. A round is a single
        imputation of each feature with missing values. The stopping criterion
        is met once `abs(max(X_t - X_{t-1}))/abs(max(X[known_vals]))` < tol,
        where `X_t` is `X` at iteration `t. Note that early stopping is only
        applied if ``sample_posterior=False``.

    tol : float, optional (default=1e-3)
        Tolerance of the stopping condition.

    n_nearest_features : int, optional (default=None)
        Number of other features to use to estimate the missing values of
        each feature column. Nearness between features is measured using
        the absolute correlation coefficient between each feature pair (after
        initial imputation). To ensure coverage of features throughout the
        imputation process, the neighbor features are not necessarily nearest,
        but are drawn with probability proportional to correlation for each
        imputed target feature. Can provide significant speed-up when the
        number of features is huge. If ``None``, all features will be used.

    initial_strategy : str, optional (default="mean")
        Which strategy to use to initialize the missing values. Same as the
        ``strategy`` parameter in :class:`sklearn.impute.SimpleImputer`
        Valid values: {"mean", "median", "most_frequent", or "constant"}.

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
        The seed of the pseudo random number generator to use. Randomizes
        selection of estimator features if n_nearest_features is not None, the
        ``imputation_order`` if ``random``, and the sampling from posterior if
        ``sample_posterior`` is True. Use an integer for determinism.
        See :term:`the Glossary <random_state>`.

    Attributes
    ----------
    initial_imputer_ : object of type :class:`sklearn.impute.SimpleImputer`
        Imputer used to initialize the missing values.

    imputation_sequence_ : list of tuples
        Each tuple has ``(feat_idx, neighbor_feat_idx, estimator)``, where
        ``feat_idx`` is the current feature to be imputed,
        ``neighbor_feat_idx`` is the array of other features used to impute the
        current feature, and ``estimator`` is the trained estimator used for
        the imputation. Length is ``self.n_features_with_missing_ *
        self.n_iter_``.

    n_iter_ : int
        Number of iteration rounds that occurred. Will be less than
        ``self.max_iter`` if early stopping criterion was reached.

    n_features_with_missing_ : int
        Number of features with missing values.

    See also
    --------
    SimpleImputer : Univariate imputation of missing values.

    Notes
    -----
    To support imputation in inductive mode we store each feature's estimator
    during the ``fit`` phase, and predict without refitting (in order) during
    the ``transform`` phase.

    Features which contain all missing values at ``fit`` are discarded upon
    ``transform``.

    Features with missing values during ``transform`` which did not have any
    missing values during ``fit`` will be imputed with the initial imputation
    method only.

    References
    ----------
    .. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice:
        Multivariate Imputation by Chained Equations in R". Journal of
        Statistical Software 45: 1-67.
        <https://www.jstatsoft.org/article/view/v045i03>`_

    .. [2] `S. F. Buck, (1960). "A Method of Estimation of Missing Values in
        Multivariate Data Suitable for use with an Electronic Computer".
        Journal of the Royal Statistical Society 22(2): 302-306.
        <https://www.jstor.org/stable/2984099>`_
    """
```
### 10 - sklearn/impute.py:

Start line: 1, End line: 49

```python
"""Transformers for missing value imputation"""

from __future__ import division

import warnings
import numbers
from time import time
from distutils.version import LooseVersion

import numpy as np
import numpy.ma as ma
import scipy
from scipy import sparse
from scipy import stats
from collections import namedtuple

from .base import BaseEstimator, TransformerMixin
from .base import clone
from .exceptions import ConvergenceWarning
from .preprocessing import normalize
from .utils import check_array, check_random_state, safe_indexing
from .utils.sparsefuncs import _get_median
from .utils.validation import check_is_fitted
from .utils.validation import FLOAT_DTYPES
from .utils.fixes import _object_dtype_isnan
from .utils import is_scalar_nan


ImputerTriplet = namedtuple('ImputerTriplet', ['feat_idx',
                                               'neighbor_feat_idx',
                                               'estimator'])

__all__ = [
    'MissingIndicator',
    'SimpleImputer',
    'IterativeImputer',
]


def _check_inputs_dtype(X, missing_values):
    if (X.dtype.kind in ("f", "i", "u") and
            not isinstance(missing_values, numbers.Real)):
        raise ValueError("'X' and 'missing_values' types are expected to be"
                         " both numerical. Got X.dtype={} and "
                         " type(missing_values)={}."
                         .format(X.dtype, type(missing_values)))
```
### 11 - sklearn/impute.py:

Start line: 1197, End line: 1225

```python
class MissingIndicator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        X = self._validate_input(X)
        self._n_features = X.shape[1]

        if self.features not in ('missing-only', 'all'):
            raise ValueError("'features' has to be either 'missing-only' or "
                             "'all'. Got {} instead.".format(self.features))

        if not ((isinstance(self.sparse, str) and
                self.sparse == "auto") or isinstance(self.sparse, bool)):
            raise ValueError("'sparse' has to be a boolean or 'auto'. "
                             "Got {!r} instead.".format(self.sparse))

        self.features_ = self._get_missing_features_info(X)[1]

        return self
```
### 14 - sklearn/impute.py:

Start line: 936, End line: 971

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

    def fit_transform(self, X, y=None):
        # ... other code
        for self.n_iter_ in range(1, self.max_iter + 1):
            if self.imputation_order == 'random':
                ordered_idx = self._get_ordered_idx(mask_missing_values)

            for feat_idx in ordered_idx:
                neighbor_feat_idx = self._get_neighbor_feat_idx(n_features,
                                                                feat_idx,
                                                                abs_corr_mat)
                Xt, estimator = self._impute_one_feature(
                    Xt, mask_missing_values, feat_idx, neighbor_feat_idx,
                    estimator=None, fit_mode=True)
                estimator_triplet = ImputerTriplet(feat_idx,
                                                   neighbor_feat_idx,
                                                   estimator)
                self.imputation_sequence_.append(estimator_triplet)

            if self.verbose > 1:
                print('[IterativeImputer] Ending imputation round '
                      '%d/%d, elapsed time %0.2f'
                      % (self.n_iter_, self.max_iter, time() - start_t))

            if not self.sample_posterior:
                inf_norm = np.linalg.norm(Xt - Xt_previous, ord=np.inf,
                                          axis=None)
                if inf_norm < normalized_tol:
                    if self.verbose > 0:
                        print('[IterativeImputer] Early stopping criterion '
                              'reached.')
                    break
                Xt_previous = Xt.copy()
        else:
            if not self.sample_posterior:
                warnings.warn("[IterativeImputer] Early stopping criterion not"
                              " reached.", ConvergenceWarning)
        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt
```
### 16 - sklearn/impute.py:

Start line: 224, End line: 275

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
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
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
### 18 - sklearn/impute.py:

Start line: 564, End line: 589

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 estimator=None,
                 missing_values=np.nan,
                 sample_posterior=False,
                 max_iter=10,
                 tol=1e-3,
                 n_nearest_features=None,
                 initial_strategy="mean",
                 imputation_order='ascending',
                 min_value=None,
                 max_value=None,
                 verbose=0,
                 random_state=None):

        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.random_state = random_state
```
### 19 - sklearn/impute.py:

Start line: 742, End line: 784

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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
### 21 - sklearn/impute.py:

Start line: 367, End line: 426

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
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
        if sparse.issparse(X):
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                mask = _get_mask(X.data, self.missing_values)
                indexes = np.repeat(np.arange(len(X.indptr) - 1, dtype=np.int),
                                    np.diff(X.indptr))[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                                copy=False)
        else:
            mask = _get_mask(X, self.missing_values)
            n_missing = np.sum(mask, axis=0)
            values = np.repeat(valid_statistics, n_missing)
            coordinates = np.where(mask.transpose())[::-1]

            X[coordinates] = values

        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 23 - sklearn/impute.py:

Start line: 277, End line: 313

```python
class SimpleImputer(BaseEstimator, TransformerMixin):

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        mask_data = _get_mask(X.data, missing_values)
        n_implicit_zeros = X.shape[0] - np.diff(X.indptr)

        statistics = np.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            statistics.fill(fill_value)

        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i]:X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if strategy == "mean":
                    s = column.size + n_zeros
                    statistics[i] = np.nan if s == 0 else column.sum() / s

                elif strategy == "median":
                    statistics[i] = _get_median(column,
                                                n_zeros)

                elif strategy == "most_frequent":
                    statistics[i] = _most_frequent(column,
                                                   0,
                                                   n_zeros)
        return statistics
```
### 26 - sklearn/impute.py:

Start line: 821, End line: 866

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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
        _check_inputs_dtype(X, self.missing_values)

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
### 27 - sklearn/impute.py:

Start line: 973, End line: 1019

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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

        X, Xt, mask_missing_values = self._initial_imputation(X)

        if self.n_iter_ == 0 or np.all(mask_missing_values):
            return Xt

        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        i_rnd = 0
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        for it, estimator_triplet in enumerate(self.imputation_sequence_):
            Xt, _ = self._impute_one_feature(
                Xt,
                mask_missing_values,
                estimator_triplet.feat_idx,
                estimator_triplet.neighbor_feat_idx,
                estimator=estimator_triplet.estimator,
                fit_mode=False
            )
            if not (it + 1) % imputations_per_round:
                if self.verbose > 1:
                    print('[IterativeImputer] Ending imputation round '
                          '%d/%d, elapsed time %0.2f'
                          % (i_rnd + 1, self.n_iter_, time() - start_t))
                i_rnd += 1

        Xt[~mask_missing_values] = X[~mask_missing_values]
        return Xt
```
### 29 - sklearn/impute.py:

Start line: 315, End line: 365

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
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            return np.full(X.shape[1], fill_value, dtype=X.dtype)
```
### 30 - sklearn/impute.py:

Start line: 1021, End line: 1041

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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

    def _more_tags(self):
        return {'allow_nan': True}
```
### 31 - sklearn/impute.py:

Start line: 868, End line: 935

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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

        if self.max_iter < 0:
            raise ValueError(
                "'max_iter' should be a positive integer. Got {} instead."
                .format(self.max_iter))

        if self.tol < 0:
            raise ValueError(
                "'tol' should be a non-negative float. Got {} instead."
                .format(self.tol)
            )

        if self.estimator is None:
            from .linear_model import BayesianRidge
            self._estimator = BayesianRidge()
        else:
            self._estimator = clone(self.estimator)

        self.imputation_sequence_ = []

        if hasattr(self._estimator, 'random_state'):
            self._estimator.random_state = self.random_state_

        self._min_value = -np.inf if self.min_value is None else self.min_value
        self._max_value = np.inf if self.max_value is None else self.max_value

        self.initial_imputer_ = None
        X, Xt, mask_missing_values = self._initial_imputation(X)

        if self.max_iter == 0 or np.all(mask_missing_values):
            self.n_iter_ = 0
            return Xt

        # order in which to impute
        # note this is probably too slow for large feature data (d > 100000)
        # and a better way would be good.
        # see: https://goo.gl/KyCNwj and subsequent comments
        ordered_idx = self._get_ordered_idx(mask_missing_values)
        self.n_features_with_missing_ = len(ordered_idx)

        abs_corr_mat = self._get_abs_corr_mat(Xt)

        n_samples, n_features = Xt.shape
        if self.verbose > 0:
            print("[IterativeImputer] Completing matrix with shape %s"
                  % (X.shape,))
        start_t = time()
        if not self.sample_posterior:
            Xt_previous = Xt.copy()
            normalized_tol = self.tol * np.max(np.abs(X[~mask_missing_values]))
        # ... other code
```
### 33 - sklearn/impute.py:

Start line: 591, End line: 660

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

    def _impute_one_feature(self,
                            X_filled,
                            mask_missing_values,
                            feat_idx,
                            neighbor_feat_idx,
                            estimator=None,
                            fit_mode=True):
        """Impute a single feature from the others provided.

        This function predicts the missing values of one of the features using
        the current estimates of all the other features. The ``estimator`` must
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

        estimator : object
            The estimator to use at this step of the round-robin imputation.
            If ``sample_posterior`` is True, the estimator must support
            ``return_std`` in its ``predict`` method.
            If None, it will be cloned from self._estimator.

        fit_mode : boolean, default=True
            Whether to fit and predict with the estimator or just predict.

        Returns
        -------
        X_filled : ndarray
            Input data with ``X_filled[missing_row_mask, feat_idx]`` updated.

        estimator : estimator with sklearn API
            The fitted estimator used to impute
            ``X_filled[missing_row_mask, feat_idx]``.
        """

        # if nothing is missing, just return the default
        # (should not happen at fit time because feat_ids would be excluded)
        missing_row_mask = mask_missing_values[:, feat_idx]
        if not np.any(missing_row_mask):
            return X_filled, estimator

        if estimator is None and fit_mode is False:
            raise ValueError("If fit_mode is False, then an already-fitted "
                             "estimator should be passed in.")

        if estimator is None:
            estimator = clone(self._estimator)

        if fit_mode:
            X_train = safe_indexing(X_filled[:, neighbor_feat_idx],
                                    ~missing_row_mask)
            y_train = safe_indexing(X_filled[:, feat_idx],
                                    ~missing_row_mask)
            estimator.fit(X_train, y_train)

        # get posterior samples
        X_test = safe_indexing(X_filled[:, neighbor_feat_idx],
                               missing_row_mask)
        # ... other code
```
### 46 - sklearn/impute.py:

Start line: 702, End line: 740

```python
class IterativeImputer(BaseEstimator, TransformerMixin):

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
### 91 - sklearn/impute.py:

Start line: 52, End line: 66

```python
def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if is_scalar_nan(value_to_mask):
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

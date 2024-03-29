# scikit-learn__scikit-learn-13496

| **scikit-learn/scikit-learn** | `3aefc834dce72e850bff48689bea3c7dff5f3fad` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 2085 |
| **Any found context length** | 263 |
| **Avg pos** | 5.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 3 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/ensemble/iforest.py b/sklearn/ensemble/iforest.py
--- a/sklearn/ensemble/iforest.py
+++ b/sklearn/ensemble/iforest.py
@@ -120,6 +120,12 @@ class IsolationForest(BaseBagging, OutlierMixin):
     verbose : int, optional (default=0)
         Controls the verbosity of the tree building process.
 
+    warm_start : bool, optional (default=False)
+        When set to ``True``, reuse the solution of the previous call to fit
+        and add more estimators to the ensemble, otherwise, just fit a whole
+        new forest. See :term:`the Glossary <warm_start>`.
+
+        .. versionadded:: 0.21
 
     Attributes
     ----------
@@ -173,7 +179,8 @@ def __init__(self,
                  n_jobs=None,
                  behaviour='old',
                  random_state=None,
-                 verbose=0):
+                 verbose=0,
+                 warm_start=False):
         super().__init__(
             base_estimator=ExtraTreeRegressor(
                 max_features=1,
@@ -185,6 +192,7 @@ def __init__(self,
             n_estimators=n_estimators,
             max_samples=max_samples,
             max_features=max_features,
+            warm_start=warm_start,
             n_jobs=n_jobs,
             random_state=random_state,
             verbose=verbose)

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/ensemble/iforest.py | 123 | 123 | 3 | 1 | 2085
| sklearn/ensemble/iforest.py | 176 | 176 | 1 | 1 | 263
| sklearn/ensemble/iforest.py | 188 | 188 | 1 | 1 | 263


## Problem Statement

```
Expose warm_start in Isolation forest
It seems to me that `sklearn.ensemble.IsolationForest` supports incremental addition of new trees with the `warm_start` parameter of its parent class, `sklearn.ensemble.BaseBagging`.

Even though this parameter is not exposed in `__init__()` , it gets inherited from `BaseBagging` and one can use it by changing it to `True` after initialization. To make it work, you have to also increment `n_estimators` on every iteration. 

It took me a while to notice that it actually works, and I had to inspect the source code of both `IsolationForest` and `BaseBagging`. Also, it looks to me that the behavior is in-line with `sklearn.ensemble.BaseForest` that is behind e.g. `sklearn.ensemble.RandomForestClassifier`.

To make it more easier to use, I'd suggest to:
* expose `warm_start` in `IsolationForest.__init__()`, default `False`;
* document it in the same way as it is documented for `RandomForestClassifier`, i.e. say:
\`\`\`py
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
\`\`\`
* add a test to make sure it works properly;
* possibly also mention in the "IsolationForest example" documentation entry;


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/ensemble/iforest.py** | 167 | 203| 263 | 263 | 4271 | 
| 2 | 2 sklearn/ensemble/forest.py | 301 | 343| 445 | 708 | 22344 | 
| **-> 3 <-** | **2 sklearn/ensemble/iforest.py** | 29 | 165| 1377 | 2085 | 22344 | 
| 4 | **2 sklearn/ensemble/iforest.py** | 401 | 440| 343 | 2428 | 22344 | 
| 5 | 3 examples/ensemble/plot_isolation_forest.py | 1 | 72| 609 | 3037 | 22953 | 
| 6 | 4 sklearn/ensemble/bagging.py | 345 | 390| 383 | 3420 | 30937 | 
| 7 | **4 sklearn/ensemble/iforest.py** | 205 | 306| 857 | 4277 | 30937 | 
| 8 | 4 sklearn/ensemble/forest.py | 1508 | 1772| 336 | 4613 | 30937 | 
| 9 | 4 sklearn/ensemble/forest.py | 1732 | 1772| 314 | 4927 | 30937 | 
| 10 | 4 sklearn/ensemble/forest.py | 1297 | 1507| 2171 | 7098 | 30937 | 
| 11 | 4 sklearn/ensemble/forest.py | 993 | 1293| 334 | 7432 | 30937 | 
| 12 | 4 sklearn/ensemble/forest.py | 125 | 155| 189 | 7621 | 30937 | 
| 13 | 4 sklearn/ensemble/forest.py | 1 | 91| 669 | 8290 | 30937 | 
| 14 | 4 sklearn/ensemble/forest.py | 1554 | 1731| 1742 | 10032 | 30937 | 
| 15 | 4 sklearn/ensemble/bagging.py | 925 | 948| 153 | 10185 | 30937 | 
| 16 | 5 benchmarks/bench_isolation_forest.py | 54 | 161| 1032 | 11217 | 32405 | 
| 17 | 5 sklearn/ensemble/forest.py | 1253 | 1550| 322 | 11539 | 32405 | 
| 18 | 5 sklearn/ensemble/forest.py | 395 | 424| 176 | 11715 | 32405 | 
| 19 | 5 sklearn/ensemble/bagging.py | 554 | 583| 184 | 11899 | 32405 | 
| 20 | 5 sklearn/ensemble/forest.py | 469 | 515| 416 | 12315 | 32405 | 
| 21 | **5 sklearn/ensemble/iforest.py** | 6 | 26| 116 | 12431 | 32405 | 
| 22 | 5 benchmarks/bench_isolation_forest.py | 1 | 29| 213 | 12644 | 32405 | 
| 23 | 6 sklearn/ensemble/gradient_boosting.py | 1472 | 1545| 672 | 13316 | 53628 | 
| 24 | 6 sklearn/ensemble/forest.py | 751 | 992| 2531 | 15847 | 53628 | 
| 25 | 6 sklearn/ensemble/forest.py | 1907 | 1949| 329 | 16176 | 53628 | 
| 26 | 6 sklearn/ensemble/forest.py | 636 | 663| 167 | 16343 | 53628 | 
| 27 | 6 sklearn/ensemble/gradient_boosting.py | 1759 | 2007| 2533 | 18876 | 53628 | 
| 28 | 6 sklearn/ensemble/forest.py | 1039 | 1252| 2187 | 21063 | 53628 | 
| 29 | 6 sklearn/ensemble/gradient_boosting.py | 1368 | 1393| 296 | 21359 | 53628 | 
| 30 | 6 sklearn/ensemble/bagging.py | 431 | 553| 1185 | 22544 | 53628 | 
| 31 | 6 sklearn/ensemble/gradient_boosting.py | 2223 | 2466| 2515 | 25059 | 53628 | 
| 32 | 7 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 25831 | 54792 | 
| 33 | 8 sklearn/ensemble/__init__.py | 1 | 36| 289 | 26120 | 55081 | 
| 34 | 8 sklearn/ensemble/forest.py | 1775 | 1905| 1340 | 27460 | 55081 | 
| 35 | 8 sklearn/ensemble/gradient_boosting.py | 2009 | 2034| 307 | 27767 | 55081 | 
| 36 | 8 sklearn/ensemble/gradient_boosting.py | 1202 | 1254| 460 | 28227 | 55081 | 
| 37 | 8 sklearn/ensemble/gradient_boosting.py | 1 | 61| 371 | 28598 | 55081 | 
| 38 | 8 benchmarks/bench_isolation_forest.py | 44 | 53| 108 | 28706 | 55081 | 
| 39 | 8 sklearn/ensemble/gradient_boosting.py | 1639 | 1663| 259 | 28965 | 55081 | 
| 40 | 8 sklearn/ensemble/gradient_boosting.py | 1340 | 1353| 143 | 29108 | 55081 | 
| 41 | 8 sklearn/ensemble/bagging.py | 990 | 1020| 248 | 29356 | 55081 | 
| 42 | 8 sklearn/ensemble/bagging.py | 585 | 620| 295 | 29651 | 55081 | 
| 43 | 8 sklearn/ensemble/forest.py | 217 | 299| 749 | 30400 | 55081 | 
| 44 | **8 sklearn/ensemble/iforest.py** | 442 | 474| 232 | 30632 | 55081 | 
| 45 | 8 sklearn/ensemble/bagging.py | 813 | 923| 1112 | 31744 | 55081 | 
| 46 | 8 sklearn/ensemble/gradient_boosting.py | 2468 | 2491| 322 | 32066 | 55081 | 
| 47 | 9 examples/ensemble/plot_ensemble_oob.py | 1 | 90| 692 | 32758 | 55825 | 
| 48 | 10 examples/ensemble/plot_forest_importances_faces.py | 1 | 50| 326 | 33084 | 56151 | 
| 49 | **10 sklearn/ensemble/iforest.py** | 363 | 399| 322 | 33406 | 56151 | 
| 50 | 10 sklearn/ensemble/forest.py | 345 | 359| 151 | 33557 | 56151 | 
| 51 | 10 sklearn/ensemble/bagging.py | 1 | 57| 380 | 33937 | 56151 | 
| 52 | 11 examples/ensemble/plot_feature_transformation.py | 1 | 83| 740 | 34677 | 57326 | 
| 53 | 11 sklearn/ensemble/forest.py | 426 | 467| 378 | 35055 | 57326 | 
| 54 | 12 sklearn/utils/estimator_checks.py | 321 | 397| 714 | 35769 | 79223 | 
| 55 | 13 sklearn/decomposition/online_lda.py | 318 | 343| 240 | 36009 | 85783 | 
| 56 | 14 examples/ensemble/plot_forest_importances.py | 1 | 55| 373 | 36382 | 86156 | 
| 57 | 14 sklearn/ensemble/gradient_boosting.py | 1547 | 1637| 834 | 37216 | 86156 | 
| 58 | 14 sklearn/ensemble/gradient_boosting.py | 1256 | 1338| 804 | 38020 | 86156 | 
| 59 | 14 sklearn/ensemble/bagging.py | 184 | 216| 208 | 38228 | 86156 | 
| 60 | 14 sklearn/ensemble/forest.py | 361 | 392| 248 | 38476 | 86156 | 
| 61 | 15 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 39269 | 87401 | 
| 62 | 15 sklearn/ensemble/forest.py | 707 | 1035| 350 | 39619 | 87401 | 
| 63 | 16 sklearn/linear_model/stochastic_gradient.py | 183 | 242| 548 | 40167 | 101179 | 
| 64 | 17 sklearn/naive_bayes.py | 489 | 567| 758 | 40925 | 109621 | 
| 65 | 18 examples/ensemble/plot_forest_iris.py | 73 | 161| 887 | 41812 | 111094 | 
| 66 | 19 sklearn/tree/tree.py | 278 | 363| 800 | 42612 | 124633 | 
| 67 | 20 examples/ensemble/plot_random_forest_embedding.py | 85 | 106| 205 | 42817 | 125595 | 
| 68 | **20 sklearn/ensemble/iforest.py** | 308 | 329| 220 | 43037 | 125595 | 
| 69 | 20 sklearn/ensemble/gradient_boosting.py | 1063 | 1088| 330 | 43367 | 125595 | 
| 70 | 21 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 44063 | 126319 | 
| 71 | 21 sklearn/ensemble/bagging.py | 60 | 116| 433 | 44496 | 126319 | 
| 72 | 21 sklearn/tree/tree.py | 1162 | 1316| 1664 | 46160 | 126319 | 
| 73 | 21 sklearn/tree/tree.py | 125 | 201| 674 | 46834 | 126319 | 
| 74 | 22 sklearn/ensemble/weight_boosting.py | 292 | 386| 892 | 47726 | 135435 | 
| 75 | 22 sklearn/ensemble/forest.py | 94 | 122| 263 | 47989 | 135435 | 
| 76 | 22 sklearn/ensemble/weight_boosting.py | 1 | 46| 263 | 48252 | 135435 | 
| 77 | 22 sklearn/ensemble/gradient_boosting.py | 1166 | 1200| 327 | 48579 | 135435 | 
| 78 | 22 sklearn/ensemble/gradient_boosting.py | 1722 | 2219| 322 | 48901 | 135435 | 
| 79 | 23 sklearn/ensemble/voting_classifier.py | 35 | 123| 972 | 49873 | 138280 | 
| 80 | 23 sklearn/ensemble/bagging.py | 246 | 343| 783 | 50656 | 138280 | 
| 81 | 24 examples/ensemble/plot_adaboost_multiclass.py | 1 | 89| 748 | 51404 | 139294 | 
| 82 | 24 sklearn/ensemble/weight_boosting.py | 49 | 70| 126 | 51530 | 139294 | 
| 83 | 25 benchmarks/bench_tree.py | 64 | 125| 523 | 52053 | 140164 | 
| 84 | 26 benchmarks/bench_covertype.py | 99 | 109| 151 | 52204 | 142055 | 
| 85 | 27 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 52750 | 142629 | 
| 86 | 28 sklearn/model_selection/_validation.py | 1309 | 1327| 238 | 52988 | 155739 | 
| 87 | 28 sklearn/ensemble/gradient_boosting.py | 1395 | 1470| 694 | 53682 | 155739 | 
| 88 | **28 sklearn/ensemble/iforest.py** | 331 | 361| 266 | 53948 | 155739 | 
| 89 | 28 sklearn/linear_model/stochastic_gradient.py | 512 | 552| 372 | 54320 | 155739 | 
| 90 | 28 sklearn/ensemble/weight_boosting.py | 429 | 445| 182 | 54502 | 155739 | 
| 91 | 29 sklearn/setup.py | 1 | 80| 577 | 55079 | 156316 | 
| 92 | 29 sklearn/tree/tree.py | 203 | 276| 816 | 55895 | 156316 | 
| 93 | 29 examples/ensemble/plot_forest_iris.py | 1 | 71| 586 | 56481 | 156316 | 
| 94 | 29 sklearn/tree/tree.py | 1317 | 1342| 189 | 56670 | 156316 | 
| 95 | 29 sklearn/ensemble/forest.py | 1976 | 2009| 304 | 56974 | 156316 | 
| 96 | 29 sklearn/ensemble/weight_boosting.py | 871 | 953| 798 | 57772 | 156316 | 
| 97 | 29 sklearn/utils/estimator_checks.py | 714 | 764| 443 | 58215 | 156316 | 
| 98 | 29 sklearn/utils/estimator_checks.py | 813 | 849| 352 | 58567 | 156316 | 
| 99 | 29 sklearn/linear_model/stochastic_gradient.py | 429 | 463| 415 | 58982 | 156316 | 
| 100 | 29 sklearn/naive_bayes.py | 267 | 307| 324 | 59306 | 156316 | 
| 101 | 29 examples/ensemble/plot_random_forest_embedding.py | 1 | 84| 757 | 60063 | 156316 | 
| 102 | 29 sklearn/linear_model/stochastic_gradient.py | 1138 | 1170| 286 | 60349 | 156316 | 
| 103 | 29 sklearn/ensemble/gradient_boosting.py | 2036 | 2046| 124 | 60473 | 156316 | 
| 104 | 29 examples/ensemble/plot_adaboost_multiclass.py | 90 | 119| 243 | 60716 | 156316 | 
| 105 | 30 sklearn/manifold/t_sne.py | 629 | 652| 251 | 60967 | 164859 | 
| 106 | 30 sklearn/utils/estimator_checks.py | 1897 | 1941| 472 | 61439 | 164859 | 
| 107 | 31 sklearn/mixture/base.py | 519 | 537| 232 | 61671 | 168728 | 
| 108 | 31 sklearn/ensemble/weight_boosting.py | 1003 | 1093| 694 | 62365 | 168728 | 
| 109 | 31 sklearn/ensemble/bagging.py | 415 | 428| 134 | 62499 | 168728 | 
| 110 | 31 sklearn/ensemble/forest.py | 157 | 179| 228 | 62727 | 168728 | 
| 111 | 32 sklearn/impute.py | 936 | 971| 344 | 63071 | 179406 | 
| 112 | 32 sklearn/tree/tree.py | 1345 | 1485| 1416 | 64487 | 179406 | 
| 113 | 32 sklearn/impute.py | 661 | 700| 499 | 64986 | 179406 | 
| 114 | 33 examples/linear_model/plot_sgd_early_stopping.py | 1 | 55| 462 | 65448 | 180703 | 
| 115 | 33 sklearn/utils/estimator_checks.py | 2427 | 2471| 423 | 65871 | 180703 | 
| 116 | 34 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 66664 | 182549 | 
| 117 | 34 sklearn/ensemble/weight_boosting.py | 492 | 548| 534 | 67198 | 182549 | 
| 118 | 34 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 340 | 67538 | 182549 | 
| 119 | 34 examples/ensemble/plot_gradient_boosting_early_stopping.py | 105 | 159| 344 | 67882 | 182549 | 
| 120 | 34 sklearn/ensemble/gradient_boosting.py | 868 | 900| 368 | 68250 | 182549 | 
| 121 | 34 sklearn/ensemble/forest.py | 2011 | 2028| 133 | 68383 | 182549 | 
| 122 | 34 sklearn/linear_model/stochastic_gradient.py | 465 | 510| 411 | 68794 | 182549 | 
| 123 | 35 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 84| 741 | 69535 | 183674 | 
| 124 | 35 sklearn/ensemble/forest.py | 1951 | 1974| 203 | 69738 | 183674 | 
| 125 | 36 sklearn/ensemble/base.py | 101 | 118| 176 | 69914 | 184831 | 
| 126 | 36 sklearn/linear_model/stochastic_gradient.py | 919 | 939| 306 | 70220 | 184831 | 
| 127 | 37 sklearn/neighbors/base.py | 162 | 247| 761 | 70981 | 192555 | 
| 128 | **37 sklearn/ensemble/iforest.py** | 477 | 511| 309 | 71290 | 192555 | 
| 129 | 38 sklearn/__init__.py | 1 | 68| 524 | 71814 | 193486 | 
| 130 | 38 sklearn/ensemble/weight_boosting.py | 550 | 596| 397 | 72211 | 193486 | 
| 131 | 38 sklearn/linear_model/stochastic_gradient.py | 1237 | 1528| 678 | 72889 | 193486 | 
| 132 | 38 sklearn/linear_model/stochastic_gradient.py | 292 | 321| 250 | 73139 | 193486 | 
| 133 | 38 sklearn/tree/tree.py | 535 | 746| 2217 | 75356 | 193486 | 
| 134 | 39 sklearn/neighbors/nca.py | 370 | 447| 549 | 75905 | 197717 | 
| 135 | 40 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 90| 749 | 76654 | 198683 | 
| 136 | 40 sklearn/utils/estimator_checks.py | 2036 | 2066| 277 | 76931 | 198683 | 
| 137 | 40 sklearn/ensemble/gradient_boosting.py | 694 | 705| 194 | 77125 | 198683 | 


### Hint

```
+1 to expose `warm_start` in `IsolationForest`, unless there was a good reason for not doing so in the first place. I could not find any related discussion in the IsolationForest PR #4163. ping @ngoix @agramfort?
no objection

>

PR welcome @petibear. Feel
free to ping me when it’s ready for reviews :).
OK, I'm working on it then. 
Happy to learn the process (of contributing) here. 
```

## Patch

```diff
diff --git a/sklearn/ensemble/iforest.py b/sklearn/ensemble/iforest.py
--- a/sklearn/ensemble/iforest.py
+++ b/sklearn/ensemble/iforest.py
@@ -120,6 +120,12 @@ class IsolationForest(BaseBagging, OutlierMixin):
     verbose : int, optional (default=0)
         Controls the verbosity of the tree building process.
 
+    warm_start : bool, optional (default=False)
+        When set to ``True``, reuse the solution of the previous call to fit
+        and add more estimators to the ensemble, otherwise, just fit a whole
+        new forest. See :term:`the Glossary <warm_start>`.
+
+        .. versionadded:: 0.21
 
     Attributes
     ----------
@@ -173,7 +179,8 @@ def __init__(self,
                  n_jobs=None,
                  behaviour='old',
                  random_state=None,
-                 verbose=0):
+                 verbose=0,
+                 warm_start=False):
         super().__init__(
             base_estimator=ExtraTreeRegressor(
                 max_features=1,
@@ -185,6 +192,7 @@ def __init__(self,
             n_estimators=n_estimators,
             max_samples=max_samples,
             max_features=max_features,
+            warm_start=warm_start,
             n_jobs=n_jobs,
             random_state=random_state,
             verbose=verbose)

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_iforest.py b/sklearn/ensemble/tests/test_iforest.py
--- a/sklearn/ensemble/tests/test_iforest.py
+++ b/sklearn/ensemble/tests/test_iforest.py
@@ -295,6 +295,28 @@ def test_score_samples():
                        clf2.score_samples([[2., 2.]]))
 
 
+@pytest.mark.filterwarnings('ignore:default contamination')
+@pytest.mark.filterwarnings('ignore:behaviour="old"')
+def test_iforest_warm_start():
+    """Test iterative addition of iTrees to an iForest """
+
+    rng = check_random_state(0)
+    X = rng.randn(20, 2)
+
+    # fit first 10 trees
+    clf = IsolationForest(n_estimators=10, max_samples=20,
+                          random_state=rng, warm_start=True)
+    clf.fit(X)
+    # remember the 1st tree
+    tree_1 = clf.estimators_[0]
+    # fit another 10 trees
+    clf.set_params(n_estimators=20)
+    clf.fit(X)
+    # expecting 20 fitted trees and no overwritten trees
+    assert len(clf.estimators_) == 20
+    assert clf.estimators_[0] is tree_1
+
+
 @pytest.mark.filterwarnings('ignore:default contamination')
 @pytest.mark.filterwarnings('ignore:behaviour="old"')
 def test_deprecation():

```


## Code snippets

### 1 - sklearn/ensemble/iforest.py:

Start line: 167, End line: 203

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination="legacy",
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=None,
                 behaviour='old',
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator=ExtraTreeRegressor(
                max_features=1,
                splitter='random',
                random_state=random_state),
            # here above max_features has no links with self.max_features
            bootstrap=bootstrap,
            bootstrap_features=False,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.behaviour = behaviour
        self.contamination = contamination

    def _set_oob_score(self, X, y):
        raise NotImplementedError("OOB score not supported by iforest")

    def _parallel_args(self):
        # ExtraTreeRegressor releases the GIL, so it's more efficient to use
        # a thread-based backend rather than a process-based backend so as
        # to avoid suffering from communication overhead and extra memory
        # copies.
        return _joblib_parallel_args(prefer='threads')
```
### 2 - sklearn/ensemble/forest.py:

Start line: 301, End line: 343

```python
class BaseForest(BaseEnsemble, MultiOutputMixin, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None):
        # ... other code

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
```
### 3 - sklearn/ensemble/iforest.py:

Start line: 29, End line: 165

```python
class IsolationForest(BaseBagging, OutlierMixin):
    """Isolation Forest Algorithm

    Return the anomaly score of each sample using the IsolationForest algorithm

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a
    measure of normality and our decision function.

    Random partitioning produces noticeably shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for particular samples, they are highly likely to be anomalies.

    Read more in the :ref:`User Guide <isolation_forest>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default="auto")
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.
            - If "auto", then `max_samples=min(256, n_samples)`.

        If max_samples is larger than the number of samples provided,
        all samples will be used for all trees (no sampling).

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the decision function. If 'auto', the decision function threshold is
        determined as in the original paper.

        .. versionchanged:: 0.20
           The default value of ``contamination`` will change from 0.1 in 0.20
           to ``'auto'`` in 0.22.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=False)
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    behaviour : str, default='old'
        Behaviour of the ``decision_function`` which can be either 'old' or
        'new'. Passing ``behaviour='new'`` makes the ``decision_function``
        change to match other anomaly detection algorithm API which will be
        the default behaviour in the future. As explained in details in the
        ``offset_`` attribute documentation, the ``decision_function`` becomes
        dependent on the contamination parameter, in such a way that 0 becomes
        its natural threshold to detect outliers.

        .. versionadded:: 0.20
           ``behaviour`` is added in 0.20 for back-compatibility purpose.

        .. deprecated:: 0.20
           ``behaviour='old'`` is deprecated in 0.20 and will not be possible
           in 0.22.

        .. deprecated:: 0.22
           ``behaviour`` parameter will be deprecated in 0.22 and removed in
           0.24.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    max_samples_ : integer
        The actual number of samples

    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: ``decision_function = score_samples - offset_``.
        Assuming behaviour == 'new', ``offset_`` is defined as follows.
        When the contamination parameter is set to "auto", the offset is equal
        to -0.5 as the scores of inliers are close to 0 and the scores of
        outliers are close to -1. When a contamination parameter different
        than "auto" is provided, the offset is defined in such a way we obtain
        the expected number of outliers (samples with decision function < 0)
        in training.
        Assuming the behaviour parameter is set to 'old', we always have
        ``offset_ = -0.5``, making the decision function independent from the
        contamination parameter.

    Notes
    -----
    The implementation is based on an ensemble of ExtraTreeRegressor. The
    maximum depth of each tree is set to ``ceil(log_2(n))`` where
    :math:`n` is the number of samples used to build the tree
    (see (Liu et al., 2008) for more details).

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    """
```
### 4 - sklearn/ensemble/iforest.py:

Start line: 401, End line: 440

```python
class IsolationForest(BaseBagging, OutlierMixin):

    @property
    def threshold_(self):
        if self.behaviour != 'old':
            raise AttributeError("threshold_ attribute does not exist when "
                                 "behaviour != 'old'")
        warn("threshold_ attribute is deprecated in 0.20 and will"
             " be removed in 0.22.", DeprecationWarning)
        return self._threshold_

    def _compute_chunked_score_samples(self, X):

        n_samples = _num_samples(X)

        if self._max_features == X.shape[1]:
            subsample_features = False
        else:
            subsample_features = True

        # We get as many rows as possible within our working_memory budget
        # (defined by sklearn.get_config()['working_memory']) to store
        # self._max_features in each row during computation.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of score will
        #    exceed working_memory.
        #  - this does only account for temporary memory usage while loading
        #    the data needed to compute the scores -- the returned scores
        #    themselves are 1D.

        chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features,
                                        max_n_rows=n_samples)
        slices = gen_batches(n_samples, chunk_n_rows)

        scores = np.zeros(n_samples, order="f")

        for sl in slices:
            # compute score on the slices of test samples:
            scores[sl] = self._compute_score_samples(X[sl], subsample_features)

        return scores
```
### 5 - examples/ensemble/plot_isolation_forest.py:

Start line: 1, End line: 72

```python
"""
==========================================
IsolationForest example
==========================================

An example using :class:`sklearn.ensemble.IsolationForest` for anomaly
detection.

The IsolationForest 'isolates' observations by randomly selecting a feature
and then randomly selecting a split value between the maximum and minimum
values of the selected feature.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a measure
of normality and our decision function.

Random partitioning produces noticeable shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path lengths
for particular samples, they are highly likely to be anomalies.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
```
### 6 - sklearn/ensemble/bagging.py:

Start line: 345, End line: 390

```python
class BaseBagging(BaseEnsemble, metaclass=ABCMeta):

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        # ... other code

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               **self._parallel_args())(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
```
### 7 - sklearn/ensemble/iforest.py:

Start line: 205, End line: 306

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def fit(self, X, y=None, sample_weight=None):
        """Fit estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        y : Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self : object
        """
        if self.contamination == "legacy":
            warn('default contamination parameter 0.1 will change '
                 'in version 0.22 to "auto". This will change the '
                 'predict method behavior.',
                 FutureWarning)
            self._contamination = 0.1
        else:
            self._contamination = self.contamination

        if self.behaviour == 'old':
            warn('behaviour="old" is deprecated and will be removed '
                 'in version 0.22. Please use behaviour="new", which '
                 'makes the decision_function change to match '
                 'other anomaly detection algorithm API.',
                 FutureWarning)

        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        rnd = check_random_state(self.random_state)
        y = rnd.uniform(size=X.shape[0])

        # ensure that max_sample is in [1, n_samples]:
        n_samples = X.shape[0]

        if isinstance(self.max_samples, str):
            if self.max_samples == 'auto':
                max_samples = min(256, n_samples)
            else:
                raise ValueError('max_samples (%s) is not supported.'
                                 'Valid choices are: "auto", int or'
                                 'float' % self.max_samples)

        elif isinstance(self.max_samples, INTEGER_TYPES):
            if self.max_samples > n_samples:
                warn("max_samples (%s) is greater than the "
                     "total number of samples (%s). max_samples "
                     "will be set to n_samples for estimation."
                     % (self.max_samples, n_samples))
                max_samples = n_samples
            else:
                max_samples = self.max_samples
        else:  # float
            if not (0. < self.max_samples <= 1.):
                raise ValueError("max_samples must be in (0, 1], got %r"
                                 % self.max_samples)
            max_samples = int(self.max_samples * X.shape[0])

        self.max_samples_ = max_samples
        max_depth = int(np.ceil(np.log2(max(max_samples, 2))))
        super()._fit(X, y, max_samples,
                     max_depth=max_depth,
                     sample_weight=sample_weight)

        if self.behaviour == 'old':
            # in this case, decision_function = 0.5 + self.score_samples(X):
            if self._contamination == "auto":
                raise ValueError("contamination parameter cannot be set to "
                                 "'auto' when behaviour == 'old'.")

            self.offset_ = -0.5
            self._threshold_ = np.percentile(self.decision_function(X),
                                             100. * self._contamination)

            return self

        # else, self.behaviour == 'new':
        if self._contamination == "auto":
            # 0.5 plays a special role as described in the original paper.
            # we take the opposite as we consider the opposite of their score.
            self.offset_ = -0.5
            return self

        # else, define offset_ wrt contamination parameter, so that the
        # threshold_ attribute is implicitly 0 and is not needed anymore:
        self.offset_ = np.percentile(self.score_samples(X),
                                     100. * self._contamination)

        return self
```
### 8 - sklearn/ensemble/forest.py:

Start line: 1508, End line: 1772

```python
class ExtraTreesClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split


class ExtraTreesRegressor(ForestRegressor):
```
### 9 - sklearn/ensemble/forest.py:

Start line: 1732, End line: 1772

```python
class ExtraTreesRegressor(ForestRegressor):
    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
```
### 10 - sklearn/ensemble/forest.py:

Start line: 1297, End line: 1507

```python
class ExtraTreesClassifier(ForestClassifier):
    """An extra-trees classifier.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and uses averaging to improve the predictive accuracy
    and control over-fitting.

    Read more in the :ref:`User Guide <forest>`.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

        .. versionchanged:: 0.20
           The default value of ``n_estimators`` will change from 10 in
           version 0.20 to 100 in version 0.22.

    criterion : string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, (default=1e-7)
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
           will be removed in 0.25. Use ``min_impurity_decrease`` instead.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that weights are
        computed based on the bootstrap sample for every tree grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    See also
    --------
    sklearn.tree.ExtraTreeClassifier : Base classifier for this ensemble.
    RandomForestClassifier : Ensemble Classifier based on trees with optimal
        splits.
    """
```
### 21 - sklearn/ensemble/iforest.py:

Start line: 6, End line: 26

```python
import numbers
import numpy as np
from scipy.sparse import issparse
from warnings import warn

from ..tree import ExtraTreeRegressor
from ..utils import (
    check_random_state,
    check_array,
    gen_batches,
    get_chunk_n_rows,
)
from ..utils.fixes import _joblib_parallel_args
from ..utils.validation import check_is_fitted, _num_samples
from ..base import OutlierMixin

from .bagging import BaseBagging

__all__ = ["IsolationForest"]

INTEGER_TYPES = (numbers.Integral, np.integer)
```
### 44 - sklearn/ensemble/iforest.py:

Start line: 442, End line: 474

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def _compute_score_samples(self, X, subsample_features):
        """Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix

        subsample_features : bool,
            whether features should be subsampled
        """
        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")

        for tree, features in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, features] if subsample_features else X

            leaves_index = tree.apply(X_subset)
            node_indicator = tree.decision_path(X_subset)
            n_samples_leaf = tree.tree_.n_node_samples[leaves_index]

            depths += (
                np.ravel(node_indicator.sum(axis=1))
                + _average_path_length(n_samples_leaf)
                - 1.0
            )

        scores = 2 ** (
            -depths
            / (len(self.estimators_)
               * _average_path_length([self.max_samples_]))
        )
        return scores
```
### 49 - sklearn/ensemble/iforest.py:

Start line: 363, End line: 399

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def score_samples(self, X):
        """Opposite of the anomaly score defined in the original paper.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : array, shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        # code structure from ForestClassifier/predict_proba
        check_is_fitted(self, ["estimators_"])

        # Check data
        X = check_array(X, accept_sparse='csr')
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Take the opposite of the scores as bigger is better (here less
        # abnormal)
        return -self._compute_chunked_score_samples(X)
```
### 68 - sklearn/ensemble/iforest.py:

Start line: 308, End line: 329

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        is_inlier : array, shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        check_is_fitted(self, ["offset_"])
        X = check_array(X, accept_sparse='csr')
        is_inlier = np.ones(X.shape[0], dtype=int)
        threshold = self.threshold_ if self.behaviour == 'old' else 0
        is_inlier[self.decision_function(X) < threshold] = -1
        return is_inlier
```
### 88 - sklearn/ensemble/iforest.py:

Start line: 331, End line: 361

```python
class IsolationForest(BaseBagging, OutlierMixin):

    def decision_function(self, X):
        """Average anomaly score of X of the base classifiers.

        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.

        The measure of normality of an observation given a tree is the depth
        of the leaf containing this observation, which is equivalent to
        the number of splittings required to isolate this point. In case of
        several observations n_left in the leaf, the average path length of
        a n_left samples isolation tree is added.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        scores : array, shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal. Negative scores represent outliers,
            positive scores represent inliers.

        """
        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        return self.score_samples(X) - self.offset_
```
### 128 - sklearn/ensemble/iforest.py:

Start line: 477, End line: 511

```python
def _average_path_length(n_samples_leaf):
    """The average path length in a n_samples iTree, which is equal to
    the average path length of an unsuccessful BST search since the
    latter has the same structure as an isolation tree.
    Parameters
    ----------
    n_samples_leaf : array-like, shape (n_samples,).
        The number of training samples in each test sample leaf, for
        each estimators.

    Returns
    -------
    average_path_length : array, same shape as n_samples_leaf

    """

    n_samples_leaf = check_array(n_samples_leaf, ensure_2d=False)

    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.
    average_path_length[mask_2] = 1.
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)
```

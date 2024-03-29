# scikit-learn__scikit-learn-13472

| **scikit-learn/scikit-learn** | `3b35104c93cb53f67fb5f52ae2fece76ef7144da` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 672 |
| **Any found context length** | 672 |
| **Avg pos** | 1.0 |
| **Min pos** | 1 |
| **Max pos** | 1 |
| **Top file pos** | 1 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/ensemble/gradient_boosting.py b/sklearn/ensemble/gradient_boosting.py
--- a/sklearn/ensemble/gradient_boosting.py
+++ b/sklearn/ensemble/gradient_boosting.py
@@ -1476,20 +1476,25 @@ def fit(self, X, y, sample_weight=None, monitor=None):
                 raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                            dtype=np.float64)
             else:
-                try:
-                    self.init_.fit(X, y, sample_weight=sample_weight)
-                except TypeError:
-                    if sample_weight_is_none:
-                        self.init_.fit(X, y)
-                    else:
-                        raise ValueError(
-                            "The initial estimator {} does not support sample "
-                            "weights.".format(self.init_.__class__.__name__))
+                # XXX clean this once we have a support_sample_weight tag
+                if sample_weight_is_none:
+                    self.init_.fit(X, y)
+                else:
+                    msg = ("The initial estimator {} does not support sample "
+                           "weights.".format(self.init_.__class__.__name__))
+                    try:
+                        self.init_.fit(X, y, sample_weight=sample_weight)
+                    except TypeError:  # regular estimator without SW support
+                        raise ValueError(msg)
+                    except ValueError as e:
+                        if 'not enough values to unpack' in str(e):  # pipeline
+                            raise ValueError(msg) from e
+                        else:  # regular estimator whose input checking failed
+                            raise
 
                 raw_predictions = \
                     self.loss_.get_init_raw_predictions(X, self.init_)
 
-
             begin_at_stage = 0
 
             # The rng state must be preserved if warm_start is True

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/ensemble/gradient_boosting.py | 1479 | 1492 | 1 | 1 | 672


## Problem Statement

```
GradientBoostingRegressor initial estimator does not play together with Pipeline
Using a pipeline as the initial estimator of GradientBoostingRegressor doesn't work due to incompatible signatures.

\`\`\`python
import sklearn
import sklearn.pipeline
import sklearn.ensemble
import sklearn.decomposition
import sklearn.linear_model
import numpy as np
init = sklearn.pipeline.make_pipeline(sklearn.decomposition.PCA(), sklearn.linear_model.ElasticNet())
model = sklearn.ensemble.GradientBoostingRegressor(init=init)
x = np.random.rand(12, 3)
y = np.random.rand(12)
model.fit(x, y)

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/Thomas/.local/miniconda3/envs/4cast/lib/python3.6/site-packages/sklearn/ensemble/gradient_boosting.py", line 1421, in fit
    self.init_.fit(X, y, sample_weight)
TypeError: fit() takes from 2 to 3 positional arguments but 4 were given
\`\`\`
The signature of `Pipeline.fit` is

\`\`\`python
# sklearn/pipeline.py
...
239 def fit(self, X, y=None, **fit_params):
...
\`\`\`
which cannot be called with three positional arguments as above.

So I guess line 1421 in `sklearn/ensemble/gradient_boosting.py` should read
`self.init_.fit(X, y, sample_weight=sample_weight)` instead and the issue is solved. Right?

#### Versions
\`\`\`python
>>> sklearn.show_versions()

System:
    python: 3.6.2 |Continuum Analytics, Inc.| (default, Jul 20 2017, 13:14:59)  [GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]
executable: /Users/Thomas/.local/miniconda3/envs/test/bin/python
   machine: Darwin-18.2.0-x86_64-i386-64bit

BLAS:
    macros: SCIPY_MKL_H=None, HAVE_CBLAS=None
  lib_dirs: /Users/Thomas/.local/miniconda3/envs/test/lib
cblas_libs: mkl_rt, pthread

Python deps:
       pip: 10.0.1
setuptools: 39.2.0
   sklearn: 0.20.2
     numpy: 1.16.1
     scipy: 1.2.0
    Cython: None
    pandas: 0.24.2
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/ensemble/gradient_boosting.py** | 1470 | 1543| 672 | 672 | 21197 | 
| 2 | 2 sklearn/utils/estimator_checks.py | 1048 | 1074| 262 | 934 | 43094 | 
| 3 | 3 sklearn/pipeline.py | 162 | 188| 248 | 1182 | 50362 | 
| 4 | 3 sklearn/pipeline.py | 244 | 288| 412 | 1594 | 50362 | 
| 5 | **3 sklearn/ensemble/gradient_boosting.py** | 2466 | 2489| 322 | 1916 | 50362 | 
| 6 | **3 sklearn/ensemble/gradient_boosting.py** | 1637 | 1661| 259 | 2175 | 50362 | 
| 7 | 3 sklearn/pipeline.py | 29 | 124| 1039 | 3214 | 50362 | 
| 8 | **3 sklearn/ensemble/gradient_boosting.py** | 1395 | 1468| 674 | 3888 | 50362 | 
| 9 | **3 sklearn/ensemble/gradient_boosting.py** | 1256 | 1338| 804 | 4692 | 50362 | 
| 10 | 4 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 5464 | 51526 | 
| 11 | **4 sklearn/ensemble/gradient_boosting.py** | 2221 | 2464| 2515 | 7979 | 51526 | 
| 12 | 5 examples/ensemble/plot_gradient_boosting_regression.py | 1 | 77| 546 | 8525 | 52100 | 
| 13 | **5 sklearn/ensemble/gradient_boosting.py** | 2034 | 2044| 124 | 8649 | 52100 | 
| 14 | **5 sklearn/ensemble/gradient_boosting.py** | 2007 | 2032| 307 | 8956 | 52100 | 
| 15 | **5 sklearn/ensemble/gradient_boosting.py** | 1 | 61| 371 | 9327 | 52100 | 
| 16 | 6 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 10120 | 53345 | 
| 17 | 7 examples/compose/plot_compare_reduction.py | 1 | 105| 826 | 10946 | 54426 | 
| 18 | 8 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 11642 | 55150 | 
| 19 | **8 sklearn/ensemble/gradient_boosting.py** | 1202 | 1254| 460 | 12102 | 55150 | 
| 20 | 9 examples/compose/plot_digits_pipe.py | 1 | 79| 572 | 12674 | 55730 | 
| 21 | **9 sklearn/ensemble/gradient_boosting.py** | 1340 | 1353| 143 | 12817 | 55730 | 
| 22 | **9 sklearn/ensemble/gradient_boosting.py** | 1757 | 2005| 2527 | 15344 | 55730 | 
| 23 | 9 sklearn/pipeline.py | 531 | 568| 278 | 15622 | 55730 | 
| 24 | 9 sklearn/pipeline.py | 1 | 26| 128 | 15750 | 55730 | 
| 25 | 10 sklearn/ensemble/weight_boosting.py | 72 | 90| 143 | 15893 | 64846 | 
| 26 | 10 sklearn/pipeline.py | 452 | 469| 134 | 16027 | 64846 | 
| 27 | 10 sklearn/pipeline.py | 290 | 319| 219 | 16246 | 64846 | 
| 28 | 10 sklearn/pipeline.py | 645 | 668| 219 | 16465 | 64846 | 
| 29 | 11 examples/ensemble/plot_adaboost_regression.py | 1 | 55| 389 | 16854 | 65263 | 
| 30 | 11 sklearn/pipeline.py | 321 | 355| 282 | 17136 | 65263 | 
| 31 | 11 sklearn/pipeline.py | 357 | 382| 207 | 17343 | 65263 | 
| 32 | 12 sklearn/ensemble/bagging.py | 925 | 948| 153 | 17496 | 73247 | 
| 33 | 12 sklearn/ensemble/weight_boosting.py | 954 | 1001| 349 | 17845 | 73247 | 
| 34 | 13 examples/ensemble/plot_feature_transformation.py | 1 | 83| 740 | 18585 | 74422 | 
| 35 | 13 sklearn/utils/estimator_checks.py | 1077 | 1102| 282 | 18867 | 74422 | 
| 36 | 14 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 19660 | 76268 | 
| 37 | **14 sklearn/ensemble/gradient_boosting.py** | 1063 | 1088| 330 | 19990 | 76268 | 
| 38 | 15 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 84| 741 | 20731 | 77393 | 
| 39 | 15 sklearn/ensemble/bagging.py | 990 | 1020| 248 | 20979 | 77393 | 
| 40 | 16 sklearn/linear_model/ransac.py | 425 | 454| 346 | 21325 | 81458 | 
| 41 | **16 sklearn/ensemble/gradient_boosting.py** | 1355 | 1366| 128 | 21453 | 81458 | 
| 42 | 16 sklearn/pipeline.py | 433 | 450| 131 | 21584 | 81458 | 
| 43 | 16 sklearn/ensemble/bagging.py | 345 | 390| 383 | 21967 | 81458 | 
| 44 | 16 sklearn/linear_model/ransac.py | 327 | 424| 809 | 22776 | 81458 | 
| 45 | 16 sklearn/pipeline.py | 596 | 642| 390 | 23166 | 81458 | 
| 46 | 16 sklearn/pipeline.py | 414 | 431| 131 | 23297 | 81458 | 
| 47 | 16 sklearn/ensemble/weight_boosting.py | 871 | 953| 798 | 24095 | 81458 | 
| 48 | 16 sklearn/ensemble/weight_boosting.py | 1095 | 1111| 180 | 24275 | 81458 | 
| 49 | 17 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 340 | 24615 | 95234 | 
| 50 | 17 sklearn/ensemble/weight_boosting.py | 1003 | 1093| 694 | 25309 | 95234 | 
| 51 | 17 examples/ensemble/plot_gradient_boosting_oob.py | 96 | 138| 423 | 25732 | 95234 | 
| 52 | 17 sklearn/ensemble/weight_boosting.py | 1 | 46| 263 | 25995 | 95234 | 
| 53 | 17 sklearn/ensemble/weight_boosting.py | 429 | 445| 182 | 26177 | 95234 | 
| 54 | **17 sklearn/ensemble/gradient_boosting.py** | 1368 | 1393| 296 | 26473 | 95234 | 
| 55 | 17 sklearn/pipeline.py | 471 | 498| 190 | 26663 | 95234 | 
| 56 | 18 examples/compose/plot_column_transformer_mixed_types.py | 1 | 104| 799 | 27462 | 96055 | 
| 57 | 18 sklearn/utils/estimator_checks.py | 1 | 62| 466 | 27928 | 96055 | 
| 58 | 18 sklearn/linear_model/stochastic_gradient.py | 1237 | 1528| 678 | 28606 | 96055 | 
| 59 | 18 sklearn/linear_model/stochastic_gradient.py | 1079 | 1105| 296 | 28902 | 96055 | 
| 60 | 18 sklearn/ensemble/bagging.py | 813 | 923| 1112 | 30014 | 96055 | 
| 61 | 19 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 90| 749 | 30763 | 97021 | 
| 62 | **19 sklearn/ensemble/gradient_boosting.py** | 694 | 705| 194 | 30957 | 97021 | 
| 63 | 20 benchmarks/bench_mnist.py | 84 | 105| 314 | 31271 | 98739 | 
| 64 | 20 sklearn/ensemble/bagging.py | 60 | 116| 433 | 31704 | 98739 | 
| 65 | **20 sklearn/ensemble/gradient_boosting.py** | 1545 | 1635| 834 | 32538 | 98739 | 
| 66 | 20 sklearn/pipeline.py | 384 | 412| 242 | 32780 | 98739 | 
| 67 | 20 sklearn/linear_model/stochastic_gradient.py | 1138 | 1170| 286 | 33066 | 98739 | 
| 68 | 20 sklearn/ensemble/bagging.py | 554 | 583| 184 | 33250 | 98739 | 
| 69 | 21 sklearn/gaussian_process/gpr.py | 250 | 262| 175 | 33425 | 103196 | 
| 70 | 21 sklearn/pipeline.py | 126 | 160| 195 | 33620 | 103196 | 
| 71 | 22 sklearn/utils/mocking.py | 51 | 70| 140 | 33760 | 104105 | 
| 72 | **22 sklearn/ensemble/gradient_boosting.py** | 1166 | 1200| 327 | 34087 | 104105 | 
| 73 | 22 sklearn/ensemble/weight_boosting.py | 92 | 173| 602 | 34689 | 104105 | 
| 74 | **22 sklearn/ensemble/gradient_boosting.py** | 593 | 599| 164 | 34853 | 104105 | 
| 75 | 22 sklearn/ensemble/bagging.py | 585 | 620| 295 | 35148 | 104105 | 
| 76 | 22 examples/impute/plot_iterative_imputer_variants_comparison.py | 85 | 131| 384 | 35532 | 104105 | 
| 77 | 22 sklearn/pipeline.py | 500 | 529| 225 | 35757 | 104105 | 
| 78 | 23 examples/ensemble/plot_bias_variance.py | 116 | 192| 684 | 36441 | 105919 | 
| 79 | **23 sklearn/ensemble/gradient_boosting.py** | 168 | 217| 351 | 36792 | 105919 | 
| 80 | 24 examples/ensemble/plot_adaboost_multiclass.py | 1 | 89| 748 | 37540 | 106933 | 
| 81 | **24 sklearn/ensemble/gradient_boosting.py** | 1720 | 2217| 322 | 37862 | 106933 | 
| 82 | 24 examples/compose/plot_compare_reduction.py | 106 | 132| 236 | 38098 | 106933 | 
| 83 | 25 examples/linear_model/plot_robust_fit.py | 1 | 73| 509 | 38607 | 107727 | 
| 84 | 25 sklearn/pipeline.py | 208 | 242| 284 | 38891 | 107727 | 
| 85 | 25 sklearn/linear_model/stochastic_gradient.py | 292 | 321| 250 | 39141 | 107727 | 
| 86 | **25 sklearn/ensemble/gradient_boosting.py** | 2531 | 2553| 195 | 39336 | 107727 | 
| 87 | 26 examples/preprocessing/plot_function_transformer.py | 1 | 73| 450 | 39786 | 108177 | 
| 88 | 26 sklearn/utils/estimator_checks.py | 1828 | 1870| 469 | 40255 | 108177 | 
| 89 | 26 sklearn/ensemble/weight_boosting.py | 387 | 427| 300 | 40555 | 108177 | 
| 90 | 26 sklearn/utils/estimator_checks.py | 1151 | 1219| 601 | 41156 | 108177 | 
| 91 | 26 examples/ensemble/plot_gradient_boosting_early_stopping.py | 105 | 159| 344 | 41500 | 108177 | 
| 92 | 27 examples/model_selection/plot_grid_search_refit_callable.py | 78 | 117| 323 | 41823 | 109009 | 
| 93 | 27 sklearn/linear_model/ransac.py | 210 | 228| 201 | 42024 | 109009 | 
| 94 | 27 sklearn/ensemble/bagging.py | 246 | 343| 783 | 42807 | 109009 | 
| 95 | 27 sklearn/linear_model/stochastic_gradient.py | 1514 | 1529| 245 | 43052 | 109009 | 
| 96 | 27 sklearn/pipeline.py | 571 | 593| 155 | 43207 | 109009 | 
| 97 | **27 sklearn/ensemble/gradient_boosting.py** | 220 | 269| 369 | 43576 | 109009 | 
| 98 | 27 sklearn/ensemble/weight_boosting.py | 292 | 386| 892 | 44468 | 109009 | 
| 99 | 27 sklearn/linear_model/stochastic_gradient.py | 394 | 426| 341 | 44809 | 109009 | 
| 100 | 27 examples/ensemble/plot_adaboost_multiclass.py | 90 | 119| 243 | 45052 | 109009 | 
| 101 | 27 sklearn/utils/estimator_checks.py | 975 | 1045| 628 | 45680 | 109009 | 
| 102 | 27 sklearn/linear_model/stochastic_gradient.py | 1052 | 1077| 320 | 46000 | 109009 | 
| 103 | 27 sklearn/pipeline.py | 822 | 854| 274 | 46274 | 109009 | 
| 104 | **27 sklearn/ensemble/gradient_boosting.py** | 125 | 165| 267 | 46541 | 109009 | 
| 105 | 27 sklearn/ensemble/bagging.py | 431 | 553| 1185 | 47726 | 109009 | 
| 106 | 28 examples/model_selection/grid_search_text_feature_extraction.py | 2 | 94| 661 | 48387 | 110104 | 
| 107 | 29 benchmarks/bench_covertype.py | 99 | 109| 151 | 48538 | 111995 | 
| 108 | 30 examples/impute/plot_missing_values.py | 1 | 46| 371 | 48909 | 113199 | 
| 109 | **30 sklearn/ensemble/gradient_boosting.py** | 1028 | 1061| 272 | 49181 | 113199 | 
| 110 | 30 sklearn/utils/estimator_checks.py | 1800 | 1825| 277 | 49458 | 113199 | 
| 111 | **30 sklearn/ensemble/gradient_boosting.py** | 306 | 326| 150 | 49608 | 113199 | 
| 112 | 30 sklearn/ensemble/bagging.py | 218 | 244| 212 | 49820 | 113199 | 
| 113 | **30 sklearn/ensemble/gradient_boosting.py** | 272 | 304| 244 | 50064 | 113199 | 
| 114 | 31 sklearn/model_selection/_validation.py | 486 | 558| 745 | 50809 | 126309 | 
| 115 | 32 examples/feature_selection/plot_feature_selection_pipeline.py | 1 | 41| 301 | 51110 | 126610 | 
| 116 | 33 examples/model_selection/plot_underfitting_overfitting.py | 1 | 72| 631 | 51741 | 127241 | 
| 117 | 33 sklearn/utils/estimator_checks.py | 1423 | 1522| 940 | 52681 | 127241 | 
| 118 | 33 sklearn/linear_model/ransac.py | 230 | 326| 806 | 53487 | 127241 | 
| 119 | 34 examples/ensemble/plot_partial_dependence.py | 61 | 115| 496 | 53983 | 128284 | 
| 120 | 34 sklearn/utils/estimator_checks.py | 2427 | 2471| 423 | 54406 | 128284 | 
| 121 | 34 sklearn/linear_model/stochastic_gradient.py | 512 | 552| 372 | 54778 | 128284 | 
| 122 | 35 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 55532 | 129503 | 
| 123 | 35 sklearn/ensemble/bagging.py | 1 | 57| 380 | 55912 | 129503 | 
| 124 | 36 sklearn/linear_model/logistic.py | 2040 | 2101| 680 | 56592 | 151545 | 
| 125 | 37 examples/linear_model/plot_sgd_early_stopping.py | 72 | 86| 135 | 56727 | 152842 | 
| 126 | 38 sklearn/linear_model/ridge.py | 1135 | 1163| 276 | 57003 | 166227 | 
| 127 | 39 sklearn/ensemble/voting_classifier.py | 124 | 205| 692 | 57695 | 169072 | 
| 128 | 39 sklearn/utils/estimator_checks.py | 2211 | 2252| 399 | 58094 | 169072 | 
| 129 | 40 sklearn/multioutput.py | 43 | 59| 123 | 58217 | 174635 | 
| 130 | 40 sklearn/gaussian_process/gpr.py | 305 | 351| 521 | 58738 | 174635 | 
| 131 | 40 sklearn/linear_model/ransac.py | 1 | 19| 111 | 58849 | 174635 | 
| 132 | 40 sklearn/ensemble/weight_boosting.py | 175 | 213| 279 | 59128 | 174635 | 
| 133 | 40 sklearn/linear_model/stochastic_gradient.py | 1172 | 1201| 227 | 59355 | 174635 | 
| 134 | 40 sklearn/linear_model/stochastic_gradient.py | 465 | 510| 411 | 59766 | 174635 | 
| 135 | 41 sklearn/cross_decomposition/pls_.py | 288 | 349| 823 | 60589 | 182854 | 
| 136 | 42 examples/ensemble/plot_adaboost_twoclass.py | 1 | 85| 705 | 61294 | 183724 | 
| 137 | 43 sklearn/linear_model/passive_aggressive.py | 389 | 412| 209 | 61503 | 187704 | 
| 138 | 43 sklearn/ensemble/weight_boosting.py | 492 | 548| 534 | 62037 | 187704 | 
| 139 | 44 examples/gaussian_process/plot_gpr_noisy.py | 1 | 67| 747 | 62784 | 188833 | 
| 140 | 45 sklearn/impute.py | 936 | 971| 344 | 63128 | 199511 | 


## Patch

```diff
diff --git a/sklearn/ensemble/gradient_boosting.py b/sklearn/ensemble/gradient_boosting.py
--- a/sklearn/ensemble/gradient_boosting.py
+++ b/sklearn/ensemble/gradient_boosting.py
@@ -1476,20 +1476,25 @@ def fit(self, X, y, sample_weight=None, monitor=None):
                 raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                            dtype=np.float64)
             else:
-                try:
-                    self.init_.fit(X, y, sample_weight=sample_weight)
-                except TypeError:
-                    if sample_weight_is_none:
-                        self.init_.fit(X, y)
-                    else:
-                        raise ValueError(
-                            "The initial estimator {} does not support sample "
-                            "weights.".format(self.init_.__class__.__name__))
+                # XXX clean this once we have a support_sample_weight tag
+                if sample_weight_is_none:
+                    self.init_.fit(X, y)
+                else:
+                    msg = ("The initial estimator {} does not support sample "
+                           "weights.".format(self.init_.__class__.__name__))
+                    try:
+                        self.init_.fit(X, y, sample_weight=sample_weight)
+                    except TypeError:  # regular estimator without SW support
+                        raise ValueError(msg)
+                    except ValueError as e:
+                        if 'not enough values to unpack' in str(e):  # pipeline
+                            raise ValueError(msg) from e
+                        else:  # regular estimator whose input checking failed
+                            raise
 
                 raw_predictions = \
                     self.loss_.get_init_raw_predictions(X, self.init_)
 
-
             begin_at_stage = 0
 
             # The rng state must be preserved if warm_start is True

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_gradient_boosting.py b/sklearn/ensemble/tests/test_gradient_boosting.py
--- a/sklearn/ensemble/tests/test_gradient_boosting.py
+++ b/sklearn/ensemble/tests/test_gradient_boosting.py
@@ -39,6 +39,9 @@
 from sklearn.exceptions import DataConversionWarning
 from sklearn.exceptions import NotFittedError
 from sklearn.dummy import DummyClassifier, DummyRegressor
+from sklearn.pipeline import make_pipeline
+from sklearn.linear_model import LinearRegression
+from sklearn.svm import NuSVR
 
 
 GRADIENT_BOOSTING_ESTIMATORS = [GradientBoostingClassifier,
@@ -1366,6 +1369,33 @@ def test_gradient_boosting_with_init(gb, dataset_maker, init_estimator):
         gb(init=init_est).fit(X, y, sample_weight=sample_weight)
 
 
+def test_gradient_boosting_with_init_pipeline():
+    # Check that the init estimator can be a pipeline (see issue #13466)
+
+    X, y = make_regression(random_state=0)
+    init = make_pipeline(LinearRegression())
+    gb = GradientBoostingRegressor(init=init)
+    gb.fit(X, y)  # pipeline without sample_weight works fine
+
+    with pytest.raises(
+            ValueError,
+            match='The initial estimator Pipeline does not support sample '
+                  'weights'):
+        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))
+
+    # Passing sample_weight to a pipeline raises a ValueError. This test makes
+    # sure we make the distinction between ValueError raised by a pipeline that
+    # was passed sample_weight, and a ValueError raised by a regular estimator
+    # whose input checking failed.
+    with pytest.raises(
+            ValueError,
+            match='nu <= 0 or nu > 1'):
+        # Note that NuSVR properly supports sample_weight
+        init = NuSVR(gamma='auto', nu=1.5)
+        gb = GradientBoostingRegressor(init=init)
+        gb.fit(X, y, sample_weight=np.ones(X.shape[0]))
+
+
 @pytest.mark.parametrize('estimator, missing_method', [
     (GradientBoostingClassifier(init=LinearSVC()), 'predict_proba'),
     (GradientBoostingRegressor(init=OneHotEncoder()), 'predict')

```


## Code snippets

### 1 - sklearn/ensemble/gradient_boosting.py:

Start line: 1470, End line: 1543

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, monitor=None):
        # ... other code

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model and initialize raw predictions
            if self.init_ == 'zero':
                raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                           dtype=np.float64)
            else:
                try:
                    self.init_.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    if sample_weight_is_none:
                        self.init_.fit(X, y)
                    else:
                        raise ValueError(
                            "The initial estimator {} does not support sample "
                            "weights.".format(self.init_.__class__.__name__))

                raw_predictions = \
                    self.loss_.get_init_raw_predictions(X, self.init_)


            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            raw_predictions = self._raw_predict(X)
            self._resize_state()

        if self.presort is True and issparse(X):
            raise ValueError(
                "Presorting is not supported for sparse matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if presort == 'auto':
            presort = not issparse(X)

        X_idx_sorted = None
        if presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        # fit the boosting stages
        n_stages = self._fit_stages(
            X, y, raw_predictions, sample_weight, self._rng, X_val, y_val,
            sample_weight_val, begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self
```
### 2 - sklearn/utils/estimator_checks.py:

Start line: 1048, End line: 1074

```python
@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if _safe_tags(estimator_orig, 'non_deterministic'):
        msg = name + ' is non deterministic'
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X -= X.min()
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
### 3 - sklearn/pipeline.py:

Start line: 162, End line: 188

```python
class Pipeline(_BaseComposition):

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None or t == 'passthrough':
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform "
                                "or be the string 'passthrough' "
                                "'%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if (estimator is not None and estimator != 'passthrough'
                and not hasattr(estimator, "fit")):
            raise TypeError(
                "Last step of Pipeline should implement fit "
                "or be the string 'passthrough'. "
                "'%s' (type %s) doesn't" % (estimator, type(estimator)))
```
### 4 - sklearn/pipeline.py:

Start line: 244, End line: 288

```python
class Pipeline(_BaseComposition):

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for step_idx, name, transformer in self._iter(with_final=False):
            if hasattr(memory, 'location'):
                # joblib >= 0.12
                if memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            elif hasattr(memory, 'cachedir'):
                # joblib < 0.11
                if memory.cachedir is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transfomer
            Xt, fitted_transformer = fit_transform_one_cached(
                cloned_transformer, Xt, y, None,
                **fit_params_steps[name])
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]
```
### 5 - sklearn/ensemble/gradient_boosting.py:

Start line: 2466, End line: 2489

```python
class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):

    _SUPPORTED_LOSS = ('ls', 'lad', 'huber', 'quantile')

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            random_state=random_state, alpha=alpha, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes, warm_start=warm_start,
            presort=presort, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol)
```
### 6 - sklearn/ensemble/gradient_boosting.py:

Start line: 1637, End line: 1661

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _make_estimator(self, append=True):
        # we don't need _make_estimator
        raise NotImplementedError()

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimtor."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if X.shape[1] != self.n_features_:
            raise ValueError("X.shape[1] should be {0:d}, not {1:d}.".format(
                self.n_features_, X.shape[1]))
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.loss_.K),
                                       dtype=np.float64)
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(
                X, self.init_).astype(np.float64)
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate,
                       raw_predictions)
        return raw_predictions
```
### 7 - sklearn/pipeline.py:

Start line: 29, End line: 124

```python
class Pipeline(_BaseComposition):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    it to 'passthrough' or ``None``.

    Read more in the :ref:`User Guide <pipeline>`.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    See also
    --------
    sklearn.pipeline.make_pipeline : convenience function for simplified
        pipeline construction.

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.datasets import samples_generator
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.feature_selection import f_regression
    >>> from sklearn.pipeline import Pipeline
    >>> # generate some data to play with
    >>> X, y = samples_generator.make_classification(
    ...     n_informative=5, n_redundant=0, random_state=42)
    >>> # ANOVA SVM-C
    >>> anova_filter = SelectKBest(f_regression, k=5)
    >>> clf = svm.SVC(kernel='linear')
    >>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
    >>> # You can set the parameters using the names issued
    >>> # For instance, fit using a k of 10 in the SelectKBest
    >>> # and a parameter 'C' of the svm
    >>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
    ...                      # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('anova', SelectKBest(...)),
                    ('svc', SVC(...))])
    >>> prediction = anova_svm.predict(X)
    >>> anova_svm.score(X, y)                        # doctest: +ELLIPSIS
    0.83
    >>> # getting the selected features chosen by anova_filter
    >>> anova_svm['anova'].get_support()
    ... # doctest: +NORMALIZE_WHITESPACE
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Another way to get selected features chosen by anova_filter
    >>> anova_svm.named_steps.anova.get_support()
    ... # doctest: +NORMALIZE_WHITESPACE
    array([False, False,  True,  True, False, False,  True,  True, False,
           True, False,  True,  True, False,  True, False,  True,  True,
           False, False])
    >>> # Indexing can also be used to extract a sub-pipeline.
    >>> sub_pipeline = anova_svm[:1]
    >>> sub_pipeline  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Pipeline(memory=None, steps=[('anova', ...)])
    >>> coef = anova_svm[-1].coef_
    >>> anova_svm['svc'] is anova_svm[-1]
    True
    >>> coef.shape
    (1, 10)
    >>> sub_pipeline.inverse_transform(coef).shape
    (1, 20)
    """
```
### 8 - sklearn/ensemble/gradient_boosting.py:

Start line: 1395, End line: 1468

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, monitor=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        y : array-like, shape (n_samples,)
            Target values (strings or integers in classification, real numbers
            in regression)
            For classification, labels must correspond to classes.

        sample_weight : array-like, shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.

        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator and the local variables of
            ``_fit_stages`` as keyword arguments ``callable(i, self,
            locals())``. If the callable returns ``True`` the fitting procedure
            is stopped. The monitor can be used for various things such as
            computing held-out estimates, early stopping, model introspect, and
            snapshoting.

        Returns
        -------
        self : object
        """
        # if not warmstart - clear the estimator state
        if not self.warm_start:
            self._clear_state()

        # Check input
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], dtype=DTYPE)
        n_samples, self.n_features_ = X.shape

        sample_weight_is_none = sample_weight is None
        if sample_weight_is_none:
            sample_weight = np.ones(n_samples, dtype=np.float32)
        else:
            sample_weight = column_or_1d(sample_weight, warn=True)
            sample_weight_is_none = False

        check_consistent_length(X, y, sample_weight)

        y = self._validate_y(y, sample_weight)

        if self.n_iter_no_change is not None:
            X, X_val, y, y_val, sample_weight, sample_weight_val = (
                train_test_split(X, y, sample_weight,
                                 random_state=self.random_state,
                                 test_size=self.validation_fraction))
            if is_classifier(self):
                if self.n_classes_ != np.unique(y).shape[0]:
                    # We choose to error here. The problem is that the init
                    # estimator would be trained on y, which has some missing
                    # classes now, so its predictions would not have the
                    # correct shape.
                    raise ValueError(
                        'The training data after the early stopping split '
                        'is missing some classes. Try using another random '
                        'seed.'
                    )
        else:
            X_val = y_val = sample_weight_val = None

        self._check_params()
        # ... other code
```
### 9 - sklearn/ensemble/gradient_boosting.py:

Start line: 1256, End line: 1338

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid. """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0 but "
                             "was %r" % self.n_estimators)

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0 but "
                             "was %r" % self.learning_rate)

        if (self.loss not in self._SUPPORTED_LOSS
                or self.loss not in _gb_losses.LOSS_FUNCTIONS):
            raise ValueError("Loss '{0:s}' not supported. ".format(self.loss))

        if self.loss == 'deviance':
            loss_class = (_gb_losses.MultinomialDeviance
                          if len(self.classes_) > 2
                          else _gb_losses.BinomialDeviance)
        else:
            loss_class = _gb_losses.LOSS_FUNCTIONS[self.loss]

        if self.loss in ('huber', 'quantile'):
            self.loss_ = loss_class(self.n_classes_, self.alpha)
        else:
            self.loss_ = loss_class(self.n_classes_)

        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0,1] but "
                             "was %r" % self.subsample)

        if self.init is not None:
            # init must be an estimator or 'zero'
            if isinstance(self.init, BaseEstimator):
                self.loss_.check_init_estimator(self.init)
            elif not (isinstance(self.init, str) and self.init == 'zero'):
                raise ValueError(
                    "The init parameter must be an estimator or 'zero'. "
                    "Got init={}".format(self.init)
                )

        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0.0, 1.0) but "
                             "was %r" % self.alpha)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                # if is_classification
                if self.n_classes_ > 1:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    # is regression
                    max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features: %r. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'." % self.max_features)
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if 0. < self.max_features <= 1.:
                max_features = max(int(self.max_features *
                                       self.n_features_), 1)
            else:
                raise ValueError("max_features must be in (0, n_features]")

        self.max_features_ = max_features

        if not isinstance(self.n_iter_no_change,
                          (numbers.Integral, np.integer, type(None))):
            raise ValueError("n_iter_no_change should either be None or an "
                             "integer. %r was passed"
                             % self.n_iter_no_change)

        allowed_presort = ('auto', True, False)
        if self.presort not in allowed_presort:
            raise ValueError("'presort' should be in {}. Got {!r} instead."
                             .format(allowed_presort, self.presort))
```
### 10 - examples/ensemble/plot_gradient_boosting_early_stopping.py:

Start line: 1, End line: 103

```python
"""
===================================
Early stopping of Gradient Boosting
===================================

Gradient boosting is an ensembling technique where several weak learners
(regression trees) are combined to yield a powerful single model, in an
iterative fashion.

Early stopping support in Gradient Boosting enables us to find the least number
of iterations which is sufficient to build a model that generalizes well to
unseen data.

The concept of early stopping is simple. We specify a ``validation_fraction``
which denotes the fraction of the whole dataset that will be kept aside from
training to assess the validation loss of the model. The gradient boosting
model is trained using the training set and evaluated using the validation set.
When each additional stage of regression tree is added, the validation set is
used to score the model.  This is continued until the scores of the model in
the last ``n_iter_no_change`` stages do not improve by atleast `tol`. After
that the model is considered to have converged and further addition of stages
is "stopped early".

The number of stages of the final model is available at the attribute
``n_estimators_``.

This example illustrates how the early stopping can used in the
:class:`sklearn.ensemble.GradientBoostingClassifier` model to achieve
almost the same accuracy as compared to a model built without early stopping
using many fewer estimators. This can significantly reduce training time,
memory usage and prediction latency.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.model_selection import train_test_split

print(__doc__)

data_list = [datasets.load_iris(), datasets.load_digits()]
data_list = [(d.data, d.target) for d in data_list]
data_list += [datasets.make_hastie_10_2()]
names = ['Iris Data', 'Digits Data', 'Hastie Data']

n_gb = []
score_gb = []
time_gb = []
n_gbes = []
score_gbes = []
time_gbes = []

n_estimators = 500

for X, y in data_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    # We specify that if the scores don't improve by atleast 0.01 for the last
    # 10 stages, stop fitting additional stages
    gbes = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                               validation_fraction=0.2,
                                               n_iter_no_change=5, tol=0.01,
                                               random_state=0)
    gb = ensemble.GradientBoostingClassifier(n_estimators=n_estimators,
                                             random_state=0)
    start = time.time()
    gb.fit(X_train, y_train)
    time_gb.append(time.time() - start)

    start = time.time()
    gbes.fit(X_train, y_train)
    time_gbes.append(time.time() - start)

    score_gb.append(gb.score(X_test, y_test))
    score_gbes.append(gbes.score(X_test, y_test))

    n_gb.append(gb.n_estimators_)
    n_gbes.append(gbes.n_estimators_)

bar_width = 0.2
n = len(data_list)
index = np.arange(0, n * bar_width, bar_width) * 2.5
index = index[0:n]

#######################################################################
# Compare scores with and without early stopping
# ----------------------------------------------

plt.figure(figsize=(9, 5))

bar1 = plt.bar(index, score_gb, bar_width, label='Without early stopping',
               color='crimson')
bar2 = plt.bar(index + bar_width, score_gbes, bar_width,
               label='With early stopping', color='coral')
```
### 11 - sklearn/ensemble/gradient_boosting.py:

Start line: 2221, End line: 2464

```python
class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):
    """Gradient Boosting for regression.

    GB builds an additive model in a forward stage-wise fashion;
    it allows for the optimization of arbitrary differentiable loss functions.
    In each stage a regression tree is fit on the negative gradient of the
    given loss function.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'ls', 'lad', 'huber', 'quantile'}, optional (default='ls')
        loss function to be optimized. 'ls' refers to least squares
        regression. 'lad' (least absolute deviation) is a highly robust
        loss function solely based on order information of the input
        variables. 'huber' is a combination of the two. 'quantile'
        allows quantile regression (use `alpha` to specify the quantile).

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

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

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

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

    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide `fit` and `predict`. If 'zero', the initial
        raw predictions are set to zero. By default a ``DummyEstimator`` is
        used, predicting either the average target value (for loss='ls'), or
        a quantile for the other losses.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    alpha : float (default=0.9)
        The alpha-quantile of the huber loss function and the quantile
        loss function. Only if ``loss='huber'`` or ``loss='quantile'``.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           optional parameter *presort*.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20


    Attributes
    ----------
    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : array of DecisionTreeRegressor, shape (n_estimators, 1)
        The collection of fitted sub-estimators.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeRegressor, RandomForestRegressor

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """
```
### 13 - sklearn/ensemble/gradient_boosting.py:

Start line: 2034, End line: 2044

```python
class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):

    def _validate_y(self, y, sample_weight):
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_trim_classes = np.count_nonzero(np.bincount(y, sample_weight))
        if n_trim_classes < 2:
            raise ValueError("y contains %d class after sample_weight "
                             "trimmed classes with zero weights, while a "
                             "minimum of 2 classes are required."
                             % n_trim_classes)
        self.n_classes_ = len(self.classes_)
        return y
```
### 14 - sklearn/ensemble/gradient_boosting.py:

Start line: 2007, End line: 2032

```python
class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):

    _SUPPORTED_LOSS = ('deviance', 'exponential')

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4):

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, presort=presort,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol)
```
### 15 - sklearn/ensemble/gradient_boosting.py:

Start line: 1, End line: 61

```python
"""Gradient Boosted Regression Trees

This module contains methods for fitting gradient boosted regression trees for
both classification and regression.

The module structure is the following:

- The ``BaseGradientBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ in the concrete ``LossFunction`` used.

- ``GradientBoostingClassifier`` implements gradient boosting for
  classification problems.

- ``GradientBoostingRegressor`` implements gradient boosting for
  regression problems.
"""

from abc import ABCMeta
from abc import abstractmethod

from .base import BaseEnsemble
from ..base import ClassifierMixin
from ..base import RegressorMixin
from ..base import BaseEstimator
from ..base import is_classifier

from ._gradient_boosting import predict_stages
from ._gradient_boosting import predict_stage
from ._gradient_boosting import _random_sample_mask

import numbers
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.special import expit

from time import time
from ..model_selection import train_test_split
from ..tree.tree import DecisionTreeRegressor
from ..tree._tree import DTYPE
from ..tree._tree import TREE_LEAF
from . import _gb_losses

from ..utils import check_random_state
from ..utils import check_array
from ..utils import check_X_y
from ..utils import column_or_1d
from ..utils import check_consistent_length
from ..utils import deprecated
from ..utils.fixes import logsumexp
from ..utils.stats import _weighted_percentile
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..exceptions import NotFittedError
```
### 19 - sklearn/ensemble/gradient_boosting.py:

Start line: 1202, End line: 1254

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _fit_stage(self, i, X, y, raw_predictions, sample_weight, sample_mask,
                   random_state, X_idx_sorted, X_csc=None, X_csr=None):
        """Fit another stage of ``n_classes_`` trees to the boosting model. """

        assert sample_mask.dtype == np.bool
        loss = self.loss_
        original_y = y

        # Need to pass a copy of raw_predictions to negative_gradient()
        # because raw_predictions is partially updated at the end of the loop
        # in update_terminal_regions(), and gradients need to be evaluated at
        # iteration i - 1.
        raw_predictions_copy = raw_predictions.copy()

        for k in range(loss.K):
            if loss.is_multi_class:
                y = np.array(original_y == k, dtype=np.float64)

            residual = loss.negative_gradient(y, raw_predictions_copy, k=k,
                                              sample_weight=sample_weight)

            # induce regression tree on residuals
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                splitter='best',
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                min_impurity_split=self.min_impurity_split,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                random_state=random_state,
                presort=self.presort)

            if self.subsample < 1.0:
                # no inplace multiplication!
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            X = X_csr if X_csr is not None else X
            tree.fit(X, residual, sample_weight=sample_weight,
                     check_input=False, X_idx_sorted=X_idx_sorted)

            # update tree leaves
            loss.update_terminal_regions(
                tree.tree_, X, y, residual, raw_predictions, sample_weight,
                sample_mask, learning_rate=self.learning_rate, k=k)

            # add tree to ensemble
            self.estimators_[i, k] = tree

        return raw_predictions
```
### 21 - sklearn/ensemble/gradient_boosting.py:

Start line: 1340, End line: 1353

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _init_state(self):
        """Initialize model state and allocate model state data structures. """

        self.init_ = self.init
        if self.init_ is None:
            self.init_ = self.loss_.init_estimator()

        self.estimators_ = np.empty((self.n_estimators, self.loss_.K),
                                    dtype=np.object)
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        # do oob?
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros((self.n_estimators),
                                             dtype=np.float64)
```
### 22 - sklearn/ensemble/gradient_boosting.py:

Start line: 1757, End line: 2005

```python
class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
    """Gradient Boosting for classification.

    GB builds an additive model in a
    forward stage-wise fashion; it allows for the optimization of
    arbitrary differentiable loss functions. In each stage ``n_classes_``
    regression trees are fit on the negative gradient of the
    binomial or multinomial deviance loss function. Binary classification
    is a special case where only a single regression tree is induced.

    Read more in the :ref:`User Guide <gradient_boosting>`.

    Parameters
    ----------
    loss : {'deviance', 'exponential'}, optional (default='deviance')
        loss function to be optimized. 'deviance' refers to
        deviance (= logistic regression) for classification
        with probabilistic outputs. For loss 'exponential' gradient
        boosting recovers the AdaBoost algorithm.

    learning_rate : float, optional (default=0.1)
        learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.

    n_estimators : int (default=100)
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.

    subsample : float, optional (default=1.0)
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias.

    criterion : string, optional (default="friedman_mse")
        The function to measure the quality of a split. Supported criteria
        are "friedman_mse" for the mean squared error with improvement
        score by Friedman, "mse" for mean squared error, and "mae" for
        the mean absolute error. The default value of "friedman_mse" is
        generally the best as it can provide a better approximation in
        some cases.

        .. versionadded:: 0.18

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

    max_depth : integer, optional (default=3)
        maximum depth of the individual regression estimators. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.

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

    init : estimator or 'zero', optional (default=None)
        An estimator object that is used to compute the initial predictions.
        ``init`` has to provide `fit` and `predict_proba`. If 'zero', the
        initial raw predictions are set to zero. By default, a
        ``DummyEstimator`` predicting the classes priors is used.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Choosing `max_features < n_features` leads to a reduction of variance
        and an increase in bias.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    verbose : int, default: 0
        Enable verbose output. If 1 then it prints progress and performance
        once in a while (the more trees the lower the frequency). If greater
        than 1 then it prints progress and performance for every tree.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    warm_start : bool, default: False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    presort : bool or 'auto', optional (default='auto')
        Whether to presort the data to speed up the finding of best splits in
        fitting. Auto mode by default will use presorting on dense data and
        default to normal sorting on sparse data. Setting presort to true on
        sparse data will raise an error.

        .. versionadded:: 0.17
           *presort* parameter.

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if ``n_iter_no_change`` is set to an integer.

        .. versionadded:: 0.20

    n_iter_no_change : int, default None
        ``n_iter_no_change`` is used to decide if early stopping will be used
        to terminate training when validation score is not improving. By
        default it is set to None to disable early stopping. If set to a
        number, it will set aside ``validation_fraction`` size of the training
        data as validation and terminate training when validation score is not
        improving in all of the previous ``n_iter_no_change`` numbers of
        iterations.

        .. versionadded:: 0.20

    tol : float, optional, default 1e-4
        Tolerance for the early stopping. When the loss is not improving
        by at least tol for ``n_iter_no_change`` iterations (if set to a
        number), the training stops.

        .. versionadded:: 0.20

    Attributes
    ----------
    n_estimators_ : int
        The number of estimators as selected by early stopping (if
        ``n_iter_no_change`` is specified). Otherwise it is set to
        ``n_estimators``.

        .. versionadded:: 0.20

    feature_importances_ : array, shape (n_features,)
        The feature importances (the higher, the more important the feature).

    oob_improvement_ : array, shape (n_estimators,)
        The improvement in loss (= deviance) on the out-of-bag samples
        relative to the previous iteration.
        ``oob_improvement_[0]`` is the improvement in
        loss of the first stage over the ``init`` estimator.

    train_score_ : array, shape (n_estimators,)
        The i-th score ``train_score_[i]`` is the deviance (= loss) of the
        model at iteration ``i`` on the in-bag sample.
        If ``subsample == 1`` this is the deviance on the training data.

    loss_ : LossFunction
        The concrete ``LossFunction`` object.

    init_ : estimator
        The estimator that provides the initial predictions.
        Set via the ``init`` argument or ``loss.init_estimator``.

    estimators_ : ndarray of DecisionTreeRegressor,\
    e (n_estimators, ``loss_.K``)
        The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
        classification, otherwise n_classes.

    Notes
    -----
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier, RandomForestClassifier
    AdaBoostClassifier

    References
    ----------
    J. Friedman, Greedy Function Approximation: A Gradient Boosting
    Machine, The Annals of Statistics, Vol. 29, No. 5, 2001.

    J. Friedman, Stochastic Gradient Boosting, 1999

    T. Hastie, R. Tibshirani and J. Friedman.
    Elements of Statistical Learning Ed. 2, Springer, 2009.
    """
```
### 37 - sklearn/ensemble/gradient_boosting.py:

Start line: 1063, End line: 1088

```python
@deprecated("All Losses in sklearn.ensemble.gradient_boosting are "
            "deprecated in version "
            "0.21 and will be removed in version 0.23.")
class ExponentialLoss(ClassificationLossFunction):

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        pred = pred.take(terminal_region, axis=0)
        y = y.take(terminal_region, axis=0)
        sample_weight = sample_weight.take(terminal_region, axis=0)

        y_ = 2. * y - 1.

        numerator = np.sum(y_ * sample_weight * np.exp(-y_ * pred))
        denominator = np.sum(sample_weight * np.exp(-y_ * pred))

        # prevents overflow and division by zero
        if abs(denominator) < 1e-150:
            tree.value[leaf, 0, 0] = 0.0
        else:
            tree.value[leaf, 0, 0] = numerator / denominator

    def _score_to_proba(self, score):
        proba = np.ones((score.shape[0], 2), dtype=np.float64)
        proba[:, 1] = expit(2.0 * score.ravel())
        proba[:, 0] -= proba[:, 1]
        return proba

    def _score_to_decision(self, score):
        return (score.ravel() >= 0.0).astype(np.int)
```
### 41 - sklearn/ensemble/gradient_boosting.py:

Start line: 1355, End line: 1366

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _clear_state(self):
        """Clear the state of the gradient boosting model. """
        if hasattr(self, 'estimators_'):
            self.estimators_ = np.empty((0, 0), dtype=np.object)
        if hasattr(self, 'train_score_'):
            del self.train_score_
        if hasattr(self, 'oob_improvement_'):
            del self.oob_improvement_
        if hasattr(self, 'init_'):
            del self.init_
        if hasattr(self, '_rng'):
            del self._rng
```
### 54 - sklearn/ensemble/gradient_boosting.py:

Start line: 1368, End line: 1393

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _resize_state(self):
        """Add additional ``n_estimators`` entries to all attributes. """
        # self.n_estimators is the number of additional est to fit
        total_n_estimators = self.n_estimators
        if total_n_estimators < self.estimators_.shape[0]:
            raise ValueError('resize with smaller n_estimators %d < %d' %
                             (total_n_estimators, self.estimators_[0]))

        self.estimators_ = np.resize(self.estimators_,
                                     (total_n_estimators, self.loss_.K))
        self.train_score_ = np.resize(self.train_score_, total_n_estimators)
        if (self.subsample < 1 or hasattr(self, 'oob_improvement_')):
            # if do oob resize arrays or create new if not available
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = np.resize(self.oob_improvement_,
                                                  total_n_estimators)
            else:
                self.oob_improvement_ = np.zeros((total_n_estimators,),
                                                 dtype=np.float64)

    def _is_initialized(self):
        return len(getattr(self, 'estimators_', [])) > 0

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self, 'estimators_')
```
### 62 - sklearn/ensemble/gradient_boosting.py:

Start line: 694, End line: 705

```python
@deprecated("All Losses in sklearn.ensemble.gradient_boosting are "
            "deprecated in version "
            "0.21 and will be removed in version 0.23.")
class HuberLossFunction(RegressionLossFunction):

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        gamma = self.gamma
        diff = (y.take(terminal_region, axis=0)
                - pred.take(terminal_region, axis=0))
        median = _weighted_percentile(diff, sample_weight, percentile=50)
        diff_minus_median = diff - median
        tree.value[leaf, 0] = median + np.mean(
            np.sign(diff_minus_median) *
            np.minimum(np.abs(diff_minus_median), gamma))
```
### 65 - sklearn/ensemble/gradient_boosting.py:

Start line: 1545, End line: 1635

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def _fit_stages(self, X, y, raw_predictions, sample_weight, random_state,
                    X_val, y_val, sample_weight_val,
                    begin_at_stage=0, monitor=None, X_idx_sorted=None):
        """Iteratively fits the stages.

        For each stage it computes the progress (OOB, train score)
        and delegates to ``_fit_stage``.
        Returns the number of stages fit; might differ from ``n_estimators``
        due to early stopping.
        """
        n_samples = X.shape[0]
        do_oob = self.subsample < 1.0
        sample_mask = np.ones((n_samples, ), dtype=np.bool)
        n_inbag = max(1, int(self.subsample * n_samples))
        loss_ = self.loss_

        # Set min_weight_leaf from min_weight_fraction_leaf
        if self.min_weight_fraction_leaf != 0. and sample_weight is not None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))
        else:
            min_weight_leaf = 0.

        if self.verbose:
            verbose_reporter = VerboseReporter(self.verbose)
            verbose_reporter.init(self, begin_at_stage)

        X_csc = csc_matrix(X) if issparse(X) else None
        X_csr = csr_matrix(X) if issparse(X) else None

        if self.n_iter_no_change is not None:
            loss_history = np.full(self.n_iter_no_change, np.inf)
            # We create a generator to get the predictions for X_val after
            # the addition of each successive stage
            y_val_pred_iter = self._staged_raw_predict(X_val)

        # perform boosting iterations
        i = begin_at_stage
        for i in range(begin_at_stage, self.n_estimators):

            # subsampling
            if do_oob:
                sample_mask = _random_sample_mask(n_samples, n_inbag,
                                                  random_state)
                # OOB score before adding this stage
                old_oob_score = loss_(y[~sample_mask],
                                      raw_predictions[~sample_mask],
                                      sample_weight[~sample_mask])

            # fit next stage of trees
            raw_predictions = self._fit_stage(
                i, X, y, raw_predictions, sample_weight, sample_mask,
                random_state, X_idx_sorted, X_csc, X_csr)

            # track deviance (= loss)
            if do_oob:
                self.train_score_[i] = loss_(y[sample_mask],
                                             raw_predictions[sample_mask],
                                             sample_weight[sample_mask])
                self.oob_improvement_[i] = (
                    old_oob_score - loss_(y[~sample_mask],
                                          raw_predictions[~sample_mask],
                                          sample_weight[~sample_mask]))
            else:
                # no need to fancy index w/ no subsampling
                self.train_score_[i] = loss_(y, raw_predictions, sample_weight)

            if self.verbose > 0:
                verbose_reporter.update(i, self)

            if monitor is not None:
                early_stopping = monitor(i, self, locals())
                if early_stopping:
                    break

            # We also provide an early stopping based on the score from
            # validation set (X_val, y_val), if n_iter_no_change is set
            if self.n_iter_no_change is not None:
                # By calling next(y_val_pred_iter), we get the predictions
                # for X_val after the addition of the current stage
                validation_loss = loss_(y_val, next(y_val_pred_iter),
                                        sample_weight_val)

                # Require validation_score to be better (less) than at least
                # one of the last n_iter_no_change evaluations
                if np.any(validation_loss + self.tol < loss_history):
                    loss_history[i % len(loss_history)] = validation_loss
                else:
                    break

        return i + 1
```
### 72 - sklearn/ensemble/gradient_boosting.py:

Start line: 1166, End line: 1200

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting. """

    @abstractmethod
    def __init__(self, loss, learning_rate, n_estimators, criterion,
                 min_samples_split, min_samples_leaf, min_weight_fraction_leaf,
                 max_depth, min_impurity_decrease, min_impurity_split,
                 init, subsample, max_features,
                 random_state, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto',
                 validation_fraction=0.1, n_iter_no_change=None,
                 tol=1e-4):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.presort = presort
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
```
### 74 - sklearn/ensemble/gradient_boosting.py:

Start line: 593, End line: 599

```python
@deprecated("All Losses in sklearn.ensemble.gradient_boosting are "
            "deprecated in version "
            "0.21 and will be removed in version 0.23.")
class LeastAbsoluteError(RegressionLossFunction):

    def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
                                residual, pred, sample_weight):
        """LAD updates terminal regions to median estimates. """
        terminal_region = np.where(terminal_regions == leaf)[0]
        sample_weight = sample_weight.take(terminal_region, axis=0)
        diff = y.take(terminal_region, axis=0) - pred.take(terminal_region, axis=0)
        tree.value[leaf, 0, 0] = _weighted_percentile(diff, sample_weight, percentile=50)
```
### 79 - sklearn/ensemble/gradient_boosting.py:

Start line: 168, End line: 217

```python
@deprecated("LogOddsEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class LogOddsEstimator:
    """An estimator predicting the log odds ratio."""
    scale = 1.0

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : array, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape (n_samples,)
            Individual weights for each sample
        """
        # pre-cond: pos, neg are encoded as 1, 0
        if sample_weight is None:
            pos = np.sum(y)
            neg = y.shape[0] - pos
        else:
            pos = np.sum(sample_weight * y)
            neg = np.sum(sample_weight * (1 - y))

        if neg == 0 or pos == 0:
            raise ValueError('y contains non binary labels.')
        self.prior = self.scale * np.log(pos / neg)

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, 'prior')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.prior)
        return y
```
### 81 - sklearn/ensemble/gradient_boosting.py:

Start line: 1720, End line: 2217

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like, shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
```
### 86 - sklearn/ensemble/gradient_boosting.py:

Start line: 2531, End line: 2553

```python
class GradientBoostingRegressor(BaseGradientBoosting, RegressorMixin):

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like, shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
        """

        leaves = super().apply(X)
        leaves = leaves.reshape(X.shape[0], self.estimators_.shape[0])
        return leaves
```
### 97 - sklearn/ensemble/gradient_boosting.py:

Start line: 220, End line: 269

```python
@deprecated("ScaledLogOddsEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class ScaledLogOddsEstimator(LogOddsEstimator):
    """Log odds ratio scaled by 0.5 -- for exponential loss. """
    scale = 0.5


@deprecated("PriorProbablityEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class PriorProbabilityEstimator:
    """An estimator predicting the probability of each
    class in the training data.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : array, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : array, shape (n_samples,)
            Individual weights for each sample
        """
        if sample_weight is None:
            sample_weight = np.ones_like(y, dtype=np.float64)
        class_counts = np.bincount(y, weights=sample_weight)
        self.priors = class_counts / class_counts.sum()

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, 'priors')

        y = np.empty((X.shape[0], self.priors.shape[0]), dtype=np.float64)
        y[:] = self.priors
        return y
```
### 104 - sklearn/ensemble/gradient_boosting.py:

Start line: 125, End line: 165

```python
@deprecated("MeanEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class MeanEstimator:
    """An estimator predicting the mean of the training targets."""
    def fit(self, X, y, sample_weight=None):
        """Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : array, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape (n_samples,)
            Individual weights for each sample
        """
        if sample_weight is None:
            self.mean = np.mean(y)
        else:
            self.mean = np.average(y, weights=sample_weight)

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, 'mean')

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        y.fill(self.mean)
        return y
```
### 109 - sklearn/ensemble/gradient_boosting.py:

Start line: 1028, End line: 1061

```python
@deprecated("All Losses in sklearn.ensemble.gradient_boosting are "
            "deprecated in version "
            "0.21 and will be removed in version 0.23.")
class ExponentialLoss(ClassificationLossFunction):

    def __call__(self, y, pred, sample_weight=None):
        """Compute the exponential loss

        Parameters
        ----------
        y : array, shape (n_samples,)
            True labels

        pred : array, shape (n_samples,)
            Predicted labels

        sample_weight : array-like, shape (n_samples,), optional
            Sample weights.
        """
        pred = pred.ravel()
        if sample_weight is None:
            return np.mean(np.exp(-(2. * y - 1.) * pred))
        else:
            return (1.0 / sample_weight.sum() *
                    np.sum(sample_weight * np.exp(-(2 * y - 1) * pred)))

    def negative_gradient(self, y, pred, **kargs):
        """Compute the residual (= negative gradient).

        Parameters
        ----------
        y : array, shape (n_samples,)
            True labels

        pred : array, shape (n_samples,)
            Predicted labels
        """
        y_ = -(2. * y - 1.)
        return y_ * np.exp(y_ * pred.ravel())
```
### 111 - sklearn/ensemble/gradient_boosting.py:

Start line: 306, End line: 326

```python
@deprecated("Using ZeroEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class ZeroEstimator:

    def predict(self, X):
        """Predict labels

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, 'n_classes')

        y = np.empty((X.shape[0], self.n_classes), dtype=np.float64)
        y.fill(0.0)
        return y

    def predict_proba(self, X):
        return self.predict(X)
```
### 113 - sklearn/ensemble/gradient_boosting.py:

Start line: 272, End line: 304

```python
@deprecated("Using ZeroEstimator is deprecated in version "
            "0.21 and will be removed in version 0.23.")
class ZeroEstimator:
    """An estimator that simply predicts zero.

    .. deprecated:: 0.21
        Using ``ZeroEstimator`` or ``init='zero'`` is deprecated in version
        0.21 and will be removed in version 0.23.

    """

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data

        y : numpy, shape (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary

        sample_weight : array, shape (n_samples,)
            Individual weights for each sample
        """
        if np.issubdtype(y.dtype, np.signedinteger):
            # classification
            self.n_classes = np.unique(y).shape[0]
            if self.n_classes == 2:
                self.n_classes = 1
        else:
            # regression
            self.n_classes = 1
```

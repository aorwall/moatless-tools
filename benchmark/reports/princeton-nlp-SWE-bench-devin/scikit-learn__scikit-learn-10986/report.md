# scikit-learn__scikit-learn-10986

| **scikit-learn/scikit-learn** | `ca436e7017ae069a29de19caf71689e9b9b9c452` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 6022 |
| **Any found context length** | 6022 |
| **Avg pos** | 7.0 |
| **Min pos** | 7 |
| **Max pos** | 7 |
| **Top file pos** | 2 |
| **Missing snippets** | 1 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -675,7 +675,13 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                     'shape (%d, %d) or (%d, %d)' % (
                         coef.shape[0], coef.shape[1], classes.size,
                         n_features, classes.size, n_features + 1))
-            w0[:, :coef.shape[1]] = coef
+
+            if n_classes == 1:
+                w0[0, :coef.shape[1]] = -coef
+                w0[1, :coef.shape[1]] = coef
+            else:
+                w0[:, :coef.shape[1]] = coef
+
 
     if multi_class == 'multinomial':
         # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/logistic.py | 678 | 678 | 7 | 2 | 6022


## Problem Statement

```
Warm start bug when fitting a LogisticRegression model on binary outcomes with `multi_class='multinomial'`.
<!--
If your issue is a usage question, submit it here instead:
- StackOverflow with the scikit-learn tag: http://stackoverflow.com/questions/tagged/scikit-learn
- Mailing List: https://mail.python.org/mailman/listinfo/scikit-learn
For more information, see User Questions: http://scikit-learn.org/stable/support.html#user-questions
-->

<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
Bug when fitting a LogisticRegression model on binary outcomes with multi_class='multinomial' when using warm start. Note that it is similar to the issue here https://github.com/scikit-learn/scikit-learn/issues/9889 i.e. only using a 1D `coef` object on binary outcomes even when using `multi_class='multinomial'` as opposed to a 2D `coef` object.

#### Steps/Code to Reproduce
<!--
Example:
\`\`\`python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = ["Help I have a bug" for i in range(1000)]

vectorizer = CountVectorizer(input=docs, analyzer='word')
lda_features = vectorizer.fit_transform(docs)

lda_model = LatentDirichletAllocation(
    n_topics=10,
    learning_method='online',
    evaluate_every=10,
    n_jobs=4,
)
model = lda_model.fit(lda_features)
\`\`\`
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics
    import numpy as np

    # Set up a logistic regression object
    lr = LogisticRegression(C=1000000, multi_class='multinomial',
                        solver='sag', tol=0.0001, warm_start=True,
                        verbose=0)

    # Set independent variable values
    Z = np.array([
    [ 0.        ,  0.        ],
    [ 1.33448632,  0.        ],
    [ 1.48790105, -0.33289528],
    [-0.47953866, -0.61499779],
    [ 1.55548163,  1.14414766],
    [-0.31476657, -1.29024053],
    [-1.40220786, -0.26316645],
    [ 2.227822  , -0.75403668],
    [-0.78170885, -1.66963585],
    [ 2.24057471, -0.74555021],
    [-1.74809665,  2.25340192],
    [-1.74958841,  2.2566389 ],
    [ 2.25984734, -1.75106702],
    [ 0.50598996, -0.77338402],
    [ 1.21968303,  0.57530831],
    [ 1.65370219, -0.36647173],
    [ 0.66569897,  1.77740068],
    [-0.37088553, -0.92379819],
    [-1.17757946, -0.25393047],
    [-1.624227  ,  0.71525192]])

    # Set dependant variable values
    Y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 
              0, 0, 1, 1, 1, 0, 0, 1, 
              0, 0, 1, 1], dtype=np.int32)
    
    # First fit model normally
    lr.fit(Z, Y)

    p = lr.predict_proba(Z)
    print(sklearn.metrics.log_loss(Y, p)) # ...

    print(lr.intercept_)
    print(lr.coef_)

    # Now fit model after a warm start
    lr.fit(Z, Y)

    p = lr.predict_proba(Z)
    print(sklearn.metrics.log_loss(Y, p)) # ...

    print(lr.intercept_)
    print(lr.coef_)



#### Expected Results
The predictions should be the same as the model converged the first time it was run.

#### Actual Results
The predictions are different. In fact the more times you re-run the fit the worse it gets. This is actually the only reason I was able to catch the bug. It is caused by the line here https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/logistic.py#L678.

     w0[:, :coef.shape[1]] = coef

As `coef` is `(1, n_features)`, but `w0` is `(2, n_features)`, this causes the `coef` value to be broadcast into the `w0`. This some sort of singularity issue when training resulting in worse performance. Note that had it not done exactly this i.e. `w0` was simply initialised by some random values, this bug would be very hard to catch because of course each time the model would converge just not as fast as one would hope when warm starting.

#### Further Information
The fix I believe is very easy, just need to swap the previous line to 

     if n_classes == 1:
         w0[0, :coef.shape[1]] = -coef  # Be careful to get these the right way around
         w0[1, :coef.shape[1]] = coef
     else:
         w0[:, :coef.shape[1]] = coef

#### Versions
<!--
Please run the following snippet and paste the output below.
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->
Linux-4.13.0-37-generic-x86_64-with-Ubuntu-16.04-xenial
Python 3.5.2 (default, Nov 23 2017, 16:37:01)
NumPy 1.14.2
SciPy 1.0.0
Scikit-Learn 0.20.dev0 (built from latest master)


<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 59| 462 | 462 | 988 | 
| 2 | **2 sklearn/linear_model/logistic.py** | 1267 | 1308| 426 | 888 | 17589 | 
| 3 | **2 sklearn/linear_model/logistic.py** | 956 | 1156| 2189 | 3077 | 17589 | 
| 4 | **2 sklearn/linear_model/logistic.py** | 1680 | 1735| 642 | 3719 | 17589 | 
| 5 | **2 sklearn/linear_model/logistic.py** | 1737 | 1792| 542 | 4261 | 17589 | 
| 6 | 3 sklearn/utils/estimator_checks.py | 1272 | 1379| 1122 | 5383 | 36879 | 
| **-> 7 <-** | **3 sklearn/linear_model/logistic.py** | 656 | 703| 639 | 6022 | 36879 | 
| 8 | 3 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 61 | 119| 519 | 6541 | 36879 | 
| 9 | 4 benchmarks/bench_glmnet.py | 47 | 129| 796 | 7337 | 37966 | 
| 10 | 5 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 80| 655 | 7992 | 38652 | 
| 11 | 6 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 312 | 8304 | 50819 | 
| 12 | 6 sklearn/utils/estimator_checks.py | 1701 | 1743| 449 | 8753 | 50819 | 
| 13 | **6 sklearn/linear_model/logistic.py** | 1368 | 1562| 2100 | 10853 | 50819 | 
| 14 | 7 examples/linear_model/plot_logistic_path.py | 1 | 56| 286 | 11139 | 51130 | 
| 15 | 8 sklearn/linear_model/randomized_l1.py | 507 | 538| 332 | 11471 | 56901 | 
| 16 | 9 sklearn/linear_model/least_angle.py | 207 | 365| 1584 | 13055 | 70622 | 
| 17 | 9 sklearn/linear_model/randomized_l1.py | 358 | 387| 296 | 13351 | 70622 | 
| 18 | **9 sklearn/linear_model/logistic.py** | 704 | 762| 706 | 14057 | 70622 | 
| 19 | 10 sklearn/linear_model/coordinate_descent.py | 1155 | 1227| 810 | 14867 | 91133 | 
| 20 | 10 sklearn/linear_model/coordinate_descent.py | 1800 | 1903| 949 | 15816 | 91133 | 
| 21 | 10 sklearn/linear_model/coordinate_descent.py | 8 | 29| 154 | 15970 | 91133 | 
| 22 | 10 sklearn/linear_model/least_angle.py | 367 | 489| 1228 | 17198 | 91133 | 
| 23 | **10 sklearn/linear_model/logistic.py** | 590 | 654| 740 | 17938 | 91133 | 
| 24 | 11 benchmarks/bench_rcv1_logreg_convergence.py | 6 | 31| 202 | 18140 | 93081 | 
| 25 | 11 sklearn/utils/estimator_checks.py | 1636 | 1674| 441 | 18581 | 93081 | 
| 26 | 11 sklearn/linear_model/coordinate_descent.py | 454 | 503| 588 | 19169 | 93081 | 
| 27 | **11 sklearn/linear_model/logistic.py** | 1158 | 1176| 202 | 19371 | 93081 | 
| 28 | 11 sklearn/utils/estimator_checks.py | 1766 | 1797| 340 | 19711 | 93081 | 
| 29 | 11 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 20084 | 93081 | 
| 30 | 12 sklearn/svm/base.py | 860 | 922| 643 | 20727 | 101135 | 
| 31 | 13 examples/linear_model/plot_lasso_model_selection.py | 92 | 156| 503 | 21230 | 102464 | 
| 32 | **13 sklearn/linear_model/logistic.py** | 1178 | 1266| 803 | 22033 | 102464 | 
| 33 | 14 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 22573 | 103988 | 
| 34 | 15 sklearn/linear_model/bayes.py | 485 | 533| 572 | 23145 | 109009 | 
| 35 | 15 sklearn/utils/estimator_checks.py | 746 | 777| 297 | 23442 | 109009 | 
| 36 | 15 sklearn/linear_model/coordinate_descent.py | 1395 | 1571| 1686 | 25128 | 109009 | 
| 37 | 15 sklearn/utils/estimator_checks.py | 1238 | 1269| 258 | 25386 | 109009 | 
| 38 | 15 sklearn/linear_model/coordinate_descent.py | 1597 | 1715| 1143 | 26529 | 109009 | 
| 39 | **15 sklearn/linear_model/logistic.py** | 1564 | 1584| 221 | 26750 | 109009 | 
| 40 | 15 sklearn/linear_model/randomized_l1.py | 390 | 506| 1055 | 27805 | 109009 | 
| 41 | **15 sklearn/linear_model/logistic.py** | 1 | 37| 219 | 28024 | 109009 | 
| 42 | 15 sklearn/linear_model/coordinate_descent.py | 1920 | 2077| 1534 | 29558 | 109009 | 
| 43 | 16 examples/text/plot_document_classification_20newsgroups.py | 250 | 328| 673 | 30231 | 111566 | 
| 44 | 16 sklearn/linear_model/stochastic_gradient.py | 1005 | 1037| 288 | 30519 | 111566 | 
| 45 | 17 examples/linear_model/plot_multi_task_lasso_support.py | 1 | 70| 587 | 31106 | 112178 | 
| 46 | 17 sklearn/linear_model/coordinate_descent.py | 1904 | 2097| 161 | 31267 | 112178 | 
| 47 | 17 sklearn/utils/estimator_checks.py | 1608 | 1633| 277 | 31544 | 112178 | 
| 48 | 18 examples/linear_model/plot_bayesian_ridge.py | 1 | 99| 771 | 32315 | 113098 | 
| 49 | 18 sklearn/linear_model/coordinate_descent.py | 506 | 649| 1391 | 33706 | 113098 | 
| 50 | 18 sklearn/linear_model/coordinate_descent.py | 805 | 926| 1125 | 34831 | 113098 | 
| 51 | 18 sklearn/linear_model/coordinate_descent.py | 383 | 452| 776 | 35607 | 113098 | 
| 52 | 18 examples/linear_model/plot_lasso_model_selection.py | 1 | 78| 691 | 36298 | 113098 | 
| 53 | 19 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 37612 | 114439 | 
| 54 | 19 sklearn/linear_model/coordinate_descent.py | 2224 | 2236| 181 | 37793 | 114439 | 
| 55 | 20 sklearn/covariance/graph_lasso_.py | 619 | 697| 821 | 38614 | 122623 | 
| 56 | 20 sklearn/linear_model/stochastic_gradient.py | 416 | 456| 383 | 38997 | 122623 | 
| 57 | 20 sklearn/linear_model/stochastic_gradient.py | 1179 | 1348| 1618 | 40615 | 122623 | 
| 58 | 20 sklearn/linear_model/coordinate_descent.py | 743 | 778| 365 | 40980 | 122623 | 
| 59 | 20 sklearn/utils/estimator_checks.py | 1 | 75| 659 | 41639 | 122623 | 
| 60 | 21 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 42398 | 124106 | 
| 61 | 22 examples/linear_model/plot_theilsen.py | 1 | 88| 785 | 43183 | 125110 | 
| 62 | 23 sklearn/multiclass.py | 375 | 410| 300 | 43483 | 131598 | 
| 63 | 23 sklearn/linear_model/bayes.py | 409 | 424| 198 | 43681 | 131598 | 
| 64 | 23 sklearn/linear_model/randomized_l1.py | 148 | 182| 314 | 43995 | 131598 | 
| 65 | 23 sklearn/linear_model/stochastic_gradient.py | 594 | 788| 1964 | 45959 | 131598 | 
| 66 | 24 sklearn/discriminant_analysis.py | 621 | 696| 719 | 46678 | 138229 | 
| 67 | 24 sklearn/linear_model/coordinate_descent.py | 1073 | 1154| 771 | 47449 | 138229 | 
| 68 | 25 sklearn/mixture/gmm.py | 490 | 581| 862 | 48311 | 145793 | 
| 69 | 25 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 48843 | 145793 | 
| 70 | **25 sklearn/linear_model/logistic.py** | 895 | 1363| 602 | 49445 | 145793 | 
| 71 | 25 sklearn/utils/estimator_checks.py | 1027 | 1095| 601 | 50046 | 145793 | 
| 72 | 25 sklearn/utils/estimator_checks.py | 1526 | 1568| 427 | 50473 | 145793 | 
| 73 | 26 sklearn/gaussian_process/gaussian_process.py | 692 | 805| 1032 | 51505 | 153330 | 
| 74 | 27 examples/neural_networks/plot_rbm_logistic_classification.py | 78 | 142| 488 | 51993 | 154421 | 
| 75 | 28 sklearn/__init__.py | 1 | 78| 659 | 52652 | 155213 | 
| 76 | 29 sklearn/linear_model/huber.py | 203 | 285| 762 | 53414 | 157748 | 
| 77 | 30 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 606 | 54020 | 158379 | 
| 78 | 30 sklearn/linear_model/randomized_l1.py | 337 | 355| 223 | 54243 | 158379 | 
| 79 | **30 sklearn/linear_model/logistic.py** | 402 | 423| 246 | 54489 | 158379 | 
| 80 | 30 sklearn/linear_model/coordinate_descent.py | 2100 | 2223| 1033 | 55522 | 158379 | 
| 81 | 30 sklearn/linear_model/bayes.py | 197 | 258| 671 | 56193 | 158379 | 
| 82 | 31 examples/preprocessing/plot_transformed_target.py | 1 | 94| 759 | 56952 | 160140 | 
| 83 | 31 sklearn/utils/estimator_checks.py | 780 | 795| 139 | 57091 | 160140 | 
| 84 | 32 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 1 | 80| 698 | 57789 | 160892 | 
| 85 | 33 examples/calibration/plot_compare_calibration.py | 1 | 78| 759 | 58548 | 162073 | 
| 86 | 33 sklearn/linear_model/coordinate_descent.py | 1380 | 1594| 208 | 58756 | 162073 | 
| 87 | 33 sklearn/linear_model/coordinate_descent.py | 1228 | 1243| 174 | 58930 | 162073 | 
| 88 | 33 sklearn/linear_model/coordinate_descent.py | 1246 | 1379| 1114 | 60044 | 162073 | 
| 89 | 33 sklearn/linear_model/stochastic_gradient.py | 1104 | 1365| 598 | 60642 | 162073 | 
| 90 | 33 sklearn/utils/estimator_checks.py | 1494 | 1523| 324 | 60966 | 162073 | 
| 91 | 33 sklearn/linear_model/randomized_l1.py | 541 | 567| 269 | 61235 | 162073 | 
| 92 | 34 sklearn/utils/fixes.py | 74 | 150| 754 | 61989 | 164470 | 
| 93 | 35 sklearn/svm/classes.py | 15 | 172| 1630 | 63619 | 175458 | 
| 94 | 35 sklearn/linear_model/huber.py | 125 | 201| 711 | 64330 | 175458 | 
| 95 | 36 examples/calibration/plot_calibration_multiclass.py | 122 | 169| 589 | 64919 | 177591 | 
| 96 | 37 benchmarks/bench_sparsify.py | 1 | 82| 754 | 65673 | 178498 | 
| 97 | 37 examples/calibration/plot_calibration_multiclass.py | 1 | 80| 764 | 66437 | 178498 | 
| 98 | **37 sklearn/linear_model/logistic.py** | 1586 | 1679| 782 | 67219 | 178498 | 
| 99 | 38 sklearn/linear_model/__init__.py | 1 | 87| 709 | 67928 | 179207 | 
| 100 | 38 sklearn/linear_model/randomized_l1.py | 313 | 335| 239 | 68167 | 179207 | 
| 101 | 39 examples/linear_model/plot_ols_ridge_variance.py | 1 | 72| 526 | 68693 | 179741 | 
| 102 | 40 sklearn/ensemble/gradient_boosting.py | 1017 | 1076| 579 | 69272 | 198005 | 
| 103 | 41 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 70044 | 199170 | 


### Hint

```
Thanks for the report. At a glance, that looks very plausible.. Test and patch welcome
I'm happy to do this although would be interested in opinions on the test. I could do either

1) Test what causes the bug above i.e. the model doesn't converge when warm starting.
2) Test that the initial `w0` used in `logistic_regression_path` is the same as the previous `w0` after the function has been run i.e. that warm starting is happening as expected.

The pros of (1) are that its quick and easy however as mentioned previously it doesn't really get to the essence of what is causing the bug. The only reason it is failing is because the `w0` is getting initialised so that the rows are exactly identical. If this were not the case but the rows also weren't warm started correctly (i.e. just randomly initialised), the model would still converge (just slower than one would hope if a good warm start had been used) and the test would unfortunately pass.

The pros of (2) are that it would correctly test that the warm starting occurred but the cons would be I don't know how I would do it as the `w0` is not available outside of the `logistic_regression_path` function.
Go for the simplest test first, open a PR and see where that leads you!
```

## Patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -675,7 +675,13 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                     'shape (%d, %d) or (%d, %d)' % (
                         coef.shape[0], coef.shape[1], classes.size,
                         n_features, classes.size, n_features + 1))
-            w0[:, :coef.shape[1]] = coef
+
+            if n_classes == 1:
+                w0[0, :coef.shape[1]] = -coef
+                w0[1, :coef.shape[1]] = coef
+            else:
+                w0[:, :coef.shape[1]] = coef
+
 
     if multi_class == 'multinomial':
         # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_logistic.py b/sklearn/linear_model/tests/test_logistic.py
--- a/sklearn/linear_model/tests/test_logistic.py
+++ b/sklearn/linear_model/tests/test_logistic.py
@@ -7,6 +7,7 @@
 from sklearn.preprocessing import LabelEncoder
 from sklearn.utils import compute_class_weight
 from sklearn.utils.testing import assert_almost_equal
+from sklearn.utils.testing import assert_allclose
 from sklearn.utils.testing import assert_array_almost_equal
 from sklearn.utils.testing import assert_array_equal
 from sklearn.utils.testing import assert_equal
@@ -1192,3 +1193,23 @@ def test_dtype_match():
             lr_64.fit(X_64, y_64)
             assert_equal(lr_64.coef_.dtype, X_64.dtype)
             assert_almost_equal(lr_32.coef_, lr_64.coef_.astype(np.float32))
+
+
+def test_warm_start_converge_LR():
+    # Test to see that the logistic regression converges on warm start,
+    # with multi_class='multinomial'. Non-regressive test for #10836
+
+    rng = np.random.RandomState(0)
+    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
+    y = np.array([1] * 100 + [-1] * 100)
+    lr_no_ws = LogisticRegression(multi_class='multinomial',
+                                  solver='sag', warm_start=False)
+    lr_ws = LogisticRegression(multi_class='multinomial',
+                               solver='sag', warm_start=True)
+
+    lr_no_ws_loss = log_loss(y, lr_no_ws.fit(X, y).predict_proba(X))
+    lr_ws_loss = [log_loss(y, lr_ws.fit(X, y).predict_proba(X)) 
+                 for _ in range(5)]
+
+    for i in range(5):
+        assert_allclose(lr_no_ws_loss, lr_ws_loss[i], rtol=1e-5)

```


## Code snippets

### 1 - examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py:

Start line: 1, End line: 59

```python
"""
=====================================================
Multiclass sparse logisitic regression on newgroups20
=====================================================

Comparison of multinomial logistic L1 vs one-versus-rest L1 logistic regression
to classify documents from the newgroups20 dataset. Multinomial logistic
regression yields more accurate results and is faster to train on the larger
scale dataset.

Here we use the l1 sparsity that trims the weights of not informative
features to zero. This is good if the goal is to extract the strongly
discriminative vocabulary of each class. If the goal is to get the best
predictive accuracy, it is better to use the non sparsity-inducing l2 penalty
instead.

A more traditional (and possibly better) way to predict on a sparse subset of
input features would be to use univariate feature selection followed by a
traditional (l2-penalised) logistic regression model.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print(__doc__)

t0 = time.clock()

# We use SAGA solver
solver = 'saga'

# Turn down for faster run time
n_samples = 10000

# Memorized fetch_rcv1 for faster access
dataset = fetch_20newsgroups_vectorized('all')
X = dataset.data
y = dataset.target
X = X[:n_samples]
y = y[:n_samples]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    stratify=y,
                                                    test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, n_features, n_classes))

models = {'ovr': {'name': 'One versus Rest', 'iters': [1, 3]},
          'multinomial': {'name': 'Multinomial', 'iters': [1, 3, 7]}}
```
### 2 - sklearn/linear_model/logistic.py:

Start line: 1267, End line: 1308

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code
        if self.multi_class == 'multinomial':
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ['sag', 'saga']:
            backend = 'threading'
        else:
            backend = 'multiprocessing'
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, tol=self.tol,
                      verbose=self.verbose, solver=self.solver,
                      multi_class=self.multi_class, max_iter=self.max_iter,
                      class_weight=self.class_weight, check_input=False,
                      random_state=self.random_state, coef=warm_start_coef_,
                      penalty=self.penalty,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if self.multi_class == 'multinomial':
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(n_classes, n_features +
                                            int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        return self
```
### 3 - sklearn/linear_model/logistic.py:

Start line: 956, End line: 1156

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme if the 'multi_class' option is set to 'ovr', and uses the cross-
    entropy loss if the 'multi_class' option is set to 'multinomial'.
    (Currently the 'multinomial' option is supported only by the 'lbfgs',
    'sag' and 'newton-cg' solvers.)

    This class implements regularized logistic regression using the
    'liblinear' library, 'newton-cg', 'sag' and 'lbfgs' solvers. It can handle
    both dense and sparse input. Use C-ordered arrays or CSR matrices
    containing 64-bit floats for optimal performance; any other input format
    will be converted (and copied).

    The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
    with primal formulation. The 'liblinear' solver supports both L1 and L2
    regularization, with a dual formulation only for the L2 penalty.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    penalty : str, 'l1' or 'l2', default: 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

        .. versionadded:: 0.19
           l1 penalty with SAGA solver (allowing 'multinomial' + L1)

    dual : bool, default: False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default: 1e-4
        Tolerance for stopping criteria.

    C : float, default: 1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default: None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           *class_weight='balanced'*

    random_state : int, RandomState instance or None, optional, default: None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Used when ``solver`` == 'sag' or
        'liblinear'.

    solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'liblinear'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
          'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
          handle multinomial loss; 'liblinear' is limited to one-versus-rest
          schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
          'liblinear' and 'saga' handle L1 penalty.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    max_iter : int, default: 100
        Useful only for the newton-cg, sag and lbfgs solvers.
        Maximum number of iterations taken for the solvers to converge.

    multi_class : str, {'ovr', 'multinomial'}, default: 'ovr'
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for 'liblinear'
        solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    verbose : int, default: 0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default: False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* to support *lbfgs*, *newton-cg*, *sag*, *saga* solvers.

    n_jobs : int, default: 1
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver``is set
        to 'liblinear' regardless of whether 'multi_class' is specified or
        not. If given a value of -1, all cores are used.

    Attributes
    ----------

    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `intercept_`
        corresponds to outcome 1 (True) and `-intercept_` corresponds to
        outcome 0 (False).

    n_iter_ : array, shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

        .. versionchanged:: 0.20

            In SciPy <= 1.0.0 the number of lbfgs iterations may exceed
            ``max_iter``. ``n_iter_`` will now report at most ``max_iter``.

    See also
    --------
    SGDClassifier : incrementally trained logistic regression (when given
        the parameter ``loss="log"``).
    sklearn.svm.LinearSVC : learns SVM models using the same algorithm.
    LogisticRegressionCV : Logistic regression with built-in cross validation

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    Predict output may not match that of standalone liblinear in certain
    cases. See :ref:`differences from liblinear <liblinear_differences>`
    in the narrative documentation.

    References
    ----------

    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
        Minimizing Finite Sums with the Stochastic Average Gradient
        https://hal.inria.fr/hal-00860051/document

    SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
        SAGA: A Fast Incremental Gradient Method With Support
        for Non-Strongly Convex Composite Objectives
        https://arxiv.org/abs/1407.0202

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    """
```
### 4 - sklearn/linear_model/logistic.py:

Start line: 1680, End line: 1735

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code
        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                               backend=backend)(
            path_func(X, y, train, test, pos_class=label, Cs=self.Cs,
                      fit_intercept=self.fit_intercept, penalty=self.penalty,
                      dual=self.dual, solver=self.solver, tol=self.tol,
                      max_iter=self.max_iter, verbose=self.verbose,
                      class_weight=class_weight, scoring=self.scoring,
                      multi_class=self.multi_class,
                      intercept_scaling=self.intercept_scaling,
                      random_state=self.random_state,
                      max_squared_sum=max_squared_sum,
                      sample_weight=sample_weight
                      )
            for label in iter_encoded_labels
            for train, test in folds)

        if self.multi_class == 'multinomial':
            multi_coefs_paths, Cs, multi_scores, n_iter_ = zip(*fold_coefs_)
            multi_coefs_paths = np.asarray(multi_coefs_paths)
            multi_scores = np.asarray(multi_scores)

            # This is just to maintain API similarity between the ovr and
            # multinomial option.
            # Coefs_paths in now n_folds X len(Cs) X n_classes X n_features
            # we need it to be n_classes X len(Cs) X n_folds X n_features
            # to be similar to "ovr".
            coefs_paths = np.rollaxis(multi_coefs_paths, 2, 0)

            # Multinomial has a true score across all labels. Hence the
            # shape is n_folds X len(Cs). We need to repeat this score
            # across all labels for API similarity.
            scores = np.tile(multi_scores, (n_classes, 1, 1))
            self.Cs_ = Cs[0]
            self.n_iter_ = np.reshape(n_iter_, (1, len(folds),
                                                len(self.Cs_)))

        else:
            coefs_paths, Cs, scores, n_iter_ = zip(*fold_coefs_)
            self.Cs_ = Cs[0]
            coefs_paths = np.reshape(coefs_paths, (n_classes, len(folds),
                                                   len(self.Cs_), -1))
            self.n_iter_ = np.reshape(n_iter_, (n_classes, len(folds),
                                                len(self.Cs_)))

        self.coefs_paths_ = dict(zip(classes, coefs_paths))
        scores = np.reshape(scores, (n_classes, len(folds), -1))
        self.scores_ = dict(zip(classes, scores))

        self.C_ = list()
        self.coef_ = np.empty((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)

        # hack to iterate only once for multinomial case.
        if self.multi_class == 'multinomial':
            scores = multi_scores
            coefs_paths = multi_coefs_paths
        # ... other code
```
### 5 - sklearn/linear_model/logistic.py:

Start line: 1737, End line: 1792

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        # ... other code

        for index, (cls, encoded_label) in enumerate(
                zip(iter_classes, iter_encoded_labels)):

            if self.multi_class == 'ovr':
                # The scores_ / coefs_paths_ dict have unencoded class
                # labels as their keys
                scores = self.scores_[cls]
                coefs_paths = self.coefs_paths_[cls]

            if self.refit:
                best_index = scores.sum(axis=0).argmax()

                C_ = self.Cs_[best_index]
                self.C_.append(C_)
                if self.multi_class == 'multinomial':
                    coef_init = np.mean(coefs_paths[:, best_index, :, :],
                                        axis=0)
                else:
                    coef_init = np.mean(coefs_paths[:, best_index, :], axis=0)

                # Note that y is label encoded and hence pos_class must be
                # the encoded label / None (for 'multinomial')
                w, _, _ = logistic_regression_path(
                    X, y, pos_class=encoded_label, Cs=[C_], solver=self.solver,
                    fit_intercept=self.fit_intercept, coef=coef_init,
                    max_iter=self.max_iter, tol=self.tol,
                    penalty=self.penalty,
                    class_weight=class_weight,
                    multi_class=self.multi_class,
                    verbose=max(0, self.verbose - 1),
                    random_state=self.random_state,
                    check_input=False, max_squared_sum=max_squared_sum,
                    sample_weight=sample_weight)
                w = w[0]

            else:
                # Take the best scores across every fold and the average of all
                # coefficients corresponding to the best scores.
                best_indices = np.argmax(scores, axis=1)
                w = np.mean([coefs_paths[i][best_indices[i]]
                             for i in range(len(folds))], axis=0)
                self.C_.append(np.mean(self.Cs_[best_indices]))

            if self.multi_class == 'multinomial':
                self.C_ = np.tile(self.C_, n_classes)
                self.coef_ = w[:, :X.shape[1]]
                if self.fit_intercept:
                    self.intercept_ = w[:, -1]
            else:
                self.coef_[index] = w[: X.shape[1]]
                if self.fit_intercept:
                    self.intercept_[index] = w[-1]

        self.C_ = np.asarray(self.C_)
        return self
```
### 6 - sklearn/utils/estimator_checks.py:

Start line: 1272, End line: 1379

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
        with assert_raises(ValueError, msg="The classifier {} does not"
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
### 7 - sklearn/linear_model/logistic.py:

Start line: 656, End line: 703

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    # ... other code

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == 'ovr':
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    'Initialization coef is of shape %d, expected shape '
                    '%d or %d' % (coef.size, n_features, w0.size))
            w0[:coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if (coef.shape[0] != n_classes or
                    coef.shape[1] not in (n_features, n_features + 1)):
                raise ValueError(
                    'Initialization coef is of shape (%d, %d), expected '
                    'shape (%d, %d) or (%d, %d)' % (
                        coef.shape[0], coef.shape[1], classes.size,
                        n_features, classes.size, n_features + 1))
            w0[:, :coef.shape[1]] = coef

    if multi_class == 'multinomial':
        # fmin_l_bfgs_b and newton-cg accepts only ravelled parameters.
        if solver in ['lbfgs', 'newton-cg']:
            w0 = w0.ravel()
        target = Y_multi
        if solver == 'lbfgs':
            func = lambda x, *args: _multinomial_loss_grad(x, *args)[0:2]
        elif solver == 'newton-cg':
            func = lambda x, *args: _multinomial_loss(x, *args)[0]
            grad = lambda x, *args: _multinomial_loss_grad(x, *args)[1]
            hess = _multinomial_grad_hess
        warm_start_sag = {'coef': w0.T}
    else:
        target = y_bin
        if solver == 'lbfgs':
            func = _logistic_loss_and_grad
        elif solver == 'newton-cg':
            func = _logistic_loss
            grad = lambda x, *args: _logistic_loss_and_grad(x, *args)[1]
            hess = _logistic_grad_hess
        warm_start_sag = {'coef': np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    # ... other code
```
### 8 - examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py:

Start line: 61, End line: 119

```python
for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params['iters']:
        print('[model=%s, solver=%s] Number of epochs: %s' %
              (model_params['name'], solver, this_max_iter))
        lr = LogisticRegression(solver=solver,
                                multi_class=model,
                                C=1,
                                penalty='l1',
                                fit_intercept=True,
                                max_iter=this_max_iter,
                                random_state=42,
                                )
        t1 = time.clock()
        lr.fit(X_train, y_train)
        train_time = time.clock() - t1

        y_pred = lr.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        density = np.mean(lr.coef_ != 0, axis=1) * 100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]['times'] = times
    models[model]['densities'] = densities
    models[model]['accuracies'] = accuracies
    print('Test accuracy for model %s: %.4f' % (model, accuracies[-1]))
    print('%% non-zero coefficients for model %s, '
          'per class:\n %s' % (model, densities[-1]))
    print('Run time (%i epochs) for model %s:'
          '%.2f' % (model_params['iters'][-1], model, times[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]['name']
    times = models[model]['times']
    accuracies = models[model]['accuracies']
    ax.plot(times, accuracies, marker='o',
            label='Model: %s' % name)
    ax.set_xlabel('Train time (s)')
    ax.set_ylabel('Test accuracy')
ax.legend()
fig.suptitle('Multinomial vs One-vs-Rest Logistic L1\n'
             'Dataset %s' % '20newsgroups')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = time.clock() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```
### 9 - benchmarks/bench_glmnet.py:

Start line: 47, End line: 129

```python
if __name__ == '__main__':
    from glmnet.elastic_net import Lasso as GlmnetLasso
    from sklearn.linear_model import Lasso as ScikitLasso
    # Delayed import of matplotlib.pyplot
    import matplotlib.pyplot as plt

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 500
    n_features = 1000
    n_informative = n_features / 10
    n_test_samples = 1000
    for i in range(1, n + 1):
        print('==================')
        print('Iteration %s of %s' % (i, n))
        print('==================')

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples, n_features=n_features,
            noise=0.1, n_informative=n_informative, coef=True)

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:(i * step)]
        Y = Y[:(i * step)]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    plt.clf()
    xx = range(0, n * step, step)
    plt.title('Lasso regression on sample dataset (%d features)' % n_features)
    plt.plot(xx, scikit_results, 'b-', label='scikit-learn')
    plt.plot(xx, glmnet_results, 'r-', label='glmnet')
    plt.legend()
    plt.xlabel('number of samples to classify')
    plt.ylabel('Time (s)')
    plt.show()

    # now do a benchmark where the number of points is fixed
    # and the variable is the number of features

    scikit_results = []
    glmnet_results = []
    n = 20
    step = 100
    n_samples = 500

    for i in range(1, n + 1):
        print('==================')
        print('Iteration %02d of %02d' % (i, n))
        print('==================')
        n_features = i * step
        n_informative = n_features / 10

        X, Y, coef_ = make_regression(
            n_samples=(i * step) + n_test_samples, n_features=n_features,
            noise=0.1, n_informative=n_informative, coef=True)

        X_test = X[-n_test_samples:]
        Y_test = Y[-n_test_samples:]
        X = X[:n_samples]
        Y = Y[:n_samples]

        print("benchmarking scikit-learn: ")
        scikit_results.append(bench(ScikitLasso, X, Y, X_test, Y_test, coef_))
        print("benchmarking glmnet: ")
        glmnet_results.append(bench(GlmnetLasso, X, Y, X_test, Y_test, coef_))

    xx = np.arange(100, 100 + n * step, step)
    plt.figure('scikit-learn vs. glmnet benchmark results')
    plt.title('Regression in high dimensional spaces (%d samples)' % n_samples)
    plt.plot(xx, scikit_results, 'b-', label='scikit-learn')
    plt.plot(xx, glmnet_results, 'r-', label='glmnet')
    plt.legend()
    plt.xlabel('number of features')
    plt.ylabel('Time (s)')
    plt.axis('tight')
    plt.show()
```
### 10 - examples/linear_model/plot_sparse_logistic_regression_mnist.py:

Start line: 1, End line: 80

```python
"""
=====================================================
MNIST classfification using multinomial logistic + L1
=====================================================

Here we fit a multinomial logistic regression with L1 penalty on a subset of
the MNIST digits classification task. We use the SAGA algorithm for this
purpose: this a solver that is fast when the number of samples is significantly
larger than the number of features and is able to finely optimize non-smooth
objective functions which is the case with the l1-penalty. Test accuracy
reaches > 0.8, while weight vectors remains *sparse* and therefore more easily
*interpretable*.

Note that this accuracy of this l1-penalized linear model is significantly
below what can be reached by an l2-penalized linear model or a non-linear
multi-layer perceptron model on this dataset.

"""
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

print(__doc__)
t0 = time.time()
train_samples = 5000

mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(C=50. / train_samples,
                         multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```
### 13 - sklearn/linear_model/logistic.py:

Start line: 1368, End line: 1562

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):
    """Logistic Regression CV (aka logit, MaxEnt) classifier.

    This class implements logistic regression using liblinear, newton-cg, sag
    of lbfgs optimizer. The newton-cg, sag and lbfgs solvers support only L2
    regularization with primal formulation. The liblinear solver supports both
    L1 and L2 regularization, with a dual formulation only for the L2 penalty.

    For the grid of Cs values (that are set by default to be ten values in
    a logarithmic scale between 1e-4 and 1e4), the best hyperparameter is
    selected by the cross-validator StratifiedKFold, but it can be changed
    using the cv parameter. In the case of newton-cg and lbfgs solvers,
    we warm start along the path i.e guess the initial coefficients of the
    present fit to be the coefficients got after convergence in the previous
    fit, so it is supposed to be faster for high-dimensional dense data.

    For a multiclass problem, the hyperparameters for each class are computed
    using the best scores got by doing a one-vs-rest in parallel across all
    folds and classes. Hence this is not the true multinomial loss.

    Read more in the :ref:`User Guide <logistic_regression>`.

    Parameters
    ----------
    Cs : list of floats | int
        Each of the values in Cs describes the inverse of regularization
        strength. If Cs is as an int, then a grid of Cs values are chosen
        in a logarithmic scale between 1e-4 and 1e4.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    cv : integer or cross-validation generator
        The default cross-validation generator used is Stratified K-Folds.
        If an integer is provided, then it is the number of folds used.
        See the module :mod:`sklearn.model_selection` module for the
        list of possible cross-validation objects.

    dual : bool
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    penalty : str, 'l1' or 'l2'
        Used to specify the norm used in the penalization. The 'newton-cg',
        'sag' and 'lbfgs' solvers support only l2 penalties.

    scoring : string, callable, or None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``. For a list of scoring functions
        that can be used, look at :mod:`sklearn.metrics`. The
        default scoring option used is 'accuracy'.

    solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
        default: 'lbfgs'
        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
          'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
          handle multinomial loss; 'liblinear' is limited to one-versus-rest
          schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
          'liblinear' and 'saga' handle L1 penalty.
        - 'liblinear' might be slower in LogisticRegressionCV because it does
          not handle warm-starting.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can preprocess the data
        with a scaler from sklearn.preprocessing.

        .. versionadded:: 0.17
           Stochastic Average Gradient descent solver.
        .. versionadded:: 0.19
           SAGA solver.

    tol : float, optional
        Tolerance for stopping criteria.

    max_iter : int, optional
        Maximum number of iterations of the optimization algorithm.

    class_weight : dict or 'balanced', optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

        .. versionadded:: 0.17
           class_weight == 'balanced'

    n_jobs : int, optional
        Number of CPU cores used during the cross-validation loop. If given
        a value of -1, all cores are used.

    verbose : int
        For the 'liblinear', 'sag' and 'lbfgs' solvers set verbose to any
        positive number for verbosity.

    refit : bool
        If set to True, the scores are averaged across all folds, and the
        coefs and the C that corresponds to the best score is taken, and a
        final refit is done using these parameters.
        Otherwise the coefs, intercepts and C that correspond to the
        best scores across folds are averaged.

    intercept_scaling : float, default 1.
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    multi_class : str, {'ovr', 'multinomial'}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option
        chosen is 'ovr', then a binary problem is fit for each label. Else
        the loss minimised is the multinomial loss fit across
        the entire probability distribution. Does not work for 'liblinear'
        solver.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    coef_ : array, shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem
        is binary.

    intercept_ : array, shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape(1,) when the problem is binary.

    Cs_ : array
        Array of C i.e. inverse of regularization parameter values used
        for cross-validation.

    coefs_paths_ : array, shape ``(n_folds, len(Cs_), n_features)`` or \
                   ``(n_folds, len(Cs_), n_features + 1)``
        dict with classes as the keys, and the path of coefficients obtained
        during cross-validating across each fold and then across each Cs
        after doing an OvR for the corresponding class as values.
        If the 'multi_class' option is set to 'multinomial', then
        the coefs_paths are the coefficients corresponding to each class.
        Each dict value has shape ``(n_folds, len(Cs_), n_features)`` or
        ``(n_folds, len(Cs_), n_features + 1)`` depending on whether the
        intercept is fit or not.

    scores_ : dict
        dict with classes as the keys, and the values as the
        grid of scores obtained during cross-validating each fold, after doing
        an OvR for the corresponding class. If the 'multi_class' option
        given is 'multinomial' then the same scores are repeated across
        all classes, since this is the multinomial class.
        Each dict value has shape (n_folds, len(Cs))

    C_ : array, shape (n_classes,) or (n_classes - 1,)
        Array of C that maps to the best scores across every class. If refit is
        set to False, then for each class, the best C is the average of the
        C's that correspond to the best scores for each fold.
        `C_` is of shape(n_classes,) when the problem is binary.

    n_iter_ : array, shape (n_classes, n_folds, n_cs) or (1, n_folds, n_cs)
        Actual number of iterations for all classes, folds and Cs.
        In the binary or multinomial cases, the first dimension is equal to 1.

    See also
    --------
    LogisticRegression

    """
```
### 18 - sklearn/linear_model/logistic.py:

Start line: 704, End line: 762

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    # ... other code
    for i, C in enumerate(Cs):
        if solver == 'lbfgs':
            w0, loss, info = optimize.fmin_l_bfgs_b(
                func, w0, fprime=None,
                args=(X, target, 1. / C, sample_weight),
                iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
            if info["warnflag"] == 1:
                warnings.warn("lbfgs failed to converge. Increase the number "
                              "of iterations.", ConvergenceWarning)
            # In scipy <= 1.0.0, nit may exceed maxiter.
            # See https://github.com/scipy/scipy/issues/7854.
            n_iter_i = min(info['nit'], max_iter)
        elif solver == 'newton-cg':
            args = (X, target, 1. / C, sample_weight)
            w0, n_iter_i = newton_cg(hess, func, grad, w0, args=args,
                                     maxiter=max_iter, tol=tol)
        elif solver == 'liblinear':
            coef_, intercept_, n_iter_i, = _fit_liblinear(
                X, target, C, fit_intercept, intercept_scaling, None,
                penalty, dual, verbose, max_iter, tol, random_state,
                sample_weight=sample_weight)
            if fit_intercept:
                w0 = np.concatenate([coef_.ravel(), intercept_])
            else:
                w0 = coef_.ravel()

        elif solver in ['sag', 'saga']:
            if multi_class == 'multinomial':
                target = target.astype(np.float64)
                loss = 'multinomial'
            else:
                loss = 'log'
            if penalty == 'l1':
                alpha = 0.
                beta = 1. / C
            else:
                alpha = 1. / C
                beta = 0.
            w0, n_iter_i, warm_start_sag = sag_solver(
                X, target, sample_weight, loss, alpha,
                beta, max_iter, tol,
                verbose, random_state, False, max_squared_sum, warm_start_sag,
                is_saga=(solver == 'saga'))

        else:
            raise ValueError("solver must be one of {'liblinear', 'lbfgs', "
                             "'newton-cg', 'sag'}, got '%s' instead" % solver)

        if multi_class == 'multinomial':
            multi_w0 = np.reshape(w0, (classes.size, -1))
            if classes.size == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0)
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return coefs, np.array(Cs), n_iter
```
### 23 - sklearn/linear_model/logistic.py:

Start line: 590, End line: 654

```python
def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                             max_iter=100, tol=1e-4, verbose=0,
                             solver='lbfgs', coef=None,
                             class_weight=None, dual=False, penalty='l2',
                             intercept_scaling=1., multi_class='ovr',
                             random_state=None, check_input=True,
                             max_squared_sum=None, sample_weight=None):
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    _check_solver_option(solver, multi_class, penalty, dual)

    # Preprocessing.
    if check_input:
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape
    classes = np.unique(y)
    random_state = check_random_state(random_state)

    if pos_class is None and multi_class != 'multinomial':
        if (classes.size > 2):
            raise ValueError('To fit OvR, use the pos_class argument')
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    if sample_weight is not None:
        sample_weight = np.array(sample_weight, dtype=X.dtype, order='C')
        check_consistent_length(y, sample_weight)
    else:
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == 'multinomial':
        class_weight_ = compute_class_weight(class_weight, classes, y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == 'ovr':
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = (y == pos_class)
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(class_weight, mask_classes,
                                                 y_bin)
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver not in ['sag', 'saga']:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros((classes.size, n_features + int(fit_intercept)),
                      order='F', dtype=X.dtype)
    # ... other code
```
### 27 - sklearn/linear_model/logistic.py:

Start line: 1158, End line: 1176

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
```
### 32 - sklearn/linear_model/logistic.py:

Start line: 1178, End line: 1266

```python
class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
        """
        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        if self.solver in ['newton-cg']:
            _dtype = [np.float64, np.float32]
        else:
            _dtype = np.float64

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=_dtype,
                         order="C")
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        _check_solver_option(self.solver, self.multi_class, self.penalty,
                             self.dual)

        if self.solver == 'liblinear':
            if self.n_jobs != 1:
                warnings.warn("'n_jobs' > 1 does not have any effect when"
                              " 'solver' is set to 'liblinear'. Got 'n_jobs'"
                              " = {}.".format(self.n_jobs))
            self.coef_, self.intercept_, n_iter_ = _fit_liblinear(
                X, y, self.C, self.fit_intercept, self.intercept_scaling,
                self.class_weight, self.penalty, self.dual, self.verbose,
                self.max_iter, self.tol, self.random_state,
                sample_weight=sample_weight)
            self.n_iter_ = np.array([n_iter_])
            return self

        if self.solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef,
                                        self.intercept_[:, np.newaxis],
                                        axis=1)

        self.coef_ = list()
        self.intercept_ = np.zeros(n_classes)

        # Hack so that we iterate only once for the multinomial case.
        # ... other code
```
### 39 - sklearn/linear_model/logistic.py:

Start line: 1564, End line: 1584

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def __init__(self, Cs=10, fit_intercept=True, cv=None, dual=False,
                 penalty='l2', scoring=None, solver='lbfgs', tol=1e-4,
                 max_iter=100, class_weight=None, n_jobs=1, verbose=0,
                 refit=True, intercept_scaling=1., multi_class='ovr',
                 random_state=None):
        self.Cs = Cs
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.dual = dual
        self.penalty = penalty
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.solver = solver
        self.refit = refit
        self.intercept_scaling = intercept_scaling
        self.multi_class = multi_class
        self.random_state = random_state
```
### 41 - sklearn/linear_model/logistic.py:

Start line: 1, End line: 37

```python
"""
Logistic Regression
"""

import numbers
import warnings

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit

from .base import LinearClassifierMixin, SparseCoefMixin, BaseEstimator
from .sag import sag_solver
from ..preprocessing import LabelEncoder, LabelBinarizer
from ..svm.base import _fit_liblinear
from ..utils import check_array, check_consistent_length, compute_class_weight
from ..utils import check_random_state
from ..utils.extmath import (log_logistic, safe_sparse_dot, softmax,
                             squared_norm)
from ..utils.extmath import row_norms
from ..utils.fixes import logsumexp
from ..utils.optimize import newton_cg
from ..utils.validation import check_X_y
from ..exceptions import NotFittedError, ConvergenceWarning
from ..utils.multiclass import check_classification_targets
from ..externals.joblib import Parallel, delayed
from ..model_selection import check_cv
from ..externals import six
from ..metrics import get_scorer
```
### 70 - sklearn/linear_model/logistic.py:

Start line: 895, End line: 1363

```python
def _log_reg_scoring_path(X, y, train, test, pos_class=None, Cs=10,
                          scoring=None, fit_intercept=False,
                          max_iter=100, tol=1e-4, class_weight=None,
                          verbose=0, solver='lbfgs', penalty='l2',
                          dual=False, intercept_scaling=1.,
                          multi_class='ovr', random_state=None,
                          max_squared_sum=None, sample_weight=None):
    _check_solver_option(solver, multi_class, penalty, dual)

    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test = y[test]

    if sample_weight is not None:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y, sample_weight)

        sample_weight = sample_weight[train]

    coefs, Cs, n_iter = logistic_regression_path(
        X_train, y_train, Cs=Cs, fit_intercept=fit_intercept,
        solver=solver, max_iter=max_iter, class_weight=class_weight,
        pos_class=pos_class, multi_class=multi_class,
        tol=tol, verbose=verbose, dual=dual, penalty=penalty,
        intercept_scaling=intercept_scaling, random_state=random_state,
        check_input=False, max_squared_sum=max_squared_sum,
        sample_weight=sample_weight)

    log_reg = LogisticRegression(fit_intercept=fit_intercept)

    # The score method of Logistic Regression has a classes_ attribute.
    if multi_class == 'ovr':
        log_reg.classes_ = np.array([-1, 1])
    elif multi_class == 'multinomial':
        log_reg.classes_ = np.unique(y_train)
    else:
        raise ValueError("multi_class should be either multinomial or ovr, "
                         "got %d" % multi_class)

    if pos_class is not None:
        mask = (y_test == pos_class)
        y_test = np.ones(y_test.shape, dtype=np.float64)
        y_test[~mask] = -1.

    scores = list()

    if isinstance(scoring, six.string_types):
        scoring = get_scorer(scoring)
    for w in coefs:
        if multi_class == 'ovr':
            w = w[np.newaxis, :]
        if fit_intercept:
            log_reg.coef_ = w[:, :-1]
            log_reg.intercept_ = w[:, -1]
        else:
            log_reg.coef_ = w
            log_reg.intercept_ = 0.

        if scoring is None:
            scores.append(log_reg.score(X_test, y_test))
        else:
            scores.append(scoring(log_reg, X_test, y_test))
    return coefs, Cs, np.array(scores), n_iter


class LogisticRegression(BaseEstimator, LinearClassifierMixin,
                         SparseCoefMixin):
```
### 79 - sklearn/linear_model/logistic.py:

Start line: 402, End line: 423

```python
def _multinomial_grad_hess(w, X, Y, alpha, sample_weight):
    # ... other code
    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        r_yhat *= sample_weight
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()

    return grad, hessp
```
### 98 - sklearn/linear_model/logistic.py:

Start line: 1586, End line: 1679

```python
class LogisticRegressionCV(LogisticRegression, BaseEstimator,
                           LinearClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
        """
        _check_solver_option(self.solver, self.multi_class, self.penalty,
                             self.dual)

        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError("Maximum number of iteration must be positive;"
                             " got (max_iter=%r)" % self.max_iter)
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError("Tolerance for stopping criteria must be "
                             "positive; got (tol=%r)" % self.tol)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        check_classification_targets(y)

        class_weight = self.class_weight

        # Encode for string labels
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        if isinstance(class_weight, dict):
            class_weight = dict((label_encoder.transform([cls])[0], v)
                                for cls, v in class_weight.items())

        # The original class labels
        classes = self.classes_ = label_encoder.classes_
        encoded_labels = label_encoder.transform(label_encoder.classes_)

        if self.solver in ['sag', 'saga']:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        # init cross-validation generator
        cv = check_cv(self.cv, y, classifier=True)
        folds = list(cv.split(X, y))

        # Use the label encoded classes
        n_classes = len(encoded_labels)

        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes[0])

        if n_classes == 2:
            # OvR in case of binary problems is as good as fitting
            # the higher label
            n_classes = 1
            encoded_labels = encoded_labels[1:]
            classes = classes[1:]

        # We need this hack to iterate only once over labels, in the case of
        # multi_class = multinomial, without changing the value of the labels.
        if self.multi_class == 'multinomial':
            iter_encoded_labels = iter_classes = [None]
        else:
            iter_encoded_labels = encoded_labels
            iter_classes = classes

        # compute the class weights for the entire dataset y
        if class_weight == "balanced":
            class_weight = compute_class_weight(class_weight,
                                                np.arange(len(self.classes_)),
                                                y)
            class_weight = dict(enumerate(class_weight))

        path_func = delayed(_log_reg_scoring_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if self.solver in ['sag', 'saga']:
            backend = 'threading'
        else:
            backend = 'multiprocessing'
        # ... other code
```

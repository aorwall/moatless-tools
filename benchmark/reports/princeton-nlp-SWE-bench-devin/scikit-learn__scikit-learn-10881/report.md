# scikit-learn__scikit-learn-10881

| **scikit-learn/scikit-learn** | `4989a9503753a92089f39e154a2bb5d160b5d276` |
| ---- | ---- |
| **No of patches** | 2 |
| **All found context length** | 20133 |
| **Any found context length** | 20133 |
| **Avg pos** | 63.5 |
| **Min pos** | 34 |
| **Max pos** | 93 |
| **Top file pos** | 4 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -707,7 +707,7 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                 func, w0, fprime=None,
                 args=(X, target, 1. / C, sample_weight),
                 iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
-            if info["warnflag"] == 1 and verbose > 0:
+            if info["warnflag"] == 1:
                 warnings.warn("lbfgs failed to converge. Increase the number "
                               "of iterations.", ConvergenceWarning)
             # In scipy <= 1.0.0, nit may exceed maxiter.
diff --git a/sklearn/svm/base.py b/sklearn/svm/base.py
--- a/sklearn/svm/base.py
+++ b/sklearn/svm/base.py
@@ -907,7 +907,7 @@ def _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight,
     # on 32-bit platforms, we can't get to the UINT_MAX limit that
     # srand supports
     n_iter_ = max(n_iter_)
-    if n_iter_ >= max_iter and verbose > 0:
+    if n_iter_ >= max_iter:
         warnings.warn("Liblinear failed to converge, increase "
                       "the number of iterations.", ConvergenceWarning)
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/linear_model/logistic.py | 710 | 710 | 34 | 4 | 20133
| sklearn/svm/base.py | 910 | 910 | 93 | 48 | 55113


## Problem Statement

```
No warning when LogisticRegression does not converge
<!--
If your issue is a usage question, submit it here instead:
- StackOverflow with the scikit-learn tag: http://stackoverflow.com/questions/tagged/scikit-learn
- Mailing List: https://mail.python.org/mailman/listinfo/scikit-learn
For more information, see User Questions: http://scikit-learn.org/stable/support.html#user-questions
-->

<!-- Instructions For Filing a Bug: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md#filing-bugs -->

#### Description
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->
I've run LogisticRegressionCV on the Wisconsin Breast Cancer data, and the output of clf.n_iter_ was 100 for all but 1 of the variables. The default of 100 iterations was probably not sufficient in this case. Should there not be some kind of warning? I have done some tests and ~3000 iterations was probably a better choice for max_iter...

#### Steps/Code to Reproduce
\`\`\`py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegressionCV

data = load_breast_cancer()
y = data.target
X = data.data

clf = LogisticRegressionCV()
clf.fit(X, y)
print(clf.n_iter_)
\`\`\`

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

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->
Some kind of error to be shown. E.g: "result did not converge, try increasing the maximum number of iterations (max_iter)"

#### Versions
<!--
Please run the following snippet and paste the output below.
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->

>>> import platform; print(platform.platform())
Darwin-16.7.0-x86_64-i386-64bit
>>> import sys; print("Python", sys.version)
('Python', '2.7.14 |Anaconda, Inc.| (default, Oct  5 2017, 02:28:52) \n[GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]')
>>> import numpy; print("NumPy", numpy.__version__)
('NumPy', '1.13.3')
>>> import scipy; print("SciPy", scipy.__version__)
('SciPy', '1.0.0')
>>> import sklearn; print("Scikit-Learn", sklearn.__version__)
('Scikit-Learn', '0.19.1')


<!-- Thanks for contributing! -->


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | 1 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 59| 462 | 462 | 988 | 
| 2 | 2 benchmarks/bench_rcv1_logreg_convergence.py | 67 | 92| 201 | 663 | 2936 | 
| 3 | 2 benchmarks/bench_rcv1_logreg_convergence.py | 197 | 239| 373 | 1036 | 2936 | 
| 4 | 2 benchmarks/bench_rcv1_logreg_convergence.py | 95 | 121| 218 | 1254 | 2936 | 
| 5 | 2 benchmarks/bench_rcv1_logreg_convergence.py | 6 | 31| 202 | 1456 | 2936 | 
| 6 | 3 sklearn/linear_model/stochastic_gradient.py | 7 | 42| 312 | 1768 | 15103 | 
| 7 | **4 sklearn/linear_model/logistic.py** | 956 | 1156| 2187 | 3955 | 31705 | 
| 8 | 5 sklearn/utils/estimator_checks.py | 1272 | 1379| 1122 | 5077 | 50981 | 
| 9 | 5 sklearn/utils/estimator_checks.py | 1027 | 1095| 601 | 5678 | 50981 | 
| 10 | 6 sklearn/exceptions.py | 99 | 128| 341 | 6019 | 52167 | 
| 11 | 6 benchmarks/bench_rcv1_logreg_convergence.py | 34 | 64| 225 | 6244 | 52167 | 
| 12 | 6 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 61 | 119| 519 | 6763 | 52167 | 
| 13 | 7 examples/linear_model/plot_lasso_model_selection.py | 92 | 156| 503 | 7266 | 53496 | 
| 14 | 7 sklearn/exceptions.py | 40 | 71| 216 | 7482 | 53496 | 
| 15 | 8 examples/text/plot_document_classification_20newsgroups.py | 250 | 328| 673 | 8155 | 56053 | 
| 16 | 8 benchmarks/bench_rcv1_logreg_convergence.py | 142 | 194| 532 | 8687 | 56053 | 
| 17 | 8 sklearn/utils/estimator_checks.py | 1636 | 1674| 441 | 9128 | 56053 | 
| 18 | 9 benchmarks/bench_glmnet.py | 47 | 129| 796 | 9924 | 57140 | 
| 19 | 9 benchmarks/bench_rcv1_logreg_convergence.py | 124 | 139| 148 | 10072 | 57140 | 
| 20 | 10 sklearn/linear_model/coordinate_descent.py | 8 | 29| 154 | 10226 | 77651 | 
| 21 | 11 sklearn/__init__.py | 1 | 78| 659 | 10885 | 78443 | 
| 22 | **11 sklearn/linear_model/logistic.py** | 1368 | 1562| 2098 | 12983 | 78443 | 
| 23 | 11 sklearn/linear_model/coordinate_descent.py | 1395 | 1571| 1686 | 14669 | 78443 | 
| 24 | 11 sklearn/utils/estimator_checks.py | 1701 | 1743| 449 | 15118 | 78443 | 
| 25 | **11 sklearn/linear_model/logistic.py** | 1737 | 1792| 542 | 15660 | 78443 | 
| 26 | 11 sklearn/exceptions.py | 74 | 96| 186 | 15846 | 78443 | 
| 27 | 11 examples/linear_model/plot_lasso_model_selection.py | 1 | 78| 691 | 16537 | 78443 | 
| 28 | 12 sklearn/covariance/graph_lasso_.py | 619 | 697| 821 | 17358 | 86627 | 
| 29 | 13 examples/linear_model/plot_logistic_path.py | 1 | 56| 286 | 17644 | 86938 | 
| 30 | 14 sklearn/learning_curve.py | 1 | 25| 145 | 17789 | 90346 | 
| 31 | **14 sklearn/linear_model/logistic.py** | 1 | 37| 219 | 18008 | 90346 | 
| 32 | 15 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 18780 | 91511 | 
| 33 | **15 sklearn/linear_model/logistic.py** | 1680 | 1735| 642 | 19422 | 91511 | 
| **-> 34 <-** | **15 sklearn/linear_model/logistic.py** | 704 | 762| 711 | 20133 | 91511 | 
| 35 | **15 sklearn/linear_model/logistic.py** | 1564 | 1584| 221 | 20354 | 91511 | 
| 36 | 15 sklearn/utils/estimator_checks.py | 1382 | 1440| 489 | 20843 | 91511 | 
| 37 | 16 sklearn/linear_model/randomized_l1.py | 390 | 506| 1055 | 21898 | 97282 | 
| 38 | 17 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 606 | 22504 | 97913 | 
| 39 | **17 sklearn/linear_model/logistic.py** | 1267 | 1308| 426 | 22930 | 97913 | 
| 40 | 18 examples/exercises/plot_cv_diabetes.py | 1 | 81| 668 | 23598 | 98581 | 
| 41 | 19 benchmarks/bench_sgd_regression.py | 4 | 151| 1314 | 24912 | 99922 | 
| 42 | 19 sklearn/utils/estimator_checks.py | 746 | 777| 297 | 25209 | 99922 | 
| 43 | 20 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 80| 655 | 25864 | 100608 | 
| 44 | 20 sklearn/linear_model/coordinate_descent.py | 454 | 503| 588 | 26452 | 100608 | 
| 45 | 21 benchmarks/bench_mnist.py | 85 | 106| 314 | 26766 | 102339 | 
| 46 | 22 examples/covariance/plot_outlier_detection.py | 1 | 76| 586 | 27352 | 103558 | 
| 47 | 23 sklearn/ensemble/gradient_boosting.py | 810 | 893| 806 | 28158 | 121822 | 
| 48 | 23 sklearn/linear_model/stochastic_gradient.py | 85 | 136| 595 | 28753 | 121822 | 
| 49 | 24 benchmarks/bench_tree.py | 64 | 125| 523 | 29276 | 122692 | 
| 50 | **24 sklearn/linear_model/logistic.py** | 656 | 703| 639 | 29915 | 122692 | 
| 51 | 24 sklearn/utils/estimator_checks.py | 1677 | 1698| 214 | 30129 | 122692 | 
| 52 | 25 examples/ensemble/plot_ensemble_oob.py | 1 | 87| 681 | 30810 | 123425 | 
| 53 | 25 sklearn/utils/estimator_checks.py | 1608 | 1633| 277 | 31087 | 123425 | 
| 54 | 26 examples/model_selection/plot_learning_curve.py | 107 | 126| 220 | 31307 | 124630 | 
| 55 | 27 examples/model_selection/plot_randomized_search.py | 54 | 89| 332 | 31639 | 125344 | 
| 56 | 27 sklearn/utils/estimator_checks.py | 1462 | 1491| 223 | 31862 | 125344 | 
| 57 | 28 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 32402 | 126868 | 
| 58 | 29 examples/calibration/plot_compare_calibration.py | 1 | 78| 759 | 33161 | 128049 | 
| 59 | 29 examples/text/plot_document_classification_20newsgroups.py | 206 | 247| 338 | 33499 | 128049 | 
| 60 | 30 examples/classification/plot_lda.py | 41 | 72| 325 | 33824 | 128640 | 
| 61 | 31 examples/ensemble/plot_gradient_boosting_oob.py | 92 | 137| 447 | 34271 | 129888 | 
| 62 | 32 examples/neural_networks/plot_mlp_training_curves.py | 1 | 48| 544 | 34815 | 130815 | 
| 63 | 32 sklearn/utils/estimator_checks.py | 1 | 74| 640 | 35455 | 130815 | 
| 64 | 33 doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py | 14 | 80| 633 | 36088 | 131541 | 
| 65 | 34 sklearn/metrics/scorer.py | 469 | 533| 744 | 36832 | 136415 | 
| 66 | 35 sklearn/linear_model/least_angle.py | 207 | 365| 1584 | 38416 | 150136 | 
| 67 | 35 sklearn/linear_model/coordinate_descent.py | 1155 | 1227| 810 | 39226 | 150136 | 
| 68 | 35 sklearn/linear_model/coordinate_descent.py | 1380 | 1594| 208 | 39434 | 150136 | 
| 69 | 35 sklearn/linear_model/randomized_l1.py | 358 | 387| 296 | 39730 | 150136 | 
| 70 | 35 examples/covariance/plot_outlier_detection.py | 77 | 130| 632 | 40362 | 150136 | 
| 71 | 36 examples/linear_model/plot_robust_fit.py | 74 | 98| 285 | 40647 | 150930 | 
| 72 | 36 sklearn/covariance/graph_lasso_.py | 202 | 277| 858 | 41505 | 150930 | 
| 73 | 37 benchmarks/bench_covertype.py | 113 | 191| 757 | 42262 | 152833 | 
| 74 | 38 sklearn/utils/validation.py | 1 | 31| 141 | 42403 | 160195 | 
| 75 | 39 examples/neighbors/plot_lof.py | 1 | 60| 510 | 42913 | 160705 | 
| 76 | 39 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 91| 772 | 43685 | 160705 | 
| 77 | 40 examples/linear_model/plot_ols_ridge_variance.py | 1 | 72| 526 | 44211 | 161239 | 
| 78 | 41 examples/model_selection/plot_nested_cross_validation_iris.py | 74 | 118| 433 | 44644 | 162313 | 
| 79 | 42 examples/plot_anomaly_comparison.py | 1 | 82| 716 | 45360 | 163427 | 
| 80 | 43 examples/covariance/plot_covariance_estimation.py | 1 | 89| 772 | 46132 | 164568 | 
| 81 | **43 sklearn/linear_model/logistic.py** | 590 | 654| 740 | 46872 | 164568 | 
| 82 | 43 sklearn/linear_model/randomized_l1.py | 507 | 538| 332 | 47204 | 164568 | 
| 83 | 44 examples/linear_model/plot_logistic.py | 1 | 67| 440 | 47644 | 165016 | 
| 84 | 45 sklearn/covariance/robust_covariance.py | 95 | 175| 755 | 48399 | 171875 | 
| 85 | 46 examples/cluster/plot_kmeans_stability_low_dim_dense.py | 70 | 120| 506 | 48905 | 172934 | 
| 86 | 46 examples/calibration/plot_compare_calibration.py | 79 | 123| 396 | 49301 | 172934 | 
| 87 | 46 sklearn/utils/estimator_checks.py | 1238 | 1269| 258 | 49559 | 172934 | 
| 88 | 46 sklearn/utils/estimator_checks.py | 1164 | 1218| 567 | 50126 | 172934 | 
| 89 | **46 sklearn/linear_model/logistic.py** | 895 | 1363| 602 | 50728 | 172934 | 
| 90 | 46 sklearn/linear_model/coordinate_descent.py | 1246 | 1379| 1114 | 51842 | 172934 | 
| 91 | 47 sklearn/cluster/k_means_.py | 1162 | 1225| 659 | 52501 | 186655 | 
| 92 | 47 sklearn/linear_model/stochastic_gradient.py | 594 | 788| 1964 | 54465 | 186655 | 
| **-> 93 <-** | **48 sklearn/svm/base.py** | 860 | 922| 648 | 55113 | 194714 | 
| 94 | 48 sklearn/utils/estimator_checks.py | 310 | 363| 520 | 55633 | 194714 | 
| 95 | 49 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 56329 | 195438 | 
| 96 | 49 sklearn/utils/estimator_checks.py | 143 | 163| 203 | 56532 | 195438 | 
| 97 | 50 benchmarks/bench_lof.py | 36 | 107| 650 | 57182 | 196354 | 
| 98 | 51 examples/covariance/plot_robust_vs_empirical_covariance.py | 77 | 141| 785 | 57967 | 198026 | 
| 99 | **51 sklearn/linear_model/logistic.py** | 1158 | 1176| 202 | 58169 | 198026 | 
| 100 | 51 benchmarks/bench_mnist.py | 109 | 181| 695 | 58864 | 198026 | 


### Hint

```
If you use verbose=1 in your snippet, you'll get plenty of ConvergenceWarning.

\`\`\`py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegressionCV

data = load_breast_cancer()
y = data.target
X = data.data

clf = LogisticRegressionCV(verbose=1)
clf.fit(X, y)
print(clf.n_iter_)
\`\`\`

I am going to close this one, please reopen if  you strongly disagree.
I also think that `LogisticRegression` ~~(and `LogisticRegressionCV`)~~ should print a `ConvergenceWarning` when it fails to converge with default parameters.

 It does not make sense to expect the users to set the verbosity in order to get the warning. Also other methods don't do this:  e.g. MLPClassifier [may raise ConvergenceWarnings](https://github.com/scikit-learn/scikit-learn/blob/de29f3f22db6e017aef9dc77935d8ef43d2d7b44/sklearn/neural_network/tests/test_mlp.py#L70) so does [MultiTaskElasticNet](https://github.com/scikit-learn/scikit-learn/blob/da71b827b8b56bd8305b7fe6c13724c7b5355209/sklearn/linear_model/tests/test_coordinate_descent.py#L407) or [LassoLars](https://github.com/scikit-learn/scikit-learn/blob/34aad31924dc9c8551e6ff8edf4e729dbfc1b973/sklearn/linear_model/tests/test_least_angle.py#L349). 

 The culprit is this [line](https://github.com/scikit-learn/scikit-learn/blob/da71b827b8b56bd8305b7fe6c13724c7b5355209/sklearn/linear_model/logistic.py#L710-L712) that only affects the `lbfgs` solver. This is more critical if we can to make `lbfgs` the default solver https://github.com/scikit-learn/scikit-learn/issues/9997 in `LogisticRegression`
After more thoughts, the case of `LogisticRegressionCV` is more debatable, as there are always CV parameters that may fail to converge and handling it the same way as `GridSearchCV` would be probably more reasonable.

However,
\`\`\`py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
y = data.target
X = data.data

clf = LogisticRegression(solver='lbfgs')
clf.fit(X, y)
print(clf.n_iter_)
\`\`\`
silently fails to converge, and this is a bug IMO. This was also recently relevant in https://github.com/scikit-learn/scikit-learn/issues/10813

A quick git grep seems to confirm that ConvergenceWarning are generally issued when verbose=0, reopening.

The same thing happens for solver='liblinear' (the default solver at the time of writing) by the way (you need to decrease max_iter e.g. set it to 1). There should be a test that checks the ConvergenceWarning for all solvers.
I see that nobody is working on it. Can I do it ?
Sure,  thank you @AlexandreSev 
Unless @oliverangelil was planning to submit a PR..
I think I need to correct also this [line](https://github.com/scikit-learn/scikit-learn/blob/da71b827b8b56bd8305b7fe6c13724c7b5355209/sklearn/linear_model/tests/test_logistic.py#L807). Do you agree ?
Moreover, I just saw that the name of this [function](https://github.com/scikit-learn/scikit-learn/blob/da71b827b8b56bd8305b7fe6c13724c7b5355209/sklearn/linear_model/tests/test_logistic.py#L92) is not perfectly correct, since it does not test the convergence warning.
Maybe we could write a test a bit like this [one](https://github.com/scikit-learn/scikit-learn/blob/da71b827b8b56bd8305b7fe6c13724c7b5355209/sklearn/linear_model/tests/test_logistic.py#L1037) to test all the warnings. What do you think ?
You can reuse test_max_iter indeed to check for all solvers that there has been a warning and add an assert_warns_message in it. If you do that, test_logistic_regression_convergence_warnings is not really needed any more. There is no need to touch test_lr_liblinear_warning.

The best is to open a PR and see how that goes! Not that you have mentioned changing the tests but you will also need to change sklearn/linear_model/logistic.py to make sure a ConvergenceWarning is issued for all the solvers.


yes I think there are a few cases where we only warn if verbose. I'd rather
consistently warn even if verbose=0

```

## Patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -707,7 +707,7 @@ def logistic_regression_path(X, y, pos_class=None, Cs=10, fit_intercept=True,
                 func, w0, fprime=None,
                 args=(X, target, 1. / C, sample_weight),
                 iprint=(verbose > 0) - 1, pgtol=tol, maxiter=max_iter)
-            if info["warnflag"] == 1 and verbose > 0:
+            if info["warnflag"] == 1:
                 warnings.warn("lbfgs failed to converge. Increase the number "
                               "of iterations.", ConvergenceWarning)
             # In scipy <= 1.0.0, nit may exceed maxiter.
diff --git a/sklearn/svm/base.py b/sklearn/svm/base.py
--- a/sklearn/svm/base.py
+++ b/sklearn/svm/base.py
@@ -907,7 +907,7 @@ def _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight,
     # on 32-bit platforms, we can't get to the UINT_MAX limit that
     # srand supports
     n_iter_ = max(n_iter_)
-    if n_iter_ >= max_iter and verbose > 0:
+    if n_iter_ >= max_iter:
         warnings.warn("Liblinear failed to converge, increase "
                       "the number of iterations.", ConvergenceWarning)
 

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_logistic.py b/sklearn/linear_model/tests/test_logistic.py
--- a/sklearn/linear_model/tests/test_logistic.py
+++ b/sklearn/linear_model/tests/test_logistic.py
@@ -800,15 +800,6 @@ def test_logistic_regression_class_weights():
         assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=6)
 
 
-def test_logistic_regression_convergence_warnings():
-    # Test that warnings are raised if model does not converge
-
-    X, y = make_classification(n_samples=20, n_features=20, random_state=0)
-    clf_lib = LogisticRegression(solver='liblinear', max_iter=2, verbose=1)
-    assert_warns(ConvergenceWarning, clf_lib.fit, X, y)
-    assert_equal(clf_lib.n_iter_, 2)
-
-
 def test_logistic_regression_multinomial():
     # Tests for the multinomial option in logistic regression
 
@@ -1033,7 +1024,6 @@ def test_logreg_predict_proba_multinomial():
     assert_greater(clf_wrong_loss, clf_multi_loss)
 
 
-@ignore_warnings
 def test_max_iter():
     # Test that the maximum number of iteration is reached
     X, y_bin = iris.data, iris.target.copy()
@@ -1049,7 +1039,7 @@ def test_max_iter():
                 lr = LogisticRegression(max_iter=max_iter, tol=1e-15,
                                         multi_class=multi_class,
                                         random_state=0, solver=solver)
-                lr.fit(X, y_bin)
+                assert_warns(ConvergenceWarning, lr.fit, X, y_bin)
                 assert_equal(lr.n_iter_[0], max_iter)
 
 
diff --git a/sklearn/model_selection/tests/test_search.py b/sklearn/model_selection/tests/test_search.py
--- a/sklearn/model_selection/tests/test_search.py
+++ b/sklearn/model_selection/tests/test_search.py
@@ -33,6 +33,7 @@
 from sklearn.base import BaseEstimator
 from sklearn.base import clone
 from sklearn.exceptions import NotFittedError
+from sklearn.exceptions import ConvergenceWarning
 from sklearn.datasets import make_classification
 from sklearn.datasets import make_blobs
 from sklearn.datasets import make_multilabel_classification
@@ -350,7 +351,9 @@ def test_return_train_score_warn():
     for estimator in estimators:
         for val in [True, False, 'warn']:
             estimator.set_params(return_train_score=val)
-            result[val] = assert_no_warnings(estimator.fit, X, y).cv_results_
+            fit_func = ignore_warnings(estimator.fit,
+                                       category=ConvergenceWarning)
+            result[val] = assert_no_warnings(fit_func, X, y).cv_results_
 
     train_keys = ['split0_train_score', 'split1_train_score',
                   'split2_train_score', 'mean_train_score', 'std_train_score']
diff --git a/sklearn/svm/tests/test_svm.py b/sklearn/svm/tests/test_svm.py
--- a/sklearn/svm/tests/test_svm.py
+++ b/sklearn/svm/tests/test_svm.py
@@ -871,13 +871,17 @@ def test_consistent_proba():
     assert_array_almost_equal(proba_1, proba_2)
 
 
-def test_linear_svc_convergence_warnings():
+def test_linear_svm_convergence_warnings():
     # Test that warnings are raised if model does not converge
 
-    lsvc = svm.LinearSVC(max_iter=2, verbose=1)
+    lsvc = svm.LinearSVC(random_state=0, max_iter=2)
     assert_warns(ConvergenceWarning, lsvc.fit, X, Y)
     assert_equal(lsvc.n_iter_, 2)
 
+    lsvr = svm.LinearSVR(random_state=0, max_iter=2)
+    assert_warns(ConvergenceWarning, lsvr.fit, iris.data, iris.target)
+    assert_equal(lsvr.n_iter_, 2)
+
 
 def test_svr_coef_sign():
     # Test that SVR(kernel="linear") has coef_ with the right sign.

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
### 2 - benchmarks/bench_rcv1_logreg_convergence.py:

Start line: 67, End line: 92

```python
def bench(clfs):
    for (name, clf, iter_range, train_losses, train_scores,
         test_scores, durations) in clfs:
        print("training %s" % name)
        clf_type = type(clf)
        clf_params = clf.get_params()

        for n_iter in iter_range:
            gc.collect()

            train_loss, train_score, test_score, duration = bench_one(
                name, clf_type, clf_params, n_iter)

            train_losses.append(train_loss)
            train_scores.append(train_score)
            test_scores.append(test_score)
            durations.append(duration)
            print("classifier: %s" % name)
            print("train_loss: %.8f" % train_loss)
            print("train_score: %.8f" % train_score)
            print("test_score: %.8f" % test_score)
            print("time for fit: %.8f seconds" % duration)
            print("")

        print("")
    return clfs
```
### 3 - benchmarks/bench_rcv1_logreg_convergence.py:

Start line: 197, End line: 239

```python
if lightning_clf is not None and not fit_intercept:
    alpha = 1. / C / n_samples
    # compute the same step_size than in LR-sag
    max_squared_sum = get_max_squared_sum(X)
    step_size = get_auto_step_size(max_squared_sum, alpha, "log",
                                   fit_intercept)

    clfs.append(
        ("Lightning-SVRG",
         lightning_clf.SVRGClassifier(alpha=alpha, eta=step_size,
                                      tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))
    clfs.append(
        ("Lightning-SAG",
         lightning_clf.SAGClassifier(alpha=alpha, eta=step_size,
                                     tol=tol, loss="log"),
         sag_iter_range, [], [], [], []))

    # We keep only 200 features, to have a dense dataset,
    # and compare to lightning SAG, which seems incorrect in the sparse case.
    X_csc = X.tocsc()
    nnz_in_each_features = X_csc.indptr[1:] - X_csc.indptr[:-1]
    X = X_csc[:, np.argsort(nnz_in_each_features)[-200:]]
    X = X.toarray()
    print("dataset: %.3f MB" % (X.nbytes / 1e6))


# Split training and testing. Switch train and test subset compared to
# LYRL2004 split, to have a larger training dataset.
n = 23149
X_test = X[:n, :]
y_test = y[:n]
X = X[n:, :]
y = y[n:]

clfs = bench(clfs)

plot_train_scores(clfs)
plot_test_scores(clfs)
plot_train_losses(clfs)
plot_dloss(clfs)
plt.show()
```
### 4 - benchmarks/bench_rcv1_logreg_convergence.py:

Start line: 95, End line: 121

```python
def plot_train_losses(clfs):
    plt.figure()
    for (name, _, _, train_losses, _, _, durations) in clfs:
        plt.plot(durations, train_losses, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train loss")


def plot_train_scores(clfs):
    plt.figure()
    for (name, _, _, _, train_scores, _, durations) in clfs:
        plt.plot(durations, train_scores, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("train score")
        plt.ylim((0.92, 0.96))


def plot_test_scores(clfs):
    plt.figure()
    for (name, _, _, _, _, test_scores, durations) in clfs:
        plt.plot(durations, test_scores, '-o', label=name)
        plt.legend(loc=0)
        plt.xlabel("seconds")
        plt.ylabel("test score")
        plt.ylim((0.92, 0.96))
```
### 5 - benchmarks/bench_rcv1_logreg_convergence.py:

Start line: 6, End line: 31

```python
import matplotlib.pyplot as plt
import numpy as np
import gc
import time

from sklearn.externals.joblib import Memory
from sklearn.linear_model import (LogisticRegression, SGDClassifier)
from sklearn.datasets import fetch_rcv1
from sklearn.linear_model.sag import get_auto_step_size

try:
    import lightning.classification as lightning_clf
except ImportError:
    lightning_clf = None

m = Memory(cachedir='.', verbose=0)


# compute logistic loss
def get_loss(w, intercept, myX, myy, C):
    n_samples = myX.shape[0]
    w = w.ravel()
    p = np.mean(np.log(1. + np.exp(-myy * (myX.dot(w) + intercept))))
    print("%f + %f" % (p, w.dot(w) / 2. / C / n_samples))
    p += w.dot(w) / 2. / C / n_samples
    return p
```
### 6 - sklearn/linear_model/stochastic_gradient.py:

Start line: 7, End line: 42

```python
import numpy as np
import warnings

from abc import ABCMeta, abstractmethod

from ..externals.joblib import Parallel, delayed

from .base import LinearClassifierMixin, SparseCoefMixin
from .base import make_dataset
from ..base import BaseEstimator, RegressorMixin
from ..utils import check_array, check_random_state, check_X_y
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.validation import check_is_fitted
from ..exceptions import ConvergenceWarning
from ..externals import six

from .sgd_fast import plain_sgd, average_sgd
from ..utils import compute_class_weight
from ..utils import deprecated
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import ModifiedHuber
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive
from .sgd_fast import SquaredEpsilonInsensitive


LEARNING_RATE_TYPES = {"constant": 1, "optimal": 2, "invscaling": 3,
                       "pa1": 4, "pa2": 5}

PENALTY_TYPES = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}

DEFAULT_EPSILON = 0.1
```
### 7 - sklearn/linear_model/logistic.py:

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

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
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
### 8 - sklearn/utils/estimator_checks.py:

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
### 9 - sklearn/utils/estimator_checks.py:

Start line: 1027, End line: 1095

```python
@ignore_warnings(category=DeprecationWarning)
def check_estimators_nan_inf(name, estimator_orig):
    # Checks that Estimator X's do not contain NaN or inf.
    rnd = np.random.RandomState(0)
    X_train_finite = pairwise_estimator_convert_X(rnd.uniform(size=(10, 3)),
                                                  estimator_orig)
    X_train_nan = rnd.uniform(size=(10, 3))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 3))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = multioutput_estimator_convert_y_2d(estimator_orig, y)
    error_string_fit = "Estimator doesn't check for NaN and inf in fit."
    error_string_predict = ("Estimator doesn't check for NaN and inf in"
                            " predict.")
    error_string_transform = ("Estimator doesn't check for NaN and inf in"
                              " transform.")
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=(DeprecationWarning, FutureWarning)):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            try:
                estimator.fit(X_train, y)
            except ValueError as e:
                if 'inf' not in repr(e) and 'NaN' not in repr(e):
                    print(error_string_fit, estimator, e)
                    traceback.print_exc(file=sys.stdout)
                    raise e
            except Exception as exc:
                print(error_string_fit, estimator, exc)
                traceback.print_exc(file=sys.stdout)
                raise exc
            else:
                raise AssertionError(error_string_fit, estimator)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                try:
                    estimator.predict(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_predict, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_predict, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_predict, estimator)

            # transform
            if hasattr(estimator, "transform"):
                try:
                    estimator.transform(X_train)
                except ValueError as e:
                    if 'inf' not in repr(e) and 'NaN' not in repr(e):
                        print(error_string_transform, estimator, e)
                        traceback.print_exc(file=sys.stdout)
                        raise e
                except Exception as exc:
                    print(error_string_transform, estimator, exc)
                    traceback.print_exc(file=sys.stdout)
                else:
                    raise AssertionError(error_string_transform, estimator)
```
### 10 - sklearn/exceptions.py:

Start line: 99, End line: 128

```python
class FitFailedWarning(RuntimeWarning):
    """Warning class used if there is an error while fitting the estimator.

    This Warning is used in meta estimators GridSearchCV and RandomizedSearchCV
    and the cross-validation helper function cross_val_score to warn when there
    is an error while fitting the estimator.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import FitFailedWarning
    >>> import warnings
    >>> warnings.simplefilter('always', FitFailedWarning)
    >>> gs = GridSearchCV(LinearSVC(), {'C': [-1, -2]}, error_score=0, cv=2)
    >>> X, y = [[1, 2], [3, 4], [5, 6], [7, 8]], [0, 0, 1, 1]
    >>> with warnings.catch_warnings(record=True) as w:
    ...     try:
    ...         gs.fit(X, y)   # This will raise a ValueError since C is < 0
    ...     except ValueError:
    ...         pass
    ...     print(repr(w[-1].message))
    ... # doctest: +NORMALIZE_WHITESPACE
    FitFailedWarning('Estimator fit failed. The score on this train-test
    partition for these parameters will be set to 0.000000.
    Details: \\nValueError: Penalty term must be positive; got (C=-2)\\n',)

    .. versionchanged:: 0.18
       Moved from sklearn.cross_validation.
    """
```
### 22 - sklearn/linear_model/logistic.py:

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

    solver : {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},
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
### 25 - sklearn/linear_model/logistic.py:

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
### 31 - sklearn/linear_model/logistic.py:

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
### 33 - sklearn/linear_model/logistic.py:

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
### 34 - sklearn/linear_model/logistic.py:

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
            if info["warnflag"] == 1 and verbose > 0:
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
### 35 - sklearn/linear_model/logistic.py:

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
### 39 - sklearn/linear_model/logistic.py:

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
### 50 - sklearn/linear_model/logistic.py:

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
### 81 - sklearn/linear_model/logistic.py:

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
### 89 - sklearn/linear_model/logistic.py:

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
### 93 - sklearn/svm/base.py:

Start line: 860, End line: 922

```python
def _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight,
                   penalty, dual, verbose, max_iter, tol,
                   random_state=None, multi_class='ovr',
                   loss='logistic_regression', epsilon=0.1,
                   sample_weight=None):
    if loss not in ['epsilon_insensitive', 'squared_epsilon_insensitive']:
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        classes_ = enc.classes_
        if len(classes_) < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        class_weight_ = compute_class_weight(class_weight, classes_, y)
    else:
        class_weight_ = np.empty(0, dtype=np.float64)
        y_ind = y
    liblinear.set_verbosity_wrap(verbose)
    rnd = check_random_state(random_state)
    if verbose:
        print('[LibLinear]', end='')

    # LinearSVC breaks when intercept_scaling is <= 0
    bias = -1.0
    if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError("Intercept scaling is %r but needs to be greater than 0."
                             " To disable fitting an intercept,"
                             " set fit_intercept=False." % intercept_scaling)
        else:
            bias = intercept_scaling

    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    else:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(sample_weight, X)

    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X, y_ind, sp.isspmatrix(X), solver_type, tol, bias, C,
        class_weight_, max_iter, rnd.randint(np.iinfo('i').max),
        epsilon, sample_weight)
    # Regarding rnd.randint(..) in the above signature:
    # seed for srand in range [0..INT_MAX); due to limitations in Numpy
    # on 32-bit platforms, we can't get to the UINT_MAX limit that
    # srand supports
    n_iter_ = max(n_iter_)
    if n_iter_ >= max_iter and verbose > 0:
        warnings.warn("Liblinear failed to converge, increase "
                      "the number of iterations.", ConvergenceWarning)

    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.

    return coef_, intercept_, n_iter_
```
### 99 - sklearn/linear_model/logistic.py:

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

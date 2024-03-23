# scikit-learn__scikit-learn-10881

* repo: scikit-learn/scikit-learn
* base_commit: `4989a9503753a92089f39e154a2bb5d160b5d276`

## Problem statement

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
```py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegressionCV

data = load_breast_cancer()
y = data.target
X = data.data

clf = LogisticRegressionCV()
clf.fit(X, y)
print(clf.n_iter_)
```

<!--
Example:
```python
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
```
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

# scikit-learn__scikit-learn-10986

* repo: scikit-learn/scikit-learn
* base_commit: `ca436e7017ae069a29de19caf71689e9b9b9c452`

## Problem statement

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

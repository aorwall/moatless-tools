# scikit-learn__scikit-learn-9939

* repo: scikit-learn/scikit-learn
* base_commit: `f247ad5adfe86b2ee64a4a3db1b496c8bf1c9dff`

## Problem statement

Incorrect predictions when fitting a LogisticRegression model on binary outcomes with `multi_class='multinomial'`.
#### Description
Incorrect predictions when fitting a LogisticRegression model on binary outcomes with `multi_class='multinomial'`.
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->

#### Steps/Code to Reproduce
```python
    from sklearn.linear_model import LogisticRegression
    import sklearn.metrics
    import numpy as np

    # Set up a logistic regression object
    lr = LogisticRegression(C=1000000, multi_class='multinomial',
                            solver='sag', tol=0.0001, warm_start=False,
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

    lr.fit(Z, Y)
    p = lr.predict_proba(Z)
    print(sklearn.metrics.log_loss(Y, p)) # ...

    print(lr.intercept_)
    print(lr.coef_)
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
If we compare against R or using `multi_class='ovr'`, the log loss (which is approximately proportional to the objective function as the regularisation is set to be negligible through the choice of `C`) is incorrect. We expect the log loss to be roughly `0.5922995`

#### Actual Results
The actual log loss when using `multi_class='multinomial'` is `0.61505641264`.

#### Further Information
See the stack exchange question https://stats.stackexchange.com/questions/306886/confusing-behaviour-of-scikit-learn-logistic-regression-multinomial-optimisation?noredirect=1#comment583412_306886 for more information.

The issue it seems is caused in https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/linear_model/logistic.py#L762. In the `multinomial` case even if `classes.size==2` we cannot reduce to a 1D case by throwing away one of the vectors of coefficients (as we can in normal binary logistic regression). This is essentially a difference between softmax (redundancy allowed) and logistic regression.

This can be fixed by commenting out the lines 762 and 763. I am apprehensive however that this may cause some other unknown issues which is why I am positing as a bug.

#### Versions
<!--
Please run the following snippet and paste the output below.
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->
Linux-4.10.0-33-generic-x86_64-with-Ubuntu-16.04-xenial
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
NumPy 1.13.1
SciPy 0.19.1
Scikit-Learn 0.19.0
<!-- Thanks for contributing! -->



## Patch

```diff
diff --git a/sklearn/linear_model/logistic.py b/sklearn/linear_model/logistic.py
--- a/sklearn/linear_model/logistic.py
+++ b/sklearn/linear_model/logistic.py
@@ -1101,14 +1101,18 @@ class LogisticRegression(BaseEstimator, LinearClassifierMixin,
     coef_ : array, shape (1, n_features) or (n_classes, n_features)
         Coefficient of the features in the decision function.
 
-        `coef_` is of shape (1, n_features) when the given problem
-        is binary.
+        `coef_` is of shape (1, n_features) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `coef_` corresponds
+        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
 
     intercept_ : array, shape (1,) or (n_classes,)
         Intercept (a.k.a. bias) added to the decision function.
 
         If `fit_intercept` is set to False, the intercept is set to zero.
-        `intercept_` is of shape(1,) when the problem is binary.
+        `intercept_` is of shape (1,) when the given problem is binary.
+        In particular, when `multi_class='multinomial'`, `intercept_`
+        corresponds to outcome 1 (True) and `-intercept_` corresponds to
+        outcome 0 (False).
 
     n_iter_ : array, shape (n_classes,) or (1, )
         Actual number of iterations for all classes. If binary or multinomial,
@@ -1332,11 +1336,17 @@ def predict_proba(self, X):
         """
         if not hasattr(self, "coef_"):
             raise NotFittedError("Call fit before prediction")
-        calculate_ovr = self.coef_.shape[0] == 1 or self.multi_class == "ovr"
-        if calculate_ovr:
+        if self.multi_class == "ovr":
             return super(LogisticRegression, self)._predict_proba_lr(X)
         else:
-            return softmax(self.decision_function(X), copy=False)
+            decision = self.decision_function(X)
+            if decision.ndim == 1:
+                # Workaround for multi_class="multinomial" and binary outcomes
+                # which requires softmax prediction with only a 1D decision.
+                decision_2d = np.c_[-decision, decision]
+            else:
+                decision_2d = decision
+            return softmax(decision_2d, copy=False)
 
     def predict_log_proba(self, X):
         """Log of probability estimates.

```

## Test Patch

```diff
diff --git a/sklearn/linear_model/tests/test_logistic.py b/sklearn/linear_model/tests/test_logistic.py
--- a/sklearn/linear_model/tests/test_logistic.py
+++ b/sklearn/linear_model/tests/test_logistic.py
@@ -198,6 +198,23 @@ def test_multinomial_binary():
         assert_greater(np.mean(pred == target), .9)
 
 
+def test_multinomial_binary_probabilities():
+    # Test multinomial LR gives expected probabilities based on the
+    # decision function, for a binary problem.
+    X, y = make_classification()
+    clf = LogisticRegression(multi_class='multinomial', solver='saga')
+    clf.fit(X, y)
+
+    decision = clf.decision_function(X)
+    proba = clf.predict_proba(X)
+
+    expected_proba_class_1 = (np.exp(decision) /
+                              (np.exp(decision) + np.exp(-decision)))
+    expected_proba = np.c_[1-expected_proba_class_1, expected_proba_class_1]
+
+    assert_almost_equal(proba, expected_proba)
+
+
 def test_sparsify():
     # Test sparsify and densify members.
     n_samples, n_features = iris.data.shape

```

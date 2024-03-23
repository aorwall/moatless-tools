# scikit-learn__scikit-learn-13280

* repo: scikit-learn/scikit-learn
* base_commit: `face9daf045846bb0a39bfb396432c8685570cdd`

## Problem statement

partial_fit does not account for unobserved target values when fitting priors to data
My understanding is that priors should be fitted to the data using observed target frequencies **and a variant of [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) to avoid assigning 0 probability to targets  not yet observed.**


It seems the implementation of `partial_fit` does not account for unobserved targets at the time of the first training batch when computing priors.

```python
    import numpy as np
    import sklearn
    from sklearn.naive_bayes import MultinomialNB
    
    print('scikit-learn version:', sklearn.__version__)
    
    # Create toy training data
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    
    # All possible targets
    classes = np.append(y, 7)
    
    clf = MultinomialNB()
    clf.partial_fit(X, y, classes=classes)
```
-----------------------------------
    /home/skojoian/.local/lib/python3.6/site-packages/sklearn/naive_bayes.py:465: RuntimeWarning: divide by zero encountered in log
      self.class_log_prior_ = (np.log(self.class_count_) -
    scikit-learn version: 0.20.2

This behavior is not very intuitive to me. It seems `partial_fit` requires `classes` for the right reason, but doesn't actually offset target frequencies to account for unobserved targets.


## Patch

```diff
diff --git a/sklearn/naive_bayes.py b/sklearn/naive_bayes.py
--- a/sklearn/naive_bayes.py
+++ b/sklearn/naive_bayes.py
@@ -460,8 +460,14 @@ def _update_class_log_prior(self, class_prior=None):
                                  " classes.")
             self.class_log_prior_ = np.log(class_prior)
         elif self.fit_prior:
+            with warnings.catch_warnings():
+                # silence the warning when count is 0 because class was not yet
+                # observed
+                warnings.simplefilter("ignore", RuntimeWarning)
+                log_class_count = np.log(self.class_count_)
+
             # empirical prior, with sample_weight taken into account
-            self.class_log_prior_ = (np.log(self.class_count_) -
+            self.class_log_prior_ = (log_class_count -
                                      np.log(self.class_count_.sum()))
         else:
             self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

```

## Test Patch

```diff
diff --git a/sklearn/tests/test_naive_bayes.py b/sklearn/tests/test_naive_bayes.py
--- a/sklearn/tests/test_naive_bayes.py
+++ b/sklearn/tests/test_naive_bayes.py
@@ -18,6 +18,7 @@
 from sklearn.utils.testing import assert_raise_message
 from sklearn.utils.testing import assert_greater
 from sklearn.utils.testing import assert_warns
+from sklearn.utils.testing import assert_no_warnings
 
 from sklearn.naive_bayes import GaussianNB, BernoulliNB
 from sklearn.naive_bayes import MultinomialNB, ComplementNB
@@ -244,6 +245,33 @@ def check_partial_fit(cls):
     assert_array_equal(clf1.feature_count_, clf3.feature_count_)
 
 
+def test_mnb_prior_unobserved_targets():
+    # test smoothing of prior for yet unobserved targets
+
+    # Create toy training data
+    X = np.array([[0, 1], [1, 0]])
+    y = np.array([0, 1])
+
+    clf = MultinomialNB()
+
+    assert_no_warnings(
+        clf.partial_fit, X, y, classes=[0, 1, 2]
+    )
+
+    assert clf.predict([[0, 1]]) == 0
+    assert clf.predict([[1, 0]]) == 1
+    assert clf.predict([[1, 1]]) == 0
+
+    # add a training example with previously unobserved class
+    assert_no_warnings(
+        clf.partial_fit, [[1, 1]], [2]
+    )
+
+    assert clf.predict([[0, 1]]) == 0
+    assert clf.predict([[1, 0]]) == 1
+    assert clf.predict([[1, 1]]) == 2
+
+
 @pytest.mark.parametrize("cls", [MultinomialNB, BernoulliNB])
 def test_discretenb_partial_fit(cls):
     check_partial_fit(cls)

```

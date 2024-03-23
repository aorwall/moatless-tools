# scikit-learn__scikit-learn-13496

* repo: scikit-learn/scikit-learn
* base_commit: `3aefc834dce72e850bff48689bea3c7dff5f3fad`

## Problem statement

Expose warm_start in Isolation forest
It seems to me that `sklearn.ensemble.IsolationForest` supports incremental addition of new trees with the `warm_start` parameter of its parent class, `sklearn.ensemble.BaseBagging`.

Even though this parameter is not exposed in `__init__()` , it gets inherited from `BaseBagging` and one can use it by changing it to `True` after initialization. To make it work, you have to also increment `n_estimators` on every iteration. 

It took me a while to notice that it actually works, and I had to inspect the source code of both `IsolationForest` and `BaseBagging`. Also, it looks to me that the behavior is in-line with `sklearn.ensemble.BaseForest` that is behind e.g. `sklearn.ensemble.RandomForestClassifier`.

To make it more easier to use, I'd suggest to:
* expose `warm_start` in `IsolationForest.__init__()`, default `False`;
* document it in the same way as it is documented for `RandomForestClassifier`, i.e. say:
```py
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.
```
* add a test to make sure it works properly;
* possibly also mention in the "IsolationForest example" documentation entry;



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

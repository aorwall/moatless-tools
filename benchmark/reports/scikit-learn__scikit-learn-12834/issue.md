# scikit-learn__scikit-learn-12834

* repo: scikit-learn/scikit-learn
* base_commit: `55a98ab7e3b10966f6d00c3562f3a99896797964`

## Problem statement

`predict` fails for multioutput ensemble models with non-numeric DVs
#### Description
<!-- Example: Joblib Error thrown when calling fit on LatentDirichletAllocation with evaluate_every > 0-->
Multioutput forest models assume that the dependent variables are numeric. Passing string DVs returns the following error:

`ValueError: could not convert string to float:`

I'm going to take a stab at submitting a fix today, but I wanted to file an issue to document the problem in case I'm not able to finish a fix.

#### Steps/Code to Reproduce
I wrote a test based on `ensemble/tests/test_forest:test_multioutput` which currently fails:

```
def check_multioutput_string(name):
    # Check estimators on multi-output problems with string outputs.

    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [["red", "blue"], ["red", "blue"], ["red", "blue"], ["green", "green"],
               ["green", "green"], ["green", "green"], ["red", "purple"],
               ["red", "purple"], ["red", "purple"], ["green", "yellow"],
               ["green", "yellow"], ["green", "yellow"]]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [["red", "blue"], ["green", "green"], ["red", "purple"], ["green", "yellow"]]

    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_almost_equal(y_pred, y_test)

    if name in FOREST_CLASSIFIERS:
        with np.errstate(divide="ignore"):
            proba = est.predict_proba(X_test)
            assert_equal(len(proba), 2)
            assert_equal(proba[0].shape, (4, 2))
            assert_equal(proba[1].shape, (4, 4))

            log_proba = est.predict_log_proba(X_test)
            assert_equal(len(log_proba), 2)
            assert_equal(log_proba[0].shape, (4, 2))
            assert_equal(log_proba[1].shape, (4, 4))


@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_multioutput_string(name):
    check_multioutput_string(name)
```

#### Expected Results
No error is thrown, can run `predict` for all ensemble multioutput models
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->
`ValueError: could not convert string to float: <DV class>`

#### Versions
I replicated this error using the current master branch of sklearn (0.21.dev0).
<!--
Please run the following snippet and paste the output below.
For scikit-learn >= 0.20:
import sklearn; sklearn.show_versions()
For scikit-learn < 0.20:
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
-->


<!-- Thanks for contributing! -->



## Patch

```diff
diff --git a/sklearn/ensemble/forest.py b/sklearn/ensemble/forest.py
--- a/sklearn/ensemble/forest.py
+++ b/sklearn/ensemble/forest.py
@@ -547,7 +547,10 @@ def predict(self, X):
 
         else:
             n_samples = proba[0].shape[0]
-            predictions = np.zeros((n_samples, self.n_outputs_))
+            # all dtypes should be the same, so just take the first
+            class_type = self.classes_[0].dtype
+            predictions = np.empty((n_samples, self.n_outputs_),
+                                   dtype=class_type)
 
             for k in range(self.n_outputs_):
                 predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_forest.py b/sklearn/ensemble/tests/test_forest.py
--- a/sklearn/ensemble/tests/test_forest.py
+++ b/sklearn/ensemble/tests/test_forest.py
@@ -532,14 +532,14 @@ def check_multioutput(name):
     if name in FOREST_CLASSIFIERS:
         with np.errstate(divide="ignore"):
             proba = est.predict_proba(X_test)
-            assert_equal(len(proba), 2)
-            assert_equal(proba[0].shape, (4, 2))
-            assert_equal(proba[1].shape, (4, 4))
+            assert len(proba) == 2
+            assert proba[0].shape == (4, 2)
+            assert proba[1].shape == (4, 4)
 
             log_proba = est.predict_log_proba(X_test)
-            assert_equal(len(log_proba), 2)
-            assert_equal(log_proba[0].shape, (4, 2))
-            assert_equal(log_proba[1].shape, (4, 4))
+            assert len(log_proba) == 2
+            assert log_proba[0].shape == (4, 2)
+            assert log_proba[1].shape == (4, 4)
 
 
 @pytest.mark.filterwarnings('ignore:The default value of n_estimators')
@@ -548,6 +548,37 @@ def test_multioutput(name):
     check_multioutput(name)
 
 
+@pytest.mark.filterwarnings('ignore:The default value of n_estimators')
+@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
+def test_multioutput_string(name):
+    # Check estimators on multi-output problems with string outputs.
+
+    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1],
+               [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
+    y_train = [["red", "blue"], ["red", "blue"], ["red", "blue"],
+               ["green", "green"], ["green", "green"], ["green", "green"],
+               ["red", "purple"], ["red", "purple"], ["red", "purple"],
+               ["green", "yellow"], ["green", "yellow"], ["green", "yellow"]]
+    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
+    y_test = [["red", "blue"], ["green", "green"],
+              ["red", "purple"], ["green", "yellow"]]
+
+    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
+    y_pred = est.fit(X_train, y_train).predict(X_test)
+    assert_array_equal(y_pred, y_test)
+
+    with np.errstate(divide="ignore"):
+        proba = est.predict_proba(X_test)
+        assert len(proba) == 2
+        assert proba[0].shape == (4, 2)
+        assert proba[1].shape == (4, 4)
+
+        log_proba = est.predict_log_proba(X_test)
+        assert len(log_proba) == 2
+        assert log_proba[0].shape == (4, 2)
+        assert log_proba[1].shape == (4, 4)
+
+
 def check_classes_shape(name):
     # Test that n_classes_ and classes_ have proper shape.
     ForestClassifier = FOREST_CLASSIFIERS[name]

```

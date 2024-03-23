# scikit-learn__scikit-learn-13313

* repo: scikit-learn/scikit-learn
* base_commit: `cdfca8cba33be63ef50ba9e14d8823cc551baf92`

## Problem statement

check_class_weight_balanced_classifiers is never run?!
> git grep check_class_weight_balanced_classifiers
sklearn/utils/estimator_checks.py:def check_class_weight_balanced_classifiers(name, Classifier, X_train, y_train,

Same for ``check_class_weight_balanced_linear_classifier``


## Patch

```diff
diff --git a/sklearn/utils/estimator_checks.py b/sklearn/utils/estimator_checks.py
--- a/sklearn/utils/estimator_checks.py
+++ b/sklearn/utils/estimator_checks.py
@@ -1992,7 +1992,10 @@ def check_class_weight_balanced_linear_classifier(name, Classifier):
     classifier.set_params(class_weight=class_weight)
     coef_manual = classifier.fit(X, y).coef_.copy()
 
-    assert_allclose(coef_balanced, coef_manual)
+    assert_allclose(coef_balanced, coef_manual,
+                    err_msg="Classifier %s is not computing"
+                    " class_weight=balanced properly."
+                    % name)
 
 
 @ignore_warnings(category=(DeprecationWarning, FutureWarning))

```

## Test Patch

```diff
diff --git a/sklearn/utils/tests/test_estimator_checks.py b/sklearn/utils/tests/test_estimator_checks.py
--- a/sklearn/utils/tests/test_estimator_checks.py
+++ b/sklearn/utils/tests/test_estimator_checks.py
@@ -13,6 +13,8 @@
                                    assert_equal, ignore_warnings,
                                    assert_warns, assert_raises)
 from sklearn.utils.estimator_checks import check_estimator
+from sklearn.utils.estimator_checks \
+    import check_class_weight_balanced_linear_classifier
 from sklearn.utils.estimator_checks import set_random_state
 from sklearn.utils.estimator_checks import set_checking_parameters
 from sklearn.utils.estimator_checks import check_estimators_unfitted
@@ -190,6 +192,28 @@ def predict(self, X):
         return np.ones(X.shape[0])
 
 
+class BadBalancedWeightsClassifier(BaseBadClassifier):
+    def __init__(self, class_weight=None):
+        self.class_weight = class_weight
+
+    def fit(self, X, y):
+        from sklearn.preprocessing import LabelEncoder
+        from sklearn.utils import compute_class_weight
+
+        label_encoder = LabelEncoder().fit(y)
+        classes = label_encoder.classes_
+        class_weight = compute_class_weight(self.class_weight, classes, y)
+
+        # Intentionally modify the balanced class_weight
+        # to simulate a bug and raise an exception
+        if self.class_weight == "balanced":
+            class_weight += 1.
+
+        # Simply assigning coef_ to the class_weight
+        self.coef_ = class_weight
+        return self
+
+
 class BadTransformerWithoutMixin(BaseEstimator):
     def fit(self, X, y=None):
         X = check_array(X)
@@ -471,6 +495,16 @@ def run_tests_without_pytest():
     runner.run(suite)
 
 
+def test_check_class_weight_balanced_linear_classifier():
+    # check that ill-computed balanced weights raises an exception
+    assert_raises_regex(AssertionError,
+                        "Classifier estimator_name is not computing"
+                        " class_weight=balanced properly.",
+                        check_class_weight_balanced_linear_classifier,
+                        'estimator_name',
+                        BadBalancedWeightsClassifier)
+
+
 if __name__ == '__main__':
     # This module is run as a script to check that we have no dependency on
     # pytest for estimator checks.

```

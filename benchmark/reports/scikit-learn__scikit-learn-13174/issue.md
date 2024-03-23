# scikit-learn__scikit-learn-13174

* repo: scikit-learn/scikit-learn
* base_commit: `09bc27630fb8feea2f10627dce25e93cd6ff258a`

## Problem statement

Minimize validation of X in ensembles with a base estimator
Currently AdaBoost\* requires `X` to be an array or sparse matrix of numerics. However, since the data is not processed directly by `AdaBoost*` but by its base estimator (on which `fit`, `predict_proba` and `predict` may be called), we should not need to constrain the data that much, allowing for `X` to be a list of text blobs or similar.

Similar may apply to other ensemble methods.

Derived from #7767.



## Patch

```diff
diff --git a/sklearn/ensemble/weight_boosting.py b/sklearn/ensemble/weight_boosting.py
--- a/sklearn/ensemble/weight_boosting.py
+++ b/sklearn/ensemble/weight_boosting.py
@@ -30,16 +30,15 @@
 from scipy.special import xlogy
 
 from .base import BaseEnsemble
-from ..base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier
+from ..base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor
 
-from .forest import BaseForest
 from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
-from ..tree.tree import BaseDecisionTree
-from ..tree._tree import DTYPE
-from ..utils import check_array, check_X_y, check_random_state
+from ..utils import check_array, check_random_state, check_X_y, safe_indexing
 from ..utils.extmath import stable_cumsum
 from ..metrics import accuracy_score, r2_score
-from sklearn.utils.validation import has_fit_parameter, check_is_fitted
+from ..utils.validation import check_is_fitted
+from ..utils.validation import has_fit_parameter
+from ..utils.validation import _num_samples
 
 __all__ = [
     'AdaBoostClassifier',
@@ -70,6 +69,26 @@ def __init__(self,
         self.learning_rate = learning_rate
         self.random_state = random_state
 
+    def _validate_data(self, X, y=None):
+
+        # Accept or convert to these sparse matrix formats so we can
+        # use safe_indexing
+        accept_sparse = ['csr', 'csc']
+        if y is None:
+            ret = check_array(X,
+                              accept_sparse=accept_sparse,
+                              ensure_2d=False,
+                              allow_nd=True,
+                              dtype=None)
+        else:
+            ret = check_X_y(X, y,
+                            accept_sparse=accept_sparse,
+                            ensure_2d=False,
+                            allow_nd=True,
+                            dtype=None,
+                            y_numeric=is_regressor(self))
+        return ret
+
     def fit(self, X, y, sample_weight=None):
         """Build a boosted classifier/regressor from the training set (X, y).
 
@@ -77,9 +96,7 @@ def fit(self, X, y, sample_weight=None):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
-            forced to DTYPE from tree._tree if the base classifier of this
-            ensemble weighted boosting classifier is a tree or forest.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         y : array-like of shape = [n_samples]
             The target values (class labels in classification, real numbers in
@@ -97,22 +114,12 @@ def fit(self, X, y, sample_weight=None):
         if self.learning_rate <= 0:
             raise ValueError("learning_rate must be greater than zero")
 
-        if (self.base_estimator is None or
-                isinstance(self.base_estimator, (BaseDecisionTree,
-                                                 BaseForest))):
-            dtype = DTYPE
-            accept_sparse = 'csc'
-        else:
-            dtype = None
-            accept_sparse = ['csr', 'csc']
-
-        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
-                         y_numeric=is_regressor(self))
+        X, y = self._validate_data(X, y)
 
         if sample_weight is None:
             # Initialize weights to 1 / n_samples
-            sample_weight = np.empty(X.shape[0], dtype=np.float64)
-            sample_weight[:] = 1. / X.shape[0]
+            sample_weight = np.empty(_num_samples(X), dtype=np.float64)
+            sample_weight[:] = 1. / _num_samples(X)
         else:
             sample_weight = check_array(sample_weight, ensure_2d=False)
             # Normalize existing weights
@@ -216,7 +223,7 @@ def staged_score(self, X, y, sample_weight=None):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         y : array-like, shape = [n_samples]
             Labels for X.
@@ -228,6 +235,8 @@ def staged_score(self, X, y, sample_weight=None):
         -------
         z : float
         """
+        X = self._validate_data(X)
+
         for y_pred in self.staged_predict(X):
             if is_classifier(self):
                 yield accuracy_score(y, y_pred, sample_weight=sample_weight)
@@ -259,18 +268,6 @@ def feature_importances_(self):
                 "since base_estimator does not have a "
                 "feature_importances_ attribute")
 
-    def _validate_X_predict(self, X):
-        """Ensure that X is in the proper format"""
-        if (self.base_estimator is None or
-                isinstance(self.base_estimator,
-                           (BaseDecisionTree, BaseForest))):
-            X = check_array(X, accept_sparse='csr', dtype=DTYPE)
-
-        else:
-            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
-
-        return X
-
 
 def _samme_proba(estimator, n_classes, X):
     """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
@@ -391,7 +388,7 @@ def fit(self, X, y, sample_weight=None):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         y : array-like of shape = [n_samples]
             The target values (class labels).
@@ -442,8 +439,7 @@ def _boost(self, iboost, X, y, sample_weight, random_state):
             The index of the current boost iteration.
 
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
-            The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            The training input samples.
 
         y : array-like of shape = [n_samples]
             The target values (class labels).
@@ -591,13 +587,15 @@ def predict(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
         y : array of shape = [n_samples]
             The predicted classes.
         """
+        X = self._validate_data(X)
+
         pred = self.decision_function(X)
 
         if self.n_classes_ == 2:
@@ -618,13 +616,16 @@ def staged_predict(self, X):
         Parameters
         ----------
         X : array-like of shape = [n_samples, n_features]
-            The input samples.
+            The input samples. Sparse matrix can be CSC, CSR, COO,
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
         y : generator of array, shape = [n_samples]
             The predicted classes.
         """
+        X = self._validate_data(X)
+
         n_classes = self.n_classes_
         classes = self.classes_
 
@@ -644,7 +645,7 @@ def decision_function(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -657,7 +658,7 @@ def decision_function(self, X):
             class in ``classes_``, respectively.
         """
         check_is_fitted(self, "n_classes_")
-        X = self._validate_X_predict(X)
+        X = self._validate_data(X)
 
         n_classes = self.n_classes_
         classes = self.classes_[:, np.newaxis]
@@ -687,7 +688,7 @@ def staged_decision_function(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -700,7 +701,7 @@ def staged_decision_function(self, X):
             class in ``classes_``, respectively.
         """
         check_is_fitted(self, "n_classes_")
-        X = self._validate_X_predict(X)
+        X = self._validate_data(X)
 
         n_classes = self.n_classes_
         classes = self.classes_[:, np.newaxis]
@@ -741,7 +742,7 @@ def predict_proba(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -750,12 +751,12 @@ def predict_proba(self, X):
             outputs is the same of that of the `classes_` attribute.
         """
         check_is_fitted(self, "n_classes_")
+        X = self._validate_data(X)
 
         n_classes = self.n_classes_
-        X = self._validate_X_predict(X)
 
         if n_classes == 1:
-            return np.ones((X.shape[0], 1))
+            return np.ones((_num_samples(X), 1))
 
         if self.algorithm == 'SAMME.R':
             # The weights are all 1. for SAMME.R
@@ -790,7 +791,7 @@ def staged_predict_proba(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -798,7 +799,7 @@ def staged_predict_proba(self, X):
             The class probabilities of the input samples. The order of
             outputs is the same of that of the `classes_` attribute.
         """
-        X = self._validate_X_predict(X)
+        X = self._validate_data(X)
 
         n_classes = self.n_classes_
         proba = None
@@ -837,7 +838,7 @@ def predict_log_proba(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -845,6 +846,7 @@ def predict_log_proba(self, X):
             The class probabilities of the input samples. The order of
             outputs is the same of that of the `classes_` attribute.
         """
+        X = self._validate_data(X)
         return np.log(self.predict_proba(X))
 
 
@@ -937,7 +939,7 @@ def fit(self, X, y, sample_weight=None):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         y : array-like of shape = [n_samples]
             The target values (real numbers).
@@ -975,8 +977,7 @@ def _boost(self, iboost, X, y, sample_weight, random_state):
             The index of the current boost iteration.
 
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
-            The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            The training input samples.
 
         y : array-like of shape = [n_samples]
             The target values (class labels in classification, real numbers in
@@ -1008,14 +1009,16 @@ def _boost(self, iboost, X, y, sample_weight, random_state):
         # For NumPy >= 1.7.0 use np.random.choice
         cdf = stable_cumsum(sample_weight)
         cdf /= cdf[-1]
-        uniform_samples = random_state.random_sample(X.shape[0])
+        uniform_samples = random_state.random_sample(_num_samples(X))
         bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
         # searchsorted returns a scalar
         bootstrap_idx = np.array(bootstrap_idx, copy=False)
 
         # Fit on the bootstrapped sample and obtain a prediction
         # for all samples in the training set
-        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
+        X_ = safe_indexing(X, bootstrap_idx)
+        y_ = safe_indexing(y, bootstrap_idx)
+        estimator.fit(X_, y_)
         y_predict = estimator.predict(X)
 
         error_vect = np.abs(y_predict - y)
@@ -1067,10 +1070,10 @@ def _get_median_predict(self, X, limit):
         median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
         median_idx = median_or_above.argmax(axis=1)
 
-        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]
+        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
 
         # Return median predictions
-        return predictions[np.arange(X.shape[0]), median_estimators]
+        return predictions[np.arange(_num_samples(X)), median_estimators]
 
     def predict(self, X):
         """Predict regression value for X.
@@ -1082,7 +1085,7 @@ def predict(self, X):
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
             The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
 
         Returns
         -------
@@ -1090,7 +1093,7 @@ def predict(self, X):
             The predicted regression values.
         """
         check_is_fitted(self, "estimator_weights_")
-        X = self._validate_X_predict(X)
+        X = self._validate_data(X)
 
         return self._get_median_predict(X, len(self.estimators_))
 
@@ -1107,8 +1110,7 @@ def staged_predict(self, X):
         Parameters
         ----------
         X : {array-like, sparse matrix} of shape = [n_samples, n_features]
-            The training input samples. Sparse matrix can be CSC, CSR, COO,
-            DOK, or LIL. DOK and LIL are converted to CSR.
+            The training input samples.
 
         Returns
         -------
@@ -1116,7 +1118,7 @@ def staged_predict(self, X):
             The predicted regression values.
         """
         check_is_fitted(self, "estimator_weights_")
-        X = self._validate_X_predict(X)
+        X = self._validate_data(X)
 
         for i, _ in enumerate(self.estimators_, 1):
             yield self._get_median_predict(X, limit=i)

```

## Test Patch

```diff
diff --git a/sklearn/ensemble/tests/test_weight_boosting.py b/sklearn/ensemble/tests/test_weight_boosting.py
--- a/sklearn/ensemble/tests/test_weight_boosting.py
+++ b/sklearn/ensemble/tests/test_weight_boosting.py
@@ -471,7 +471,6 @@ def fit(self, X, y, sample_weight=None):
 def test_sample_weight_adaboost_regressor():
     """
     AdaBoostRegressor should work without sample_weights in the base estimator
-
     The random weighted sampling is done internally in the _boost method in
     AdaBoostRegressor.
     """
@@ -486,3 +485,27 @@ def predict(self, X):
     boost = AdaBoostRegressor(DummyEstimator(), n_estimators=3)
     boost.fit(X, y_regr)
     assert_equal(len(boost.estimator_weights_), len(boost.estimator_errors_))
+
+
+def test_multidimensional_X():
+    """
+    Check that the AdaBoost estimators can work with n-dimensional
+    data matrix
+    """
+
+    from sklearn.dummy import DummyClassifier, DummyRegressor
+
+    rng = np.random.RandomState(0)
+
+    X = rng.randn(50, 3, 3)
+    yc = rng.choice([0, 1], 50)
+    yr = rng.randn(50)
+
+    boost = AdaBoostClassifier(DummyClassifier(strategy='most_frequent'))
+    boost.fit(X, yc)
+    boost.predict(X)
+    boost.predict_proba(X)
+
+    boost = AdaBoostRegressor(DummyRegressor())
+    boost.fit(X, yr)
+    boost.predict(X)

```

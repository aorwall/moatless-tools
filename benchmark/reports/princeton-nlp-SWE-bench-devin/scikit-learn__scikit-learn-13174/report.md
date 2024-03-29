# scikit-learn__scikit-learn-13174

| **scikit-learn/scikit-learn** | `09bc27630fb8feea2f10627dce25e93cd6ff258a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 180 |
| **Avg pos** | 680.0 |
| **Min pos** | 1 |
| **Max pos** | 56 |
| **Top file pos** | 1 |
| **Missing snippets** | 29 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/ensemble/weight_boosting.py | 33 | 42 | 14 | 1 | 4965
| sklearn/ensemble/weight_boosting.py | 73 | 73 | 56 | 1 | 24563
| sklearn/ensemble/weight_boosting.py | 80 | 82 | 56 | 1 | 24563
| sklearn/ensemble/weight_boosting.py | 100 | 115 | 56 | 1 | 24563
| sklearn/ensemble/weight_boosting.py | 219 | 219 | - | 1 | -
| sklearn/ensemble/weight_boosting.py | 231 | 231 | - | 1 | -
| sklearn/ensemble/weight_boosting.py | 262 | 273 | - | 1 | -
| sklearn/ensemble/weight_boosting.py | 394 | 394 | 49 | 1 | 19212
| sklearn/ensemble/weight_boosting.py | 445 | 446 | - | 1 | -
| sklearn/ensemble/weight_boosting.py | 594 | 594 | 16 | 1 | 5588
| sklearn/ensemble/weight_boosting.py | 621 | 621 | 4 | 1 | 1151
| sklearn/ensemble/weight_boosting.py | 647 | 647 | 44 | 1 | 16964
| sklearn/ensemble/weight_boosting.py | 660 | 660 | 44 | 1 | 16964
| sklearn/ensemble/weight_boosting.py | 690 | 690 | 15 | 1 | 5403
| sklearn/ensemble/weight_boosting.py | 703 | 703 | 15 | 1 | 5403
| sklearn/ensemble/weight_boosting.py | 744 | 744 | 53 | 1 | 22973
| sklearn/ensemble/weight_boosting.py | 753 | 756 | 53 | 1 | 22973
| sklearn/ensemble/weight_boosting.py | 793 | 793 | 10 | 1 | 4149
| sklearn/ensemble/weight_boosting.py | 801 | 801 | 10 | 1 | 4149
| sklearn/ensemble/weight_boosting.py | 840 | 840 | 22 | 1 | 7710
| sklearn/ensemble/weight_boosting.py | 848 | 848 | 22 | 1 | 7710
| sklearn/ensemble/weight_boosting.py | 940 | 940 | 18 | 1 | 6130
| sklearn/ensemble/weight_boosting.py | 978 | 979 | 32 | 1 | 11635
| sklearn/ensemble/weight_boosting.py | 1011 | 1018 | 32 | 1 | 11635
| sklearn/ensemble/weight_boosting.py | 1070 | 1073 | 1 | 1 | 180
| sklearn/ensemble/weight_boosting.py | 1085 | 1085 | 23 | 1 | 7886
| sklearn/ensemble/weight_boosting.py | 1093 | 1093 | 23 | 1 | 7886
| sklearn/ensemble/weight_boosting.py | 1110 | 1111 | 6 | 1 | 2444
| sklearn/ensemble/weight_boosting.py | 1119 | 1119 | 6 | 1 | 2444


## Problem Statement

```
Minimize validation of X in ensembles with a base estimator
Currently AdaBoost\* requires `X` to be an array or sparse matrix of numerics. However, since the data is not processed directly by `AdaBoost*` but by its base estimator (on which `fit`, `predict_proba` and `predict` may be called), we should not need to constrain the data that much, allowing for `X` to be a list of text blobs or similar.

Similar may apply to other ensemble methods.

Derived from #7767.


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/ensemble/weight_boosting.py** | 1057 | 1073| 180 | 180 | 8697 | 
| 2 | 2 sklearn/ensemble/gradient_boosting.py | 1418 | 1477| 575 | 755 | 28928 | 
| 3 | 3 sklearn/tree/tree.py | 387 | 403| 176 | 931 | 42455 | 
| **-> 4 <-** | **3 sklearn/ensemble/weight_boosting.py** | 608 | 638| 220 | 1151 | 42455 | 
| 5 | 4 sklearn/utils/validation.py | 596 | 704| 1067 | 2218 | 50330 | 
| **-> 6 <-** | **4 sklearn/ensemble/weight_boosting.py** | 1097 | 1123| 226 | 2444 | 50330 | 
| 7 | **4 sklearn/ensemble/weight_boosting.py** | 414 | 430| 182 | 2626 | 50330 | 
| 8 | 4 sklearn/ensemble/gradient_boosting.py | 1649 | 2141| 322 | 2948 | 50330 | 
| 9 | 5 sklearn/ensemble/bagging.py | 246 | 343| 783 | 3731 | 58314 | 
| **-> 10 <-** | **5 sklearn/ensemble/weight_boosting.py** | 777 | 827| 418 | 4149 | 58314 | 
| 11 | 5 sklearn/ensemble/gradient_boosting.py | 1571 | 1590| 218 | 4367 | 58314 | 
| 12 | 5 sklearn/ensemble/gradient_boosting.py | 2452 | 2474| 195 | 4562 | 58314 | 
| 13 | 6 sklearn/ensemble/forest.py | 2010 | 2027| 133 | 4695 | 76379 | 
| **-> 14 <-** | **6 sklearn/ensemble/weight_boosting.py** | 1 | 47| 270 | 4965 | 76379 | 
| **-> 15 <-** | **6 sklearn/ensemble/weight_boosting.py** | 680 | 731| 438 | 5403 | 76379 | 
| **-> 16 <-** | **6 sklearn/ensemble/weight_boosting.py** | 584 | 606| 185 | 5588 | 76379 | 
| 17 | 6 sklearn/utils/validation.py | 705 | 728| 276 | 5864 | 76379 | 
| **-> 18 <-** | **6 sklearn/ensemble/weight_boosting.py** | 933 | 964| 266 | 6130 | 76379 | 
| 19 | 6 sklearn/ensemble/bagging.py | 990 | 1020| 248 | 6378 | 76379 | 
| 20 | 7 sklearn/decomposition/online_lda.py | 458 | 511| 354 | 6732 | 82939 | 
| 21 | 8 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 7525 | 84184 | 
| **-> 22 <-** | **8 sklearn/ensemble/weight_boosting.py** | 829 | 848| 185 | 7710 | 84184 | 
| **-> 23 <-** | **8 sklearn/ensemble/weight_boosting.py** | 1075 | 1095| 176 | 7886 | 84184 | 
| 24 | 9 sklearn/svm/base.py | 450 | 475| 275 | 8161 | 92374 | 
| 25 | **9 sklearn/ensemble/weight_boosting.py** | 50 | 71| 126 | 8287 | 92374 | 
| 26 | 9 sklearn/ensemble/bagging.py | 345 | 390| 383 | 8670 | 92374 | 
| 27 | 10 examples/ensemble/plot_adaboost_multiclass.py | 1 | 89| 748 | 9418 | 93388 | 
| 28 | 11 sklearn/utils/estimator_checks.py | 772 | 786| 154 | 9572 | 114883 | 
| 29 | 12 examples/ensemble/plot_adaboost_hastie_10_2.py | 1 | 90| 749 | 10321 | 115849 | 
| 30 | **12 sklearn/ensemble/weight_boosting.py** | 536 | 582| 397 | 10718 | 115849 | 
| 31 | 12 sklearn/ensemble/bagging.py | 218 | 244| 212 | 10930 | 115849 | 
| **-> 32 <-** | **12 sklearn/ensemble/weight_boosting.py** | 966 | 1055| 705 | 11635 | 115849 | 
| 33 | 13 sklearn/model_selection/_validation.py | 750 | 782| 348 | 11983 | 128918 | 
| 34 | **13 sklearn/ensemble/weight_boosting.py** | 851 | 931| 644 | 12627 | 128918 | 
| 35 | 13 sklearn/ensemble/gradient_boosting.py | 1208 | 1291| 798 | 13425 | 128918 | 
| 36 | 14 sklearn/preprocessing/imputation.py | 4 | 29| 141 | 13566 | 131841 | 
| 37 | **14 sklearn/ensemble/weight_boosting.py** | 295 | 385| 736 | 14302 | 131841 | 
| 38 | 14 sklearn/ensemble/bagging.py | 585 | 620| 295 | 14597 | 131841 | 
| 39 | 14 sklearn/ensemble/bagging.py | 813 | 923| 1112 | 15709 | 131841 | 
| 40 | 14 sklearn/ensemble/gradient_boosting.py | 1962 | 1972| 124 | 15833 | 131841 | 
| 41 | 14 sklearn/ensemble/gradient_boosting.py | 2116 | 2473| 215 | 16048 | 131841 | 
| 42 | 14 sklearn/utils/estimator_checks.py | 2044 | 2064| 200 | 16248 | 131841 | 
| 43 | 14 sklearn/ensemble/gradient_boosting.py | 1 | 58| 349 | 16597 | 131841 | 
| **-> 44 <-** | **14 sklearn/ensemble/weight_boosting.py** | 640 | 678| 367 | 16964 | 131841 | 
| 45 | 14 sklearn/ensemble/forest.py | 344 | 358| 147 | 17111 | 131841 | 
| 46 | 14 sklearn/model_selection/_validation.py | 840 | 885| 462 | 17573 | 131841 | 
| 47 | 14 sklearn/ensemble/bagging.py | 431 | 553| 1185 | 18758 | 131841 | 
| 48 | 14 sklearn/ensemble/bagging.py | 622 | 650| 231 | 18989 | 131841 | 
| **-> 49 <-** | **14 sklearn/ensemble/weight_boosting.py** | 387 | 412| 223 | 19212 | 131841 | 
| 50 | 15 sklearn/neighbors/base.py | 162 | 247| 757 | 19969 | 139557 | 
| 51 | 15 sklearn/ensemble/gradient_boosting.py | 2021 | 2038| 144 | 20113 | 139557 | 
| 52 | 15 sklearn/ensemble/gradient_boosting.py | 2145 | 2386| 2474 | 22587 | 139557 | 
| **-> 53 <-** | **15 sklearn/ensemble/weight_boosting.py** | 733 | 775| 386 | 22973 | 139557 | 
| 54 | 15 sklearn/ensemble/gradient_boosting.py | 2040 | 2060| 177 | 23150 | 139557 | 
| 55 | 16 examples/ensemble/plot_adaboost_twoclass.py | 1 | 85| 705 | 23855 | 140427 | 
| **-> 56 <-** | **16 sklearn/ensemble/weight_boosting.py** | 73 | 166| 708 | 24563 | 140427 | 
| 57 | 16 sklearn/svm/base.py | 326 | 346| 223 | 24786 | 140427 | 
| 58 | 16 sklearn/model_selection/_validation.py | 640 | 749| 961 | 25747 | 140427 | 
| 59 | 17 sklearn/ensemble/iforest.py | 193 | 294| 857 | 26604 | 144363 | 
| 60 | 18 examples/plot_isotonic_regression.py | 1 | 59| 391 | 26995 | 144793 | 
| 61 | 18 sklearn/utils/estimator_checks.py | 2027 | 2041| 211 | 27206 | 144793 | 
| 62 | 18 sklearn/neighbors/base.py | 856 | 873| 166 | 27372 | 144793 | 
| 63 | 18 sklearn/ensemble/gradient_boosting.py | 1593 | 1618| 260 | 27632 | 144793 | 
| 64 | 19 sklearn/ensemble/base.py | 99 | 116| 176 | 27808 | 145932 | 
| 65 | 19 sklearn/ensemble/gradient_boosting.py | 1998 | 2019| 199 | 28007 | 145932 | 
| 66 | 19 sklearn/ensemble/gradient_boosting.py | 2431 | 2450| 160 | 28167 | 145932 | 
| 67 | 20 sklearn/utils/extmath.py | 629 | 657| 160 | 28327 | 152246 | 
| 68 | 20 sklearn/svm/base.py | 428 | 448| 205 | 28532 | 152246 | 
| 69 | **20 sklearn/ensemble/weight_boosting.py** | 478 | 534| 534 | 29066 | 152246 | 
| 70 | 20 sklearn/ensemble/forest.py | 1975 | 2008| 304 | 29370 | 152246 | 
| 71 | 20 sklearn/ensemble/gradient_boosting.py | 1003 | 1040| 373 | 29743 | 152246 | 
| 72 | 20 sklearn/ensemble/gradient_boosting.py | 1686 | 1933| 2499 | 32242 | 152246 | 
| 73 | 20 sklearn/ensemble/gradient_boosting.py | 1154 | 1206| 460 | 32702 | 152246 | 
| 74 | 20 sklearn/ensemble/gradient_boosting.py | 2413 | 2429| 145 | 32847 | 152246 | 
| 75 | 21 sklearn/naive_bayes.py | 38 | 66| 209 | 33056 | 160635 | 
| 76 | 22 examples/ensemble/plot_adaboost_regression.py | 1 | 55| 389 | 33445 | 161052 | 
| 77 | 22 sklearn/ensemble/bagging.py | 392 | 413| 203 | 33648 | 161052 | 
| 78 | 22 sklearn/decomposition/online_lda.py | 615 | 633| 146 | 33794 | 161052 | 
| 79 | 22 sklearn/utils/estimator_checks.py | 1108 | 1128| 257 | 34051 | 161052 | 
| 80 | 22 sklearn/utils/validation.py | 421 | 475| 547 | 34598 | 161052 | 
| 81 | 22 sklearn/ensemble/gradient_boosting.py | 253 | 296| 274 | 34872 | 161052 | 
| 82 | 22 sklearn/utils/estimator_checks.py | 1505 | 1570| 567 | 35439 | 161052 | 
| 83 | 22 sklearn/utils/estimator_checks.py | 482 | 528| 467 | 35906 | 161052 | 
| 84 | 22 sklearn/svm/base.py | 272 | 305| 353 | 36259 | 161052 | 
| 85 | 23 sklearn/ensemble/__init__.py | 1 | 36| 289 | 36548 | 161341 | 
| 86 | 24 sklearn/cluster/birch.py | 6 | 35| 239 | 36787 | 166625 | 
| 87 | 24 sklearn/ensemble/gradient_boosting.py | 2388 | 2411| 322 | 37109 | 166625 | 
| 88 | 24 sklearn/ensemble/bagging.py | 554 | 583| 184 | 37293 | 166625 | 
| 89 | 24 sklearn/ensemble/gradient_boosting.py | 2092 | 2114| 179 | 37472 | 166625 | 
| 90 | 25 sklearn/preprocessing/data.py | 1841 | 1855| 128 | 37600 | 191095 | 
| 91 | 25 sklearn/ensemble/gradient_boosting.py | 1118 | 1152| 327 | 37927 | 191095 | 
| 92 | 26 sklearn/feature_selection/rfe.py | 225 | 261| 249 | 38176 | 195671 | 
| 93 | 26 sklearn/ensemble/bagging.py | 415 | 428| 134 | 38310 | 195671 | 
| 94 | 26 sklearn/model_selection/_validation.py | 216 | 255| 409 | 38719 | 195671 | 
| 95 | 26 sklearn/decomposition/online_lda.py | 584 | 613| 249 | 38968 | 195671 | 
| 96 | 26 sklearn/ensemble/bagging.py | 1 | 57| 380 | 39348 | 195671 | 
| 97 | 27 sklearn/feature_selection/variance_threshold.py | 48 | 83| 255 | 39603 | 196266 | 
| 98 | 27 sklearn/ensemble/gradient_boosting.py | 1935 | 1960| 307 | 39910 | 196266 | 
| 99 | 27 sklearn/preprocessing/data.py | 959 | 976| 136 | 40046 | 196266 | 
| 100 | 27 sklearn/decomposition/online_lda.py | 513 | 582| 571 | 40617 | 196266 | 
| 101 | 27 sklearn/ensemble/iforest.py | 296 | 317| 220 | 40837 | 196266 | 
| 102 | 27 sklearn/cluster/birch.py | 538 | 551| 140 | 40977 | 196266 | 
| 103 | 27 sklearn/cluster/birch.py | 553 | 574| 164 | 41141 | 196266 | 
| 104 | 27 sklearn/preprocessing/data.py | 940 | 957| 134 | 41275 | 196266 | 
| 105 | 28 examples/ensemble/plot_gradient_boosting_regularization.py | 1 | 80| 696 | 41971 | 196990 | 
| 106 | 28 sklearn/ensemble/forest.py | 300 | 342| 441 | 42412 | 196990 | 
| 107 | 29 sklearn/manifold/isomap.py | 183 | 199| 118 | 42530 | 198792 | 
| 108 | 29 sklearn/utils/estimator_checks.py | 2309 | 2342| 319 | 42849 | 198792 | 
| 109 | **29 sklearn/ensemble/weight_boosting.py** | 237 | 272| 279 | 43128 | 198792 | 


### Hint

```
That could be applied to any meta-estimator that uses a base estimator, right?

Yes, it could be. I didn't have time when I wrote this issue to check the applicability to other ensembles.

Updated title and description

@jnothman I think that we have two options.
- Validate the input early as it is now and introduce a new parameter `check_input`  in `fit`, `predict`, etc with default vaule `True` in order to preserve the current behavior.  The `check_input` could be in the constrcutor.
- Relax the validation in the ensemble and let base estimator to handle the validation.

What do you think? I'll sent a PR.

IMO assuming the base estimator manages validation is fine.

Is this still open ? can I work on it?

@Chaitya62 I didn't have the time to work on this. So, go ahead.
@chkoar onit!
After reading code for 2 days and trying to understand what actually needs to be changed I figured out that in that a call to check_X_y is being made which is forcing X to be 2d now for the patch should I do what @chkoar  suggested ? 
As in let the base estimator handle validation? Yes, IMO

On 5 December 2016 at 06:46, Chaitya Shah <notifications@github.com> wrote:

> After reading code for 2 days and trying to understand what actually needs
> to be changed I figured out that in that a call to check_X_y is being made
> which is forcing X to be 2d now for the patch should I do what @chkoar
> <https://github.com/chkoar> suggested ?
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/7768#issuecomment-264726009>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz69Z4CUcCqOlkaOc0xpln9o1ovc85ks5rExiygaJpZM4KiQ_P>
> .
>

Cool I ll submit a PR soon 
@Chaitya62, Let me inform if you are not working on this anymore. I want to work on this. 
@devanshdalal I am working  on it have a minor  issue which I hope I ll soon solve
@Chaitya62 Are you still working on this?

@dalmia go ahead work on it I am not able to test my code properly

@Chaitya62 Thanks!
I'd like to work on this, if that's ok
As a first step, I tried looking at the behavior of meta-estimators when passed a 3D tensor. Looks like almost all meta-estimators which accept a base estimator fail :

\`\`\`
>>> pytest -sx -k 'test_meta_estimators' sklearn/tests/test_common.py
<....>
AdaBoostClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
AdaBoostRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
BaggingClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
BaggingRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
ExtraTreesClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
ExtraTreesRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data

Skipping GradientBoostingClassifier - 'base_estimator' key not supported
Skipping GradientBoostingRegressor - 'base_estimator' key not supported
IsolationForest raised error 'default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.' when parsing data   

RANSACRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data                                     
RandomForestClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
RandomForestRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
\`\`\`
@jnothman @amueller considering this, should this be a WONTFIX, or should all the meta-estimators be fixed?
Thanks for looking into this. Not all ensembles are meta-estimators. Here
we intend things that should be generic enough to support non-scikit-learn
use-cases: not just dealing with rectangular feature matrices.

On Fri, 14 Sep 2018 at 08:23, Karthik Duddu <notifications@github.com>
wrote:

> As a first step, I tried looking at behavior of meta-estimators when
> passed a 3D tensor. Looks like almost all meta-estimators which accept a
> base estimator fail :
>
> >>> pytest -sx -k 'test_meta_estimators' sklearn/tests/test_common.py
> <....>
> AdaBoostClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> AdaBoostRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> BaggingClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> BaggingRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> ExtraTreesClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> ExtraTreesRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
>
> Skipping GradientBoostingClassifier - 'base_estimator' key not supported
> Skipping GradientBoostingRegressor - 'base_estimator' key not supported
> IsolationForest raised error 'default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.' when parsing data
>
> RANSACRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> RandomForestClassifier raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
> RandomForestRegressor raised error 'Found array with dim 3. Estimator expected <= 2.' when parsing data
>
> @jnothman <https://github.com/jnothman> @amueller
> <https://github.com/amueller> considering this, should this be a WONTFIX,
> or should all the meta-estimators be fixed?
>
> —
> You are receiving this because you were mentioned.
> Reply to this email directly, view it on GitHub
> <https://github.com/scikit-learn/scikit-learn/issues/7768#issuecomment-421171742>,
> or mute the thread
> <https://github.com/notifications/unsubscribe-auth/AAEz68Axs1khuYjm7lM4guYgyf2IlUL_ks5uatrDgaJpZM4KiQ_P>
> .
>

@jnothman  `Adaboost` tests are [testing](https://github.com/scikit-learn/scikit-learn/blob/ff28c42b192aa9aab8b61bc8a56b5ceb1170dec7/sklearn/ensemble/tests/test_weight_boosting.py#L323) the sparsity of the `X`. This means that we should skip these tests in order to relax the validation, right?
Sounds like it as long as it doesn't do other things with X than fit the
base estimator

```

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


## Code snippets

### 1 - sklearn/ensemble/weight_boosting.py:

Start line: 1057, End line: 1073

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]
```
### 2 - sklearn/ensemble/gradient_boosting.py:

Start line: 1418, End line: 1477

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None, monitor=None):
        # ... other code

        if not self._is_initialized():
            # init state
            self._init_state()

            # fit initial model - FIXME make sample_weight optional
            self.init_.fit(X, y, sample_weight)

            # init predictions
            y_pred = self.init_.predict(X)
            begin_at_stage = 0

            # The rng state must be preserved if warm_start is True
            self._rng = check_random_state(self.random_state)

        else:
            # add more estimators to fitted model
            # invariant: warm_start = True
            if self.n_estimators < self.estimators_.shape[0]:
                raise ValueError('n_estimators=%d must be larger or equal to '
                                 'estimators_.shape[0]=%d when '
                                 'warm_start==True'
                                 % (self.n_estimators,
                                    self.estimators_.shape[0]))
            begin_at_stage = self.estimators_.shape[0]
            # The requirements of _decision_function (called in two lines
            # below) are more constrained than fit. It accepts only CSR
            # matrices.
            X = check_array(X, dtype=DTYPE, order="C", accept_sparse='csr')
            y_pred = self._decision_function(X)
            self._resize_state()

        if self.presort is True and issparse(X):
            raise ValueError(
                "Presorting is not supported for sparse matrices.")

        presort = self.presort
        # Allow presort to be 'auto', which means True if the dataset is dense,
        # otherwise it will be False.
        if presort == 'auto':
            presort = not issparse(X)

        X_idx_sorted = None
        if presort:
            X_idx_sorted = np.asfortranarray(np.argsort(X, axis=0),
                                             dtype=np.int32)

        # fit the boosting stages
        n_stages = self._fit_stages(X, y, y_pred, sample_weight, self._rng,
                                    X_val, y_val, sample_weight_val,
                                    begin_at_stage, monitor, X_idx_sorted)

        # change shape of arrays after fit (early-stopping or additional ests)
        if n_stages != self.estimators_.shape[0]:
            self.estimators_ = self.estimators_[:n_stages]
            self.train_score_ = self.train_score_[:n_stages]
            if hasattr(self, 'oob_improvement_'):
                self.oob_improvement_ = self.oob_improvement_[:n_stages]

        self.n_estimators_ = n_stages
        return self
```
### 3 - sklearn/tree/tree.py:

Start line: 387, End line: 403

```python
class BaseDecisionTree(BaseEstimator, metaclass=ABCMeta):

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or
                                X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X
```
### 4 - sklearn/ensemble/weight_boosting.py:

Start line: 608, End line: 638

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(
                    np.argmax(pred, axis=1), axis=0))
```
### 5 - sklearn/utils/validation.py:

Start line: 596, End line: 704

```python
def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True,
              dtype="numeric", order=None, copy=False, force_all_finite=True,
              ensure_2d=True, allow_nd=False, multi_output=False,
              ensure_min_samples=1, ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):
    """Input validation for standard estimators.

    Checks X and y for consistent length, enforces X to be 2D and y 1D. By
    default, X is checked to be non-empty and containing only finite values.
    Standard input checks are also applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2D and sparse y. If the dtype of X is
    object, attempt converting to float, raising on failure.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, boolean or list of string (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse will cause it to be accepted only
        if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.
        The possibilities are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan': accept only np.nan values in X. Values cannot be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2D.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2D y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    # ... other code
```
### 6 - sklearn/ensemble/weight_boosting.py:

Start line: 1097, End line: 1123

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)
```
### 7 - sklearn/ensemble/weight_boosting.py:

Start line: 414, End line: 430

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == 'SAMME.R':
            if not hasattr(self.base_estimator_, 'predict_proba'):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead.")
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)
```
### 8 - sklearn/ensemble/gradient_boosting.py:

Start line: 1649, End line: 2141

```python
class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):

    def apply(self, X):
        """Apply trees in the ensemble to X, return leaf indices.

        .. versionadded:: 0.17

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will
            be converted to a sparse ``csr_matrix``.

        Returns
        -------
        X_leaves : array-like, shape (n_samples, n_estimators, n_classes)
            For each datapoint x in X and for each tree in the ensemble,
            return the index of the leaf x ends up in each estimator.
            In the case of binary classification n_classes is 1.
        """

        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)

        # n_classes will be equal to 1 in the binary classification or the
        # regression case.
        n_estimators, n_classes = self.estimators_.shape
        leaves = np.zeros((X.shape[0], n_estimators, n_classes))

        for i in range(n_estimators):
            for j in range(n_classes):
                estimator = self.estimators_[i, j]
                leaves[:, i, j] = estimator.apply(X, check_input=False)

        return leaves


class GradientBoostingClassifier(BaseGradientBoosting, ClassifierMixin):
```
### 9 - sklearn/ensemble/bagging.py:

Start line: 246, End line: 343

```python
class BaseBagging(BaseEnsemble, metaclass=ABCMeta):

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.

        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
        """
        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)
        # ... other code
```
### 10 - sklearn/ensemble/weight_boosting.py:

Start line: 777, End line: 827

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_proba = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_proba = estimator.predict_proba(X) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba
```
### 14 - sklearn/ensemble/weight_boosting.py:

Start line: 1, End line: 47

```python
"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- ``AdaBoostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.special import xlogy

from .base import BaseEnsemble
from ..base import ClassifierMixin, RegressorMixin, is_regressor, is_classifier

from .forest import BaseForest
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..tree.tree import BaseDecisionTree
from ..tree._tree import DTYPE
from ..utils import check_array, check_X_y, check_random_state
from ..utils.extmath import stable_cumsum
from ..metrics import accuracy_score, r2_score
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

__all__ = [
    'AdaBoostClassifier',
    'AdaBoostRegressor',
]
```
### 15 - sklearn/ensemble/weight_boosting.py:

Start line: 680, End line: 731

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            if self.algorithm == 'SAMME.R':
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm
```
### 16 - sklearn/ensemble/weight_boosting.py:

Start line: 584, End line: 606

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
```
### 18 - sklearn/ensemble/weight_boosting.py:

Start line: 933, End line: 964

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (real numbers).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check loss
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))
```
### 22 - sklearn/ensemble/weight_boosting.py:

Start line: 829, End line: 848

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))
```
### 23 - sklearn/ensemble/weight_boosting.py:

Start line: 1075, End line: 1095

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self, "estimator_weights_")
        X = self._validate_X_predict(X)

        return self._get_median_predict(X, len(self.estimators_))
```
### 25 - sklearn/ensemble/weight_boosting.py:

Start line: 50, End line: 71

```python
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state
```
### 30 - sklearn/ensemble/weight_boosting.py:

Start line: 536, End line: 582

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1. - estimator_error) / estimator_error) +
            np.log(n_classes - 1.))

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * incorrect *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, estimator_weight, estimator_error
```
### 32 - sklearn/ensemble/weight_boosting.py:

Start line: 966, End line: 1055

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression

        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        # For NumPy >= 1.7.0 use np.random.choice
        cdf = stable_cumsum(sample_weight)
        cdf /= cdf[-1]
        uniform_samples = random_state.random_sample(X.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X[bootstrap_idx], y[bootstrap_idx])
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        error_max = error_vect.max()

        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight *= np.power(
                beta,
                (1. - error_vect) * self.learning_rate)

        return sample_weight, estimator_weight, estimator_error
```
### 34 - sklearn/ensemble/weight_boosting.py:

Start line: 851, End line: 931

```python
class AdaBoostRegressor(BaseWeightBoosting, RegressorMixin):
    """An AdaBoost regressor.

    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.

    This class implements the algorithm known as AdaBoost.R2 [2].

    Read more in the :ref:`User Guide <adaboost>`.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required. If ``None``, then
        the base estimator is ``DecisionTreeRegressor(max_depth=3)``

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
        The loss function to use when updating the weights after each
        boosting iteration.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Regression error for each estimator in the boosted ensemble.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.

    See also
    --------
    AdaBoostClassifier, GradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.loss = loss
        self.random_state = random_state
```
### 37 - sklearn/ensemble/weight_boosting.py:

Start line: 295, End line: 385

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.

    See also
    --------
    AdaBoostRegressor, GradientBoostingClassifier,
    sklearn.tree.DecisionTreeClassifier

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
```
### 44 - sklearn/ensemble/weight_boosting.py:

Start line: 640, End line: 678

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(_samme_proba(estimator, n_classes, X)
                       for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            pred = sum((estimator.predict(X) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
```
### 49 - sklearn/ensemble/weight_boosting.py:

Start line: 387, End line: 412

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight)
```
### 53 - sklearn/ensemble/weight_boosting.py:

Start line: 733, End line: 775

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        if self.algorithm == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(_samme_proba(estimator, n_classes, X)
                        for estimator in self.estimators_)
        else:   # self.algorithm == "SAMME"
            proba = sum(estimator.predict_proba(X) * w
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
```
### 56 - sklearn/ensemble/weight_boosting.py:

Start line: 73, End line: 166

```python
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype,
                         y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self
```
### 69 - sklearn/ensemble/weight_boosting.py:

Start line: 478, End line: 534

```python
class AdaBoostClassifier(BaseWeightBoosting, ClassifierMixin):

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * ((n_classes - 1.) / n_classes)
                            * xlogy(y_coding, y_predict_proba).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error
```
### 109 - sklearn/ensemble/weight_boosting.py:

Start line: 237, End line: 272

```python
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X
```

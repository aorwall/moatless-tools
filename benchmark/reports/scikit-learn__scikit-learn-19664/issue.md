# scikit-learn__scikit-learn-19664

* repo: scikit-learn/scikit-learn
* base_commit: `2620a5545a806ee416d9d10e07c2de30cdd9bf20`

## Problem statement

LabelPropagation raises TypeError: A sparse matrix was passed
#### Describe the bug

LabelPropagation (and LabelSpreading) error out for sparse matrices.

#### Steps/Code to Reproduce

```
import sklearn
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.semi_supervised import LabelPropagation

print(sklearn.__version__)

X, y = make_classification()
classifier = LabelPropagation(kernel='knn')
classifier.fit(X, y)
y_pred = classifier.predict(X)

X, y = make_classification()
classifier = LabelPropagation(kernel='knn')
classifier.fit(csr_matrix(X), y)
y_pred = classifier.predict(csr_matrix(X))
```

#### Expected Results

Sparse case should work as does the dense one.

#### Actual Results

```
0.22.2.post1
Traceback (most recent call last):
[...]
TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
```

#### Fix

Changing 

```
        X, y = check_X_y(X, y)
```

in _label_propagation.py line 224 to 

```
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo', 'dok',
                                              'bsr', 'lil', 'dia'])
```

seems to fix the problem for me (BTW: a similar check accepting sparse matrices is done in BaseLabelPropagations predict_proba at line 189). This fix also heals LabelSpreading.

FIX LabelPropagation handling of sparce matrices #17085
#### Reference Issues/PRs

Fixes #17085

#### What does this implement/fix? Explain your changes.

Label propagation and spreading allow to classify using sparse data according to documentation. Tests only covered the dense case. Newly added coverage for sparse matrices allows to reproduce the problem in #17085. The proposed fix in #17085 works for the extended tests.

#### Any other comments?

- Supporting scipy's dok_matrix produces the UserWarning "Can't check dok sparse matrix for nan or inf.". So this format seems to be unsuitable?
- `test_label_propagation_closed_form` fails for sparse matrices 



## Patch

```diff
diff --git a/sklearn/semi_supervised/_label_propagation.py b/sklearn/semi_supervised/_label_propagation.py
--- a/sklearn/semi_supervised/_label_propagation.py
+++ b/sklearn/semi_supervised/_label_propagation.py
@@ -241,7 +241,7 @@ def fit(self, X, y):
 
         Parameters
         ----------
-        X : array-like of shape (n_samples, n_features)
+        X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data, where `n_samples` is the number of samples
             and `n_features` is the number of features.
 
@@ -256,7 +256,12 @@ def fit(self, X, y):
             Returns the instance itself.
         """
         self._validate_params()
-        X, y = self._validate_data(X, y)
+        X, y = self._validate_data(
+            X,
+            y,
+            accept_sparse=["csr", "csc"],
+            reset=True,
+        )
         self.X_ = X
         check_classification_targets(y)
 
@@ -365,7 +370,7 @@ class LabelPropagation(BaseLabelPropagation):
 
     Attributes
     ----------
-    X_ : ndarray of shape (n_samples, n_features)
+    X_ : {array-like, sparse matrix} of shape (n_samples, n_features)
         Input array.
 
     classes_ : ndarray of shape (n_classes,)
@@ -463,7 +468,7 @@ def fit(self, X, y):
 
         Parameters
         ----------
-        X : array-like of shape (n_samples, n_features)
+        X : {array-like, sparse matrix} of shape (n_samples, n_features)
             Training data, where `n_samples` is the number of samples
             and `n_features` is the number of features.
 

```

## Test Patch

```diff
diff --git a/sklearn/semi_supervised/tests/test_label_propagation.py b/sklearn/semi_supervised/tests/test_label_propagation.py
--- a/sklearn/semi_supervised/tests/test_label_propagation.py
+++ b/sklearn/semi_supervised/tests/test_label_propagation.py
@@ -15,6 +15,9 @@
     assert_allclose,
     assert_array_equal,
 )
+from sklearn.utils._testing import _convert_container
+
+CONSTRUCTOR_TYPES = ("array", "sparse_csr", "sparse_csc")
 
 ESTIMATORS = [
     (label_propagation.LabelPropagation, {"kernel": "rbf"}),
@@ -122,9 +125,27 @@ def test_label_propagation_closed_form(global_dtype):
     assert_allclose(expected, clf.label_distributions_, atol=1e-4)
 
 
-def test_convergence_speed():
+@pytest.mark.parametrize("accepted_sparse_type", ["sparse_csr", "sparse_csc"])
+@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
+@pytest.mark.parametrize("dtype", [np.float32, np.float64])
+@pytest.mark.parametrize("Estimator, parameters", ESTIMATORS)
+def test_sparse_input_types(
+    accepted_sparse_type, index_dtype, dtype, Estimator, parameters
+):
+    # This is non-regression test for #17085
+    X = _convert_container([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], accepted_sparse_type)
+    X.data = X.data.astype(dtype, copy=False)
+    X.indices = X.indices.astype(index_dtype, copy=False)
+    X.indptr = X.indptr.astype(index_dtype, copy=False)
+    labels = [0, 1, -1]
+    clf = Estimator(**parameters).fit(X, labels)
+    assert_array_equal(clf.predict([[0.5, 2.5]]), np.array([1]))
+
+
+@pytest.mark.parametrize("constructor_type", CONSTRUCTOR_TYPES)
+def test_convergence_speed(constructor_type):
     # This is a non-regression test for #5774
-    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]])
+    X = _convert_container([[1.0, 0.0], [0.0, 1.0], [1.0, 2.5]], constructor_type)
     y = np.array([0, 1, -1])
     mdl = label_propagation.LabelSpreading(kernel="rbf", max_iter=5000)
     mdl.fit(X, y)

```

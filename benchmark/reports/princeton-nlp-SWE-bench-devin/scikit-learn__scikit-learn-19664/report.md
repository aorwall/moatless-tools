# scikit-learn__scikit-learn-19664

| **scikit-learn/scikit-learn** | `2620a5545a806ee416d9d10e07c2de30cdd9bf20` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 4994 |
| **Any found context length** | 1520 |
| **Avg pos** | 30.0 |
| **Min pos** | 3 |
| **Max pos** | 11 |
| **Top file pos** | 1 |
| **Missing snippets** | 4 |
| **Missing patch files** | 0 |


## Expected patch

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

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/semi_supervised/_label_propagation.py | 244 | 244 | 8 | 1 | 4105
| sklearn/semi_supervised/_label_propagation.py | 259 | 259 | 8 | 1 | 4105
| sklearn/semi_supervised/_label_propagation.py | 368 | 368 | 3 | 1 | 1520
| sklearn/semi_supervised/_label_propagation.py | 466 | 466 | 11 | 1 | 4994


## Problem Statement

```
LabelPropagation raises TypeError: A sparse matrix was passed
#### Describe the bug

LabelPropagation (and LabelSpreading) error out for sparse matrices.

#### Steps/Code to Reproduce

\`\`\`
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
\`\`\`

#### Expected Results

Sparse case should work as does the dense one.

#### Actual Results

\`\`\`
0.22.2.post1
Traceback (most recent call last):
[...]
TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
\`\`\`

#### Fix

Changing 

\`\`\`
        X, y = check_X_y(X, y)
\`\`\`

in _label_propagation.py line 224 to 

\`\`\`
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csr', 'coo', 'dok',
                                              'bsr', 'lil', 'dia'])
\`\`\`

seems to fix the problem for me (BTW: a similar check accepting sparse matrices is done in BaseLabelPropagations predict_proba at line 189). This fix also heals LabelSpreading.

FIX LabelPropagation handling of sparce matrices #17085
#### Reference Issues/PRs

Fixes #17085

#### What does this implement/fix? Explain your changes.

Label propagation and spreading allow to classify using sparse data according to documentation. Tests only covered the dense case. Newly added coverage for sparse matrices allows to reproduce the problem in #17085. The proposed fix in #17085 works for the extended tests.

#### Any other comments?

- Supporting scipy's dok_matrix produces the UserWarning "Can't check dok sparse matrix for nan or inf.". So this format seems to be unsuitable?
- `test_label_propagation_closed_form` fails for sparse matrices 


```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| 1 | **1 sklearn/semi_supervised/_label_propagation.py** | 1 | 73| 616 | 616 | 4814 | 
| 2 | **1 sklearn/semi_supervised/_label_propagation.py** | 607 | 622| 185 | 801 | 4814 | 
| **-> 3 <-** | **1 sklearn/semi_supervised/_label_propagation.py** | 334 | 423| 719 | 1520 | 4814 | 
| 4 | **1 sklearn/semi_supervised/_label_propagation.py** | 579 | 605| 165 | 1685 | 4814 | 
| 5 | **1 sklearn/semi_supervised/_label_propagation.py** | 425 | 459| 228 | 1913 | 4814 | 
| 6 | **1 sklearn/semi_supervised/_label_propagation.py** | 483 | 577| 797 | 2710 | 4814 | 
| 7 | 2 examples/semi_supervised/plot_label_propagation_structure.py | 1 | 106| 620 | 3330 | 5541 | 
| **-> 8 <-** | **2 sklearn/semi_supervised/_label_propagation.py** | 234 | 331| 775 | 4105 | 5541 | 
| 9 | **2 sklearn/semi_supervised/_label_propagation.py** | 145 | 166| 200 | 4305 | 5541 | 
| 10 | **2 sklearn/semi_supervised/_label_propagation.py** | 76 | 143| 538 | 4843 | 5541 | 
| **-> 11 <-** | **2 sklearn/semi_supervised/_label_propagation.py** | 461 | 480| 151 | 4994 | 5541 | 
| 12 | **2 sklearn/semi_supervised/_label_propagation.py** | 168 | 192| 213 | 5207 | 5541 | 
| 13 | 3 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 50 | 124| 689 | 5896 | 6567 | 
| 14 | 4 examples/semi_supervised/plot_label_propagation_digits.py | 1 | 118| 693 | 6589 | 7307 | 
| 15 | 5 examples/release_highlights/plot_release_highlights_1_0_0.py | 167 | 242| 729 | 7318 | 9642 | 
| 16 | 5 examples/semi_supervised/plot_label_propagation_digits_active_learning.py | 1 | 48| 317 | 7635 | 9642 | 
| 17 | 6 sklearn/utils/estimator_checks.py | 817 | 901| 772 | 8407 | 45957 | 
| 18 | 7 benchmarks/bench_sparsify.py | 83 | 107| 153 | 8560 | 46856 | 
| 19 | 8 examples/release_highlights/plot_release_highlights_1_2_0.py | 92 | 167| 718 | 9278 | 48341 | 
| 20 | 9 examples/release_highlights/plot_release_highlights_0_24_0.py | 211 | 265| 510 | 9788 | 50779 | 
| 21 | 10 sklearn/decomposition/_dict_learning.py | 133 | 228| 821 | 10609 | 69730 | 
| 22 | 10 sklearn/decomposition/_dict_learning.py | 368 | 437| 747 | 11356 | 69730 | 
| 23 | 11 examples/miscellaneous/plot_multilabel.py | 113 | 131| 199 | 11555 | 70867 | 
| 24 | 12 sklearn/preprocessing/_label.py | 170 | 263| 741 | 12296 | 78054 | 
| 25 | 12 sklearn/utils/estimator_checks.py | 1807 | 2553| 6231 | 18527 | 78054 | 
| 26 | 12 sklearn/preprocessing/_label.py | 519 | 593| 599 | 19126 | 78054 | 
| 27 | 12 benchmarks/bench_sparsify.py | 1 | 82| 746 | 19872 | 78054 | 
| 28 | 13 sklearn/cluster/_affinity_propagation.py | 88 | 174| 763 | 20635 | 82533 | 
| 29 | 14 examples/release_highlights/plot_release_highlights_0_22_0.py | 195 | 281| 764 | 21399 | 84943 | 
| 30 | 14 sklearn/preprocessing/_label.py | 9 | 33| 142 | 21541 | 84943 | 
| 31 | 14 sklearn/cluster/_affinity_propagation.py | 429 | 467| 292 | 21833 | 84943 | 
| 32 | 14 examples/release_highlights/plot_release_highlights_0_22_0.py | 1 | 89| 749 | 22582 | 84943 | 
| 33 | 15 examples/linear_model/plot_lasso_dense_vs_sparse_data.py | 1 | 88| 759 | 23341 | 85702 | 
| 34 | 15 examples/release_highlights/plot_release_highlights_1_2_0.py | 1 | 90| 766 | 24107 | 85702 | 
| 35 | 16 examples/release_highlights/plot_release_highlights_0_23_0.py | 98 | 175| 789 | 24896 | 87484 | 
| 36 | 16 sklearn/preprocessing/_label.py | 425 | 518| 766 | 25662 | 87484 | 
| 37 | 16 examples/release_highlights/plot_release_highlights_0_23_0.py | 1 | 97| 857 | 26519 | 87484 | 
| 38 | 16 sklearn/cluster/_affinity_propagation.py | 294 | 427| 1166 | 27685 | 87484 | 
| 39 | 17 examples/svm/plot_svm_scale_c.py | 105 | 173| 676 | 28361 | 89100 | 
| 40 | 17 examples/release_highlights/plot_release_highlights_0_24_0.py | 121 | 210| 777 | 29138 | 89100 | 
| 41 | 18 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 63| 484 | 29622 | 90111 | 
| 42 | 19 sklearn/svm/_base.py | 609 | 647| 325 | 29947 | 99702 | 
| 43 | 19 sklearn/svm/_base.py | 471 | 506| 255 | 30202 | 99702 | 
| 44 | 19 examples/release_highlights/plot_release_highlights_0_22_0.py | 90 | 193| 896 | 31098 | 99702 | 
| 45 | 20 sklearn/utils/validation.py | 547 | 587| 292 | 31390 | 116645 | 
| 46 | 20 sklearn/cluster/_affinity_propagation.py | 567 | 587| 173 | 31563 | 116645 | 
| 47 | 21 examples/cluster/plot_affinity_propagation.py | 1 | 75| 554 | 32117 | 117199 | 
| 48 | 21 examples/miscellaneous/plot_multilabel.py | 1 | 51| 449 | 32566 | 117199 | 
| 49 | 22 sklearn/utils/fixes.py | 1 | 46| 303 | 32869 | 118942 | 
| 50 | 23 examples/multioutput/plot_classifier_chain_yeast.py | 1 | 73| 732 | 33601 | 120000 | 
| 51 | 24 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 42 | 93| 619 | 34220 | 120929 | 
| 52 | 25 sklearn/decomposition/_lda.py | 543 | 569| 175 | 34395 | 128381 | 
| 53 | 25 sklearn/svm/_base.py | 930 | 964| 249 | 34644 | 128381 | 
| 54 | 26 examples/linear_model/plot_sparse_logistic_regression_mnist.py | 1 | 85| 678 | 35322 | 129083 | 
| 55 | 27 examples/linear_model/plot_sgdocsvm_vs_ocsvm.py | 78 | 155| 780 | 36102 | 130636 | 
| 56 | 28 examples/cross_decomposition/plot_compare_cross_decomposition.py | 1 | 94| 756 | 36858 | 132110 | 
| 57 | 28 sklearn/svm/_base.py | 1 | 30| 291 | 37149 | 132110 | 
| 58 | 28 sklearn/utils/validation.py | 447 | 545| 760 | 37909 | 132110 | 
| 59 | 29 examples/release_highlights/plot_release_highlights_1_1_0.py | 98 | 177| 847 | 38756 | 134247 | 
| 60 | 29 sklearn/preprocessing/_label.py | 314 | 368| 297 | 39053 | 134247 | 
| 61 | 30 sklearn/linear_model/_logistic.py | 784 | 1066| 3073 | 42126 | 153604 | 
| 62 | 31 examples/semi_supervised/plot_semi_supervised_newsgroups.py | 88 | 112| 313 | 42439 | 154515 | 
| 63 | 31 sklearn/utils/estimator_checks.py | 2554 | 3236| 5959 | 48398 | 154515 | 
| 64 | 32 examples/preprocessing/plot_scaling_importance.py | 1 | 74| 520 | 48918 | 156831 | 
| 65 | 32 examples/cross_decomposition/plot_compare_cross_decomposition.py | 95 | 172| 719 | 49637 | 156831 | 
| 66 | 33 sklearn/utils/sparsefuncs.py | 6 | 28| 186 | 49823 | 161713 | 
| 67 | 34 benchmarks/bench_saga.py | 136 | 234| 651 | 50474 | 164281 | 
| 68 | 35 examples/gaussian_process/plot_gpr_on_structured_data.py | 103 | 189| 671 | 51145 | 165771 | 
| 69 | 35 examples/preprocessing/plot_scaling_importance.py | 157 | 256| 911 | 52056 | 165771 | 
| 70 | 36 examples/impute/plot_iterative_imputer_variants_comparison.py | 1 | 91| 780 | 52836 | 167106 | 
| 71 | 36 sklearn/preprocessing/_label.py | 265 | 312| 358 | 53194 | 167106 | 
| 72 | 36 examples/release_highlights/plot_release_highlights_1_1_0.py | 1 | 96| 775 | 53969 | 167106 | 
| 73 | 36 sklearn/decomposition/_dict_learning.py | 231 | 367| 1186 | 55155 | 167106 | 
| 74 | 37 sklearn/datasets/_samples_generator.py | 435 | 457| 301 | 55456 | 182517 | 
| 75 | 38 examples/covariance/plot_sparse_cov.py | 1 | 93| 761 | 56217 | 183788 | 
| 76 | 38 sklearn/cluster/_affinity_propagation.py | 469 | 532| 475 | 56692 | 183788 | 
| 77 | 38 sklearn/decomposition/_dict_learning.py | 35 | 132| 756 | 57448 | 183788 | 
| 78 | 39 examples/linear_model/plot_lasso_and_elasticnet.py | 1 | 98| 653 | 58101 | 184441 | 


### Hint

```
Just checked: the fix seems to work for kernel='rbf', too.
Hi, I would like to take over since this is stalled.
Hi @cozek , sure go ahead: FYI you can comment "take" in this issue and it will automatically assigned to you.

```

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


## Code snippets

### 1 - sklearn/semi_supervised/_label_propagation.py:

Start line: 1, End line: 73

```python
# coding=utf8
"""
Label propagation in the context of this module refers to a set of
semi-supervised classification algorithms. At a high level, these algorithms
work by forming a fully-connected graph between all points given and solving
for the steady-state distribution of labels at each point.

These algorithms perform very well in practice. The cost of running can be very
expensive, at approximately O(N^3) where N is the number of (labeled and
unlabeled) points. The theory (why they perform so well) is motivated by
intuitions from random walk algorithms and geometric relationships in the data.
For more information see the references below.

Model Features
--------------
Label clamping:
  The algorithm tries to learn distributions of labels over the dataset given
  label assignments over an initial subset. In one variant, the algorithm does
  not allow for any errors in the initial assignment (hard-clamping) while
  in another variant, the algorithm allows for some wiggle room for the initial
  assignments, allowing them to change by a fraction alpha in each iteration
  (soft-clamping).

Kernel:
  A function which projects a vector into some higher dimensional space. This
  implementation supports RBF and KNN kernels. Using the RBF kernel generates
  a dense matrix of size O(N^2). KNN kernel will generate a sparse matrix of
  size O(k*N) which will run much faster. See the documentation for SVMs for
  more info on kernels.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets
>>> from sklearn.semi_supervised import LabelPropagation
>>> label_prop_model = LabelPropagation()
>>> iris = datasets.load_iris()
>>> rng = np.random.RandomState(42)
>>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
>>> labels = np.copy(iris.target)
>>> labels[random_unlabeled_points] = -1
>>> label_prop_model.fit(iris.data, labels)
LabelPropagation(...)

Notes
-----
References:
[1] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux. In Semi-Supervised
Learning (2006), pp. 193-216

[2] Olivier Delalleau, Yoshua Bengio, Nicolas Le Roux. Efficient
Non-Parametric Function Induction in Semi-Supervised Learning. AISTAT 2005
"""
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from ..base import BaseEstimator, ClassifierMixin
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ..utils._param_validation import Interval, StrOptions
from ..exceptions import ConvergenceWarning
```
### 2 - sklearn/semi_supervised/_label_propagation.py:

Start line: 607, End line: 622

```python
class LabelSpreading(BaseLabelPropagation):

    def _build_graph(self):
        """Graph matrix for Label Spreading computes the graph laplacian"""
        # compute affinity matrix (or gram matrix)
        if self.kernel == "knn":
            self.nn_fit = None
        n_samples = self.X_.shape[0]
        affinity_matrix = self._get_kernel(self.X_)
        laplacian = csgraph.laplacian(affinity_matrix, normed=True)
        laplacian = -laplacian
        if sparse.isspmatrix(laplacian):
            diag_mask = laplacian.row == laplacian.col
            laplacian.data[diag_mask] = 0.0
        else:
            laplacian.flat[:: n_samples + 1] = 0.0  # set diag to 0.0
        return laplacian
```
### 3 - sklearn/semi_supervised/_label_propagation.py:

Start line: 334, End line: 423

```python
class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.

    gamma : float, default=20
        Parameter for rbf kernel.

    n_neighbors : int, default=7
        Parameter for knn kernel which need to be strictly positive.

    max_iter : int, default=1000
        Change maximum number of iterations allowed.

    tol : float, 1e-3
        Convergence tolerance: threshold to consider the system at steady
        state.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Input array.

    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.

    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.

    transduction_ : ndarray of shape (n_samples)
        Label assigned to each item during :term:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    BaseLabelPropagation : Base class for label propagation module.
    LabelSpreading : Alternate label propagation strategy more robust to noise.

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelPropagation(...)
    """

    _variant = "propagation"

    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    _parameter_constraints.pop("alpha")
```
### 4 - sklearn/semi_supervised/_label_propagation.py:

Start line: 579, End line: 605

```python
class LabelSpreading(BaseLabelPropagation):

    _variant = "spreading"

    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    _parameter_constraints["alpha"] = [Interval(Real, 0, 1, closed="neither")]

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        alpha=0.2,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):

        # this one has different base parameters
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
        )
```
### 5 - sklearn/semi_supervised/_label_propagation.py:

Start line: 425, End line: 459

```python
class LabelPropagation(BaseLabelPropagation):

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        max_iter=1000,
        tol=1e-3,
        n_jobs=None,
    ):
        super().__init__(
            kernel=kernel,
            gamma=gamma,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
            alpha=None,
        )

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        if self.kernel == "knn":
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        if sparse.isspmatrix(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            affinity_matrix /= normalizer[:, np.newaxis]
        return affinity_matrix
```
### 6 - sklearn/semi_supervised/_label_propagation.py:

Start line: 483, End line: 577

```python
class LabelSpreading(BaseLabelPropagation):
    """LabelSpreading model for semi-supervised learning.

    This model is similar to the basic Label Propagation algorithm,
    but uses affinity matrix based on the normalized graph Laplacian
    and soft clamping across the labels.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.

    gamma : float, default=20
      Parameter for rbf kernel.

    n_neighbors : int, default=7
      Parameter for knn kernel which is a strictly positive integer.

    alpha : float, default=0.2
      Clamping factor. A value in (0, 1) that specifies the relative amount
      that an instance should adopt the information from its neighbors as
      opposed to its initial label.
      alpha=0 means keeping the initial label information; alpha=1 means
      replacing all initial information.

    max_iter : int, default=30
      Maximum number of iterations allowed.

    tol : float, default=1e-3
      Convergence tolerance: threshold to consider the system at steady
      state.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Input array.

    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.

    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.

    transduction_ : ndarray of shape (n_samples,)
        Label assigned to each item during :term:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    LabelPropagation : Unregularized graph based semi-supervised learning.

    References
    ----------
    `Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston,
    Bernhard Schoelkopf. Learning with local and global consistency (2004)
    <https://citeseerx.ist.psu.edu/doc_view/pid/d74c37aabf2d5cae663007cbd8718175466aea8c>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelSpreading
    >>> label_prop_model = LabelSpreading()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelSpreading(...)
    """
```
### 7 - examples/semi_supervised/plot_label_propagation_structure.py:

Start line: 1, End line: 106

```python
"""
==============================================
Label Propagation learning a complex structure
==============================================

Example of LabelPropagation learning a complex internal structure
to demonstrate "manifold learning". The outer circle should be
labeled "red" and the inner circle "blue". Because both label groups
lie inside their own distinct shape, we can see that the labels
propagate correctly around the circle.

"""

import numpy as np
from sklearn.datasets import make_circles

n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)
outer, inner = 0, 1
labels = np.full(n_samples, -1.0)
labels[0] = outer
labels[-1] = inner

# %%
# Plot raw data
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
plt.scatter(
    X[labels == outer, 0],
    X[labels == outer, 1],
    color="navy",
    marker="s",
    lw=0,
    label="outer labeled",
    s=10,
)
plt.scatter(
    X[labels == inner, 0],
    X[labels == inner, 1],
    color="c",
    marker="s",
    lw=0,
    label="inner labeled",
    s=10,
)
plt.scatter(
    X[labels == -1, 0],
    X[labels == -1, 1],
    color="darkorange",
    marker=".",
    label="unlabeled",
)
plt.legend(scatterpoints=1, shadow=False, loc="center")
_ = plt.title("Raw data (2 classes=outer and inner)")

# %%
#
# The aim of :class:`~sklearn.semi_supervised.LabelSpreading` is to associate
# a label to sample where the label is initially unknown.
from sklearn.semi_supervised import LabelSpreading

label_spread = LabelSpreading(kernel="knn", alpha=0.8)
label_spread.fit(X, labels)

# %%
# Now, we can check which labels have been associated with each sample
# when the label was unknown.
output_labels = label_spread.transduction_
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]

plt.figure(figsize=(4, 4))
plt.scatter(
    X[outer_numbers, 0],
    X[outer_numbers, 1],
    color="navy",
    marker="s",
    lw=0,
    s=10,
    label="outer learned",
)
plt.scatter(
    X[inner_numbers, 0],
    X[inner_numbers, 1],
    color="c",
    marker="s",
    lw=0,
    s=10,
    label="inner learned",
)
plt.legend(scatterpoints=1, shadow=False, loc="center")
plt.title("Labels learned with Label Spreading (KNN)")
plt.show()
```
### 8 - sklearn/semi_supervised/_label_propagation.py:

Start line: 234, End line: 331

```python
class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model to X.

        The input samples (labeled and unlabeled) are provided by matrix X,
        and target labels are provided by matrix y. We conventionally apply the
        label -1 to unlabeled samples in matrix y in a semi-supervised
        classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target class values with unlabeled points marked as -1.
            All unlabeled samples will be transductively assigned labels
            internally, which are stored in `transduction_`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()
        X, y = self._validate_data(X, y)
        self.X_ = X
        check_classification_targets(y)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique(y)
        classes = classes[classes != -1]
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        y = np.asarray(y)
        unlabeled = y == -1

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1

        y_static = np.copy(self.label_distributions_)
        if self._variant == "propagation":
            # LabelPropagation
            y_static[unlabeled] = 0
        else:
            # LabelSpreading
            y_static *= 1 - self.alpha

        l_previous = np.zeros((self.X_.shape[0], n_classes))

        unlabeled = unlabeled[:, np.newaxis]
        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_
            self.label_distributions_ = safe_sparse_dot(
                graph_matrix, self.label_distributions_
            )

            if self._variant == "propagation":
                normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
                normalizer[normalizer == 0] = 1
                self.label_distributions_ /= normalizer
                self.label_distributions_ = np.where(
                    unlabeled, self.label_distributions_, y_static
                )
            else:
                # clamp
                self.label_distributions_ = (
                    np.multiply(self.alpha, self.label_distributions_) + y_static
                )
        else:
            warnings.warn(
                "max_iter=%d was reached without convergence." % self.max_iter,
                category=ConvergenceWarning,
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        self.transduction_ = transduction.ravel()
        return self
```
### 9 - sklearn/semi_supervised/_label_propagation.py:

Start line: 145, End line: 166

```python
class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(
                    n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
                ).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(
                    self.nn_fit._fit_X, self.n_neighbors, mode="connectivity"
                )
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)
```
### 10 - sklearn/semi_supervised/_label_propagation.py:

Start line: 76, End line: 143

```python
class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for label propagation module.

     Parameters
     ----------
     kernel : {'knn', 'rbf'} or callable, default='rbf'
         String identifier for kernel function to use or the kernel function
         itself. Only 'rbf' and 'knn' strings are valid inputs. The function
         passed should take two inputs, each of shape (n_samples, n_features),
         and return a (n_samples, n_samples) shaped weight matrix.

     gamma : float, default=20
         Parameter for rbf kernel.

     n_neighbors : int, default=7
         Parameter for knn kernel. Need to be strictly positive.

     alpha : float, default=1.0
         Clamping factor.

     max_iter : int, default=30
         Change maximum number of iterations allowed.

     tol : float, default=1e-3
         Convergence tolerance: threshold to consider the system at steady
         state.

    n_jobs : int, default=None
         The number of parallel jobs to run.
         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
         for more details.
    """

    _parameter_constraints: dict = {
        "kernel": [StrOptions({"knn", "rbf"}), callable],
        "gamma": [Interval(Real, 0, None, closed="left")],
        "n_neighbors": [Interval(Integral, 0, None, closed="neither")],
        "alpha": [None, Interval(Real, 0, 1, closed="neither")],
        "max_iter": [Interval(Integral, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        alpha=1,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs
```
### 11 - sklearn/semi_supervised/_label_propagation.py:

Start line: 461, End line: 480

```python
class LabelPropagation(BaseLabelPropagation):

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target class values with unlabeled points marked as -1.
            All unlabeled samples will be transductively assigned labels
            internally, which are stored in `transduction_`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y)
```
### 12 - sklearn/semi_supervised/_label_propagation.py:

Start line: 168, End line: 192

```python
class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError(
            "Graph construction must be implemented to fit a label propagation model."
        )

    def predict(self, X):
        """Perform inductive inference across the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predictions for input data.
        """
        # Note: since `predict` does not accept semi-supervised labels as input,
        # `fit(X, y).predict(X) != fit(X, y).transduction_`.
        # Hence, `fit_predict` is not implemented.
        # See https://github.com/scikit-learn/scikit-learn/pull/24898
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()
```

# scikit-learn__scikit-learn-10452

| **scikit-learn/scikit-learn** | `3e5469eda719956c076ae8e685ec1183bfd98569` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 1107 |
| **Any found context length** | 235 |
| **Avg pos** | 4.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 2 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/preprocessing/data.py b/sklearn/preprocessing/data.py
--- a/sklearn/preprocessing/data.py
+++ b/sklearn/preprocessing/data.py
@@ -1325,7 +1325,7 @@ def fit(self, X, y=None):
         -------
         self : instance
         """
-        n_samples, n_features = check_array(X).shape
+        n_samples, n_features = check_array(X, accept_sparse=True).shape
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
@@ -1338,31 +1338,42 @@ def transform(self, X):
 
         Parameters
         ----------
-        X : array-like, shape [n_samples, n_features]
+        X : array-like or sparse matrix, shape [n_samples, n_features]
             The data to transform, row by row.
+            Sparse input should preferably be in CSC format.
 
         Returns
         -------
-        XP : np.ndarray shape [n_samples, NP]
+        XP : np.ndarray or CSC sparse matrix, shape [n_samples, NP]
             The matrix of features, where NP is the number of polynomial
             features generated from the combination of inputs.
         """
         check_is_fitted(self, ['n_input_features_', 'n_output_features_'])
 
-        X = check_array(X, dtype=FLOAT_DTYPES)
+        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse='csc')
         n_samples, n_features = X.shape
 
         if n_features != self.n_input_features_:
             raise ValueError("X shape does not match training shape")
 
-        # allocate output data
-        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
-
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
-        for i, c in enumerate(combinations):
-            XP[:, i] = X[:, c].prod(1)
+        if sparse.isspmatrix(X):
+            columns = []
+            for comb in combinations:
+                if comb:
+                    out_col = 1
+                    for col_idx in comb:
+                        out_col = X[:, col_idx].multiply(out_col)
+                    columns.append(out_col)
+                else:
+                    columns.append(sparse.csc_matrix(np.ones((X.shape[0], 1))))
+            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
+        else:
+            XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
+            for i, comb in enumerate(combinations):
+                XP[:, i] = X[:, comb].prod(1)
 
         return XP
 

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/preprocessing/data.py | 1328 | 1328 | 3 | 1 | 1107
| sklearn/preprocessing/data.py | 1341 | 1365 | 1 | 1 | 235


## Problem Statement

```
Polynomial Features for sparse data
I'm not sure if that came up before but PolynomialFeatures doesn't support sparse data, which is not great. Should be easy but I haven't checked ;)

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/preprocessing/data.py** | 1340 | 1371| 235 | 235 | 25777 | 
| 2 | **1 sklearn/preprocessing/data.py** | 1203 | 1276| 743 | 978 | 25777 | 
| **-> 3 <-** | **1 sklearn/preprocessing/data.py** | 1318 | 1338| 129 | 1107 | 25777 | 
| 4 | 2 examples/linear_model/plot_ols_3d.py | 1 | 37| 214 | 1321 | 26332 | 
| 5 | 3 examples/linear_model/plot_polynomial_interpolation.py | 1 | 73| 512 | 1833 | 26867 | 
| 6 | **3 sklearn/preprocessing/data.py** | 1278 | 1316| 294 | 2127 | 26867 | 
| 7 | 4 examples/model_selection/plot_underfitting_overfitting.py | 1 | 72| 631 | 2758 | 27498 | 
| 8 | 5 examples/linear_model/plot_lasso_dense_vs_sparse_data.py | 1 | 67| 511 | 3269 | 28009 | 
| 9 | 6 sklearn/linear_model/randomized_l1.py | 1 | 29| 171 | 3440 | 33780 | 
| 10 | 7 benchmarks/bench_sparsify.py | 1 | 82| 754 | 4194 | 34687 | 
| 11 | 8 sklearn/utils/estimator_checks.py | 406 | 447| 427 | 4621 | 52604 | 
| 12 | 9 sklearn/decomposition/dict_learning.py | 107 | 167| 632 | 5253 | 63288 | 
| 13 | 9 sklearn/utils/estimator_checks.py | 1742 | 1762| 233 | 5486 | 63288 | 
| 14 | 10 examples/linear_model/plot_ard.py | 102 | 117| 155 | 5641 | 64211 | 
| 15 | 10 benchmarks/bench_sparsify.py | 83 | 106| 153 | 5794 | 64211 | 
| 16 | 11 sklearn/feature_extraction/text.py | 776 | 836| 467 | 6261 | 75724 | 
| 17 | 12 examples/covariance/plot_sparse_cov.py | 1 | 89| 754 | 7015 | 76923 | 
| 18 | 12 sklearn/decomposition/dict_learning.py | 255 | 309| 479 | 7494 | 76923 | 
| 19 | 13 examples/linear_model/plot_omp.py | 1 | 83| 557 | 8051 | 77480 | 
| 20 | 14 examples/feature_selection/plot_feature_selection.py | 1 | 87| 616 | 8667 | 78096 | 
| 21 | 14 sklearn/utils/estimator_checks.py | 727 | 758| 297 | 8964 | 78096 | 
| 22 | 15 sklearn/decomposition/sparse_pca.py | 1 | 99| 683 | 9647 | 80291 | 
| 23 | 16 sklearn/metrics/pairwise.py | 1254 | 1280| 192 | 9839 | 91823 | 
| 24 | 17 examples/linear_model/plot_sparse_logistic_regression_20newsgroups.py | 1 | 59| 462 | 10301 | 92811 | 
| 25 | 18 examples/linear_model/plot_logistic_l1_l2_sparsity.py | 1 | 80| 698 | 10999 | 93563 | 
| 26 | 19 examples/decomposition/plot_sparse_coding.py | 1 | 22| 172 | 11171 | 94590 | 
| 27 | 20 examples/linear_model/plot_multi_task_lasso_support.py | 1 | 70| 587 | 11758 | 95202 | 
| 28 | 21 examples/linear_model/plot_robust_fit.py | 1 | 73| 509 | 12267 | 95996 | 
| 29 | 21 sklearn/metrics/pairwise.py | 1310 | 1320| 107 | 12374 | 95996 | 
| 30 | 21 sklearn/decomposition/dict_learning.py | 170 | 254| 771 | 13145 | 95996 | 
| 31 | 22 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 94 | 109| 187 | 13332 | 96985 | 
| 32 | 23 benchmarks/bench_lasso.py | 60 | 97| 382 | 13714 | 97780 | 
| 33 | 23 sklearn/decomposition/dict_learning.py | 791 | 841| 365 | 14079 | 97780 | 
| 34 | 23 sklearn/decomposition/sparse_pca.py | 140 | 186| 390 | 14469 | 97780 | 
| 35 | 24 benchmarks/bench_plot_incremental_pca.py | 48 | 58| 147 | 14616 | 99238 | 
| 36 | 24 benchmarks/bench_plot_incremental_pca.py | 86 | 106| 229 | 14845 | 99238 | 
| 37 | 25 benchmarks/bench_random_projections.py | 70 | 86| 174 | 15019 | 100998 | 
| 38 | 26 sklearn/linear_model/coordinate_descent.py | 771 | 793| 170 | 15189 | 121394 | 
| 39 | 27 examples/svm/plot_svm_kernels.py | 1 | 86| 631 | 15820 | 122033 | 
| 40 | 27 sklearn/utils/estimator_checks.py | 347 | 403| 300 | 16120 | 122033 | 
| 41 | 28 examples/applications/plot_model_complexity_influence.py | 45 | 66| 213 | 16333 | 123557 | 
| 42 | 29 sklearn/linear_model/base.py | 375 | 402| 257 | 16590 | 128525 | 
| 43 | 30 examples/preprocessing/plot_scaling_importance.py | 1 | 82| 762 | 17352 | 129731 | 
| 44 | 30 benchmarks/bench_plot_incremental_pca.py | 35 | 45| 146 | 17498 | 129731 | 
| 45 | 31 examples/plot_johnson_lindenstrauss_bound.py | 162 | 200| 376 | 17874 | 131511 | 
| 46 | 32 examples/text/plot_document_classification_20newsgroups.py | 1 | 121| 783 | 18657 | 134068 | 
| 47 | 32 sklearn/feature_extraction/text.py | 735 | 774| 376 | 19033 | 134068 | 
| 48 | 32 examples/plot_johnson_lindenstrauss_bound.py | 94 | 160| 646 | 19679 | 134068 | 
| 49 | 32 sklearn/metrics/pairwise.py | 745 | 778| 236 | 19915 | 134068 | 
| 50 | 32 benchmarks/bench_plot_incremental_pca.py | 109 | 157| 485 | 20400 | 134068 | 
| 51 | 32 sklearn/decomposition/dict_learning.py | 844 | 938| 734 | 21134 | 134068 | 
| 52 | 33 benchmarks/bench_plot_randomized_svd.py | 111 | 128| 156 | 21290 | 138452 | 
| 53 | 33 sklearn/decomposition/dict_learning.py | 27 | 105| 677 | 21967 | 138452 | 
| 54 | 34 examples/linear_model/plot_bayesian_ridge.py | 100 | 115| 149 | 22116 | 139372 | 
| 55 | 35 sklearn/ensemble/partial_dependence.py | 242 | 322| 767 | 22883 | 143121 | 
| 56 | 36 sklearn/feature_extraction/hashing.py | 102 | 133| 262 | 23145 | 144658 | 
| 57 | 37 examples/classification/plot_lda.py | 41 | 72| 325 | 23470 | 145249 | 
| 58 | 38 sklearn/random_projection.py | 264 | 292| 292 | 23762 | 150222 | 
| 59 | 39 examples/text/plot_hashing_vs_dict_vectorizer.py | 1 | 112| 802 | 24564 | 151041 | 
| 60 | 40 examples/ensemble/plot_feature_transformation.py | 1 | 91| 765 | 25329 | 152164 | 
| 61 | 40 examples/decomposition/plot_sparse_coding.py | 34 | 101| 741 | 26070 | 152164 | 
| 62 | 40 sklearn/decomposition/sparse_pca.py | 101 | 138| 277 | 26347 | 152164 | 
| 63 | 41 examples/plot_kernel_approximation.py | 88 | 170| 777 | 27124 | 154105 | 
| 64 | 42 examples/svm/plot_svm_regression.py | 1 | 45| 356 | 27480 | 154461 | 
| 65 | 42 sklearn/linear_model/randomized_l1.py | 32 | 60| 260 | 27740 | 154461 | 
| 66 | 42 sklearn/metrics/pairwise.py | 1 | 30| 131 | 27871 | 154461 | 
| 67 | 42 examples/plot_kernel_approximation.py | 1 | 87| 735 | 28606 | 154461 | 
| 68 | **42 sklearn/preprocessing/data.py** | 1752 | 1808| 552 | 29158 | 154461 | 
| 69 | 43 examples/preprocessing/plot_all_scaling.py | 219 | 320| 895 | 30053 | 157563 | 
| 70 | 43 examples/applications/plot_model_complexity_influence.py | 121 | 170| 540 | 30593 | 157563 | 
| 71 | 43 sklearn/ensemble/partial_dependence.py | 323 | 343| 277 | 30870 | 157563 | 
| 72 | 44 examples/svm/plot_svm_anova.py | 1 | 59| 426 | 31296 | 157989 | 
| 73 | 44 examples/covariance/plot_sparse_cov.py | 91 | 136| 411 | 31707 | 157989 | 
| 74 | 45 sklearn/feature_extraction/dict_vectorizer.py | 274 | 318| 314 | 32021 | 160717 | 
| 75 | 45 benchmarks/bench_lasso.py | 21 | 57| 300 | 32321 | 160717 | 
| 76 | 45 benchmarks/bench_plot_randomized_svd.py | 1 | 110| 951 | 33272 | 160717 | 
| 77 | 45 sklearn/metrics/pairwise.py | 1394 | 1410| 187 | 33459 | 160717 | 
| 78 | 45 sklearn/ensemble/partial_dependence.py | 344 | 396| 669 | 34128 | 160717 | 
| 79 | 46 examples/linear_model/plot_lasso_and_elasticnet.py | 1 | 70| 535 | 34663 | 161252 | 
| 80 | 47 examples/cross_decomposition/plot_compare_cross_decomposition.py | 83 | 156| 759 | 35422 | 162735 | 
| 81 | 48 examples/applications/plot_out_of_core_classification.py | 188 | 215| 234 | 35656 | 166044 | 
| 82 | 49 sklearn/utils/fixes.py | 153 | 257| 776 | 36432 | 168440 | 
| 83 | 50 examples/hetero_feature_union.py | 1 | 43| 311 | 36743 | 169769 | 
| 84 | 51 sklearn/svm/base.py | 412 | 432| 209 | 36952 | 177605 | 
| 85 | 51 sklearn/utils/estimator_checks.py | 1 | 72| 625 | 37577 | 177605 | 
| 86 | 51 examples/cluster/plot_feature_agglomeration_vs_univariate_selection.py | 1 | 93| 777 | 38354 | 177605 | 
| 87 | 52 sklearn/preprocessing/imputation.py | 169 | 247| 635 | 38989 | 180585 | 
| 88 | 53 benchmarks/bench_plot_nmf.py | 371 | 423| 514 | 39503 | 184498 | 
| 89 | 54 sklearn/datasets/samples_generator.py | 1057 | 1117| 462 | 39965 | 198320 | 
| 91 | 55 sklearn/svm/base.py | 434 | 458| 273 | 41096 | 199486 | 


### Hint

```
I'll take this up.
@dalmia @amueller any news on this feature? We're eagerly anticipating it ;-)

See also #3512, #3514
For the benefit of @amueller, my [comment](https://github.com/scikit-learn/scikit-learn/pull/8380#issuecomment-299164291) on #8380:

>  [@jnothman] closed both #3512 and #3514 at the time, in favor of #4286, but sparsity is still not conserved. Is there some way to get attention back to the issue of sparse polynomial features?
@jnothman Thanks! :smiley: 
This should be pretty straightforward to solve. I've given a couple of options at https://github.com/scikit-learn/scikit-learn/pull/8380#issuecomment-299120531.

Would a contributor like to take it up? @SebastinSanty, perhaps?
@jnothman Yes, I would like to take it up.
It will be very useful for me, @SebastinSanty when will it be available, even your develop branch:)
Does "Polynomial Features for sparse data" implemented? 
I also asked a question about this in [Stackoverflow](https://stackoverflow.com/questions/48199391/how-to-make-polynomial-features-using-sparse-matrix-in-scikit-learn)
I've had enough of people asking for this and it not getting closed, despite almost all the work having been done. I'll open a PR soon.
```

## Patch

```diff
diff --git a/sklearn/preprocessing/data.py b/sklearn/preprocessing/data.py
--- a/sklearn/preprocessing/data.py
+++ b/sklearn/preprocessing/data.py
@@ -1325,7 +1325,7 @@ def fit(self, X, y=None):
         -------
         self : instance
         """
-        n_samples, n_features = check_array(X).shape
+        n_samples, n_features = check_array(X, accept_sparse=True).shape
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
@@ -1338,31 +1338,42 @@ def transform(self, X):
 
         Parameters
         ----------
-        X : array-like, shape [n_samples, n_features]
+        X : array-like or sparse matrix, shape [n_samples, n_features]
             The data to transform, row by row.
+            Sparse input should preferably be in CSC format.
 
         Returns
         -------
-        XP : np.ndarray shape [n_samples, NP]
+        XP : np.ndarray or CSC sparse matrix, shape [n_samples, NP]
             The matrix of features, where NP is the number of polynomial
             features generated from the combination of inputs.
         """
         check_is_fitted(self, ['n_input_features_', 'n_output_features_'])
 
-        X = check_array(X, dtype=FLOAT_DTYPES)
+        X = check_array(X, dtype=FLOAT_DTYPES, accept_sparse='csc')
         n_samples, n_features = X.shape
 
         if n_features != self.n_input_features_:
             raise ValueError("X shape does not match training shape")
 
-        # allocate output data
-        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
-
         combinations = self._combinations(n_features, self.degree,
                                           self.interaction_only,
                                           self.include_bias)
-        for i, c in enumerate(combinations):
-            XP[:, i] = X[:, c].prod(1)
+        if sparse.isspmatrix(X):
+            columns = []
+            for comb in combinations:
+                if comb:
+                    out_col = 1
+                    for col_idx in comb:
+                        out_col = X[:, col_idx].multiply(out_col)
+                    columns.append(out_col)
+                else:
+                    columns.append(sparse.csc_matrix(np.ones((X.shape[0], 1))))
+            XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
+        else:
+            XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
+            for i, comb in enumerate(combinations):
+                XP[:, i] = X[:, comb].prod(1)
 
         return XP
 

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_data.py b/sklearn/preprocessing/tests/test_data.py
--- a/sklearn/preprocessing/tests/test_data.py
+++ b/sklearn/preprocessing/tests/test_data.py
@@ -7,10 +7,12 @@
 
 import warnings
 import re
+
 import numpy as np
 import numpy.linalg as la
 from scipy import sparse, stats
 from distutils.version import LooseVersion
+import pytest
 
 from sklearn.utils import gen_batches
 
@@ -155,6 +157,28 @@ def test_polynomial_feature_names():
                        feature_names)
 
 
+@pytest.mark.parametrize(['deg', 'include_bias', 'interaction_only', 'dtype'],
+                         [(1, True, False, int),
+                          (2, True, False, int),
+                          (2, True, False, np.float32),
+                          (2, True, False, np.float64),
+                          (3, False, False, np.float64),
+                          (3, False, True, np.float64)])
+def test_polynomial_features_sparse_X(deg, include_bias, interaction_only,
+                                      dtype):
+    rng = np.random.RandomState(0)
+    X = rng.randint(0, 2, (100, 2))
+    X_sparse = sparse.csr_matrix(X)
+
+    est = PolynomialFeatures(deg, include_bias=include_bias)
+    Xt_sparse = est.fit_transform(X_sparse.astype(dtype))
+    Xt_dense = est.fit_transform(X.astype(dtype))
+
+    assert isinstance(Xt_sparse, sparse.csc_matrix)
+    assert Xt_sparse.dtype == Xt_dense.dtype
+    assert_array_almost_equal(Xt_sparse.A, Xt_dense)
+
+
 def test_standard_scaler_1d():
     # Test scaling of dataset along single axis
     for X in [X_1row, X_1col, X_list_1row, X_list_1row]:

```


## Code snippets

### 1 - sklearn/preprocessing/data.py:

Start line: 1340, End line: 1371

```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Transform data to polynomial features

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        XP : np.ndarray shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = check_array(X, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)

        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        for i, c in enumerate(combinations):
            XP[:, i] = X[:, c].prod(1)

        return XP
```
### 2 - sklearn/preprocessing/data.py:

Start line: 1203, End line: 1276

```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree : integer
        The degree of the polynomial features. Default = 2.

    interaction_only : boolean, default = False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

    include_bias : boolean
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    Examples
    --------
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[  1.,   0.,   1.,   0.,   0.,   1.],
           [  1.,   2.,   3.,   4.,   6.,   9.],
           [  1.,   4.,   5.,  16.,  20.,  25.]])
    >>> poly = PolynomialFeatures(interaction_only=True)
    >>> poly.fit_transform(X)
    array([[  1.,   0.,   1.,   0.],
           [  1.,   2.,   3.,   6.],
           [  1.,   4.,   5.,  20.]])

    Attributes
    ----------
    powers_ : array, shape (n_output_features, n_input_features)
        powers_[i, j] is the exponent of the jth input in the ith output.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.

    See :ref:`examples/linear_model/plot_polynomial_interpolation.py
    <sphx_glr_auto_examples_linear_model_plot_polynomial_interpolation.py>`
    """
    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    @staticmethod
    def _combinations(n_features, degree, interaction_only, include_bias):
        comb = (combinations if interaction_only else combinations_w_r)
        start = int(not include_bias)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(start, degree + 1))
```
### 3 - sklearn/preprocessing/data.py:

Start line: 1318, End line: 1338

```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """
        Compute number of output features.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(X).shape
        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self
```
### 4 - examples/linear_model/plot_ols_3d.py:

Start line: 1, End line: 37

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Sparsity Example: Fitting only features 1  and 2
=========================================================

Features 1 and 2 of the diabetes-dataset are fitted and
plotted below. It illustrates that although feature 2
has a strong coefficient on the full model, it does not
give us much regarding `y` when compared to just feature 1

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()
indices = (0, 1)

X_train = diabetes.data[:-20, indices]
X_test = diabetes.data[-20:, indices]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
```
### 5 - examples/linear_model/plot_polynomial_interpolation.py:

Start line: 1, End line: 73

```python
#!/usr/bin/env python
"""
========================
Polynomial interpolation
========================

This example demonstrates how to approximate a function with a polynomial of
degree n_degree by using ridge regression. Concretely, from n_samples 1d
points, it suffices to build the Vandermonde matrix, which is n_samples x
n_degree+1 and has the following form:

[[1, x_1, x_1 ** 2, x_1 ** 3, ...],
 [1, x_2, x_2 ** 2, x_2 ** 3, ...],
 ...]

Intuitively, this matrix can be interpreted as a matrix of pseudo features (the
points raised to some power). The matrix is akin to (but different from) the
matrix induced by a polynomial kernel.

This example shows that you can do non-linear regression with a linear model,
using a pipeline to add non-linear features. Kernel methods extend this idea
and can induce very high (even infinite) dimensional feature spaces.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 2
plt.plot(x_plot, f(x_plot), color='cornflowerblue', linewidth=lw,
         label="ground truth")
plt.scatter(x, y, color='navy', s=30, marker='o', label="training points")

for count, degree in enumerate([3, 4, 5]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()
```
### 6 - sklearn/preprocessing/data.py:

Start line: 1278, End line: 1316

```python
class PolynomialFeatures(BaseEstimator, TransformerMixin):

    @property
    def powers_(self):
        check_is_fitted(self, 'n_input_features_')

        combinations = self._combinations(self.n_input_features_, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        return np.vstack(np.bincount(c, minlength=self.n_input_features_)
                         for c in combinations)

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features

        """
        powers = self.powers_
        if input_features is None:
            input_features = ['x%d' % i for i in range(powers.shape[1])]
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join("%s^%d" % (input_features[ind], exp)
                                if exp != 1 else input_features[ind]
                                for ind, exp in zip(inds, row[inds]))
            else:
                name = "1"
            feature_names.append(name)
        return feature_names
```
### 7 - examples/model_selection/plot_underfitting_overfitting.py:

Start line: 1, End line: 72

```python
"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
```
### 8 - examples/linear_model/plot_lasso_dense_vs_sparse_data.py:

Start line: 1, End line: 67

```python
"""
==============================
Lasso on dense and sparse data
==============================

We show that linear_model.Lasso provides the same results for dense and sparse
data and that in the case of sparse data the speed is improved.

"""
print(__doc__)

from time import time
from scipy import sparse
from scipy import linalg

from sklearn.datasets.samples_generator import make_regression
from sklearn.linear_model import Lasso


# #############################################################################
# The two Lasso implementations on Dense data
print("--- Dense matrices")

X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
X_sp = sparse.coo_matrix(X)

alpha = 1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

t0 = time()
sparse_lasso.fit(X_sp, y)
print("Sparse Lasso done in %fs" % (time() - t0))

t0 = time()
dense_lasso.fit(X, y)
print("Dense Lasso done in %fs" % (time() - t0))

print("Distance between coefficients : %s"
      % linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_))

# #############################################################################
# The two Lasso implementations on Sparse data
print("--- Sparse matrices")

Xs = X.copy()
Xs[Xs < 2.5] = 0.0
Xs = sparse.coo_matrix(Xs)
Xs = Xs.tocsc()

print("Matrix density : %s %%" % (Xs.nnz / float(X.size) * 100))

alpha = 0.1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

t0 = time()
sparse_lasso.fit(Xs, y)
print("Sparse Lasso done in %fs" % (time() - t0))

t0 = time()
dense_lasso.fit(Xs.toarray(), y)
print("Dense Lasso done in %fs" % (time() - t0))

print("Distance between coefficients : %s"
      % linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_))
```
### 9 - sklearn/linear_model/randomized_l1.py:

Start line: 1, End line: 29

```python
"""
Randomized Lasso/Logistic: feature selection based on Lasso and
sparse Logistic Regression
"""

import warnings
import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import issparse
from scipy import sparse
from scipy.interpolate import interp1d

from .base import _preprocess_data
from ..base import BaseEstimator
from ..externals import six
from ..externals.joblib import Memory, Parallel, delayed
from ..feature_selection.base import SelectorMixin
from ..utils import (as_float_array, check_random_state, check_X_y, safe_mask,
                     deprecated)
from ..utils.validation import check_is_fitted
from .least_angle import lars_path, LassoLarsIC
from .logistic import LogisticRegression
from ..exceptions import ConvergenceWarning
```
### 10 - benchmarks/bench_sparsify.py:

Start line: 1, End line: 82

```python
"""
Benchmark SGD prediction time with dense/sparse coefficients.

Invoke with
-----------

$ kernprof.py -l sparsity_benchmark.py
$ python -m line_profiler sparsity_benchmark.py.lprof

Typical output
--------------

input data sparsity: 0.050000
true coef sparsity: 0.000100
test data sparsity: 0.027400
model sparsity: 0.000024
r^2 on test data (dense model) : 0.233651
r^2 on test data (sparse model) : 0.233651
Wrote profile results to sparsity_benchmark.py.lprof
Timer unit: 1e-06 s

File: sparsity_benchmark.py
Function: benchmark_dense_predict at line 51
Total time: 0.532979 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    51                                           @profile
    52                                           def benchmark_dense_predict():
    53       301          640      2.1      0.1      for _ in range(300):
    54       300       532339   1774.5     99.9          clf.predict(X_test)

File: sparsity_benchmark.py
Function: benchmark_sparse_predict at line 56
Total time: 0.39274 s

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    56                                           @profile
    57                                           def benchmark_sparse_predict():
    58         1        10854  10854.0      2.8      X_test_sparse = csr_matrix(X_test)
    59       301          477      1.6      0.1      for _ in range(300):
    60       300       381409   1271.4     97.1          clf.predict(X_test_sparse)
"""

from scipy.sparse.csr import csr_matrix
import numpy as np
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.metrics import r2_score

np.random.seed(42)


def sparsity_ratio(X):
    return np.count_nonzero(X) / float(n_samples * n_features)

n_samples, n_features = 5000, 300
X = np.random.randn(n_samples, n_features)
inds = np.arange(n_samples)
np.random.shuffle(inds)
X[inds[int(n_features / 1.2):]] = 0  # sparsify input
print("input data sparsity: %f" % sparsity_ratio(X))
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[n_features // 2:]] = 0  # sparsify coef
print("true coef sparsity: %f" % sparsity_ratio(coef))
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal((n_samples,))

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]
print("test data sparsity: %f" % sparsity_ratio(X_test))

###############################################################################
clf = SGDRegressor(penalty='l1', alpha=.2, fit_intercept=True, max_iter=2000,
                   tol=None)
clf.fit(X_train, y_train)
```
### 68 - sklearn/preprocessing/data.py:

Start line: 1752, End line: 1808

```python
def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------

    X : {array, sparse matrix}, shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.

    Examples
    --------

    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[ 1.,  0.,  1.],
           [ 1.,  1.,  0.]])
    """
    X = check_array(X, accept_sparse=['csc', 'csr', 'coo'], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sparse.issparse(X):
        if sparse.isspmatrix_coo(X):
            # Shift columns to the right.
            col = X.col + 1
            # Column indices of dummy feature are 0 everywhere.
            col = np.concatenate((np.zeros(n_samples), col))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            row = np.concatenate((np.arange(n_samples), X.row))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sparse.coo_matrix((data, (row, col)), shape)
        elif sparse.isspmatrix_csc(X):
            # Shift index pointers since we need to add n_samples elements.
            indptr = X.indptr + n_samples
            # indptr[0] must be 0.
            indptr = np.concatenate((np.array([0]), indptr))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.ones(n_samples) * value, X.data))
            return sparse.csc_matrix((data, indices, indptr), shape)
        else:
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        return np.hstack((np.ones((n_samples, 1)) * value, X))
```

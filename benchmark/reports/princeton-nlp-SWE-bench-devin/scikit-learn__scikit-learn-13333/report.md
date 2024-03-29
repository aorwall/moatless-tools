# scikit-learn__scikit-learn-13333

| **scikit-learn/scikit-learn** | `04a5733b86bba57a48520b97b9c0a5cd325a1b9a` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | - |
| **Any found context length** | 831 |
| **Avg pos** | 7.0 |
| **Min pos** | 1 |
| **Max pos** | 3 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/preprocessing/data.py b/sklearn/preprocessing/data.py
--- a/sklearn/preprocessing/data.py
+++ b/sklearn/preprocessing/data.py
@@ -424,7 +424,7 @@ def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
         X_scaled = X_std * (max - min) + min
 
     where min, max = feature_range.
- 
+
     The transformation is calculated as (when ``axis=0``)::
 
        X_scaled = scale * X + min - X.min(axis=0) * scale
@@ -592,7 +592,7 @@ class StandardScaler(BaseEstimator, TransformerMixin):
     -----
     NaNs are treated as missing values: disregarded in fit, and maintained in
     transform.
-    
+
     We use a biased estimator for the standard deviation, equivalent to
     `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
     affect model performance.
@@ -2041,9 +2041,13 @@ class QuantileTransformer(BaseEstimator, TransformerMixin):
 
     Parameters
     ----------
-    n_quantiles : int, optional (default=1000)
+    n_quantiles : int, optional (default=1000 or n_samples)
         Number of quantiles to be computed. It corresponds to the number
         of landmarks used to discretize the cumulative distribution function.
+        If n_quantiles is larger than the number of samples, n_quantiles is set
+        to the number of samples as a larger number of quantiles does not give
+        a better approximation of the cumulative distribution function
+        estimator.
 
     output_distribution : str, optional (default='uniform')
         Marginal distribution for the transformed data. The choices are
@@ -2072,6 +2076,10 @@ class QuantileTransformer(BaseEstimator, TransformerMixin):
 
     Attributes
     ----------
+    n_quantiles_ : integer
+        The actual number of quantiles used to discretize the cumulative
+        distribution function.
+
     quantiles_ : ndarray, shape (n_quantiles, n_features)
         The values corresponding the quantiles of reference.
 
@@ -2218,10 +2226,19 @@ def fit(self, X, y=None):
                                                        self.subsample))
 
         X = self._check_inputs(X)
+        n_samples = X.shape[0]
+
+        if self.n_quantiles > n_samples:
+            warnings.warn("n_quantiles (%s) is greater than the total number "
+                          "of samples (%s). n_quantiles is set to "
+                          "n_samples."
+                          % (self.n_quantiles, n_samples))
+        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
+
         rng = check_random_state(self.random_state)
 
         # Create the quantiles of reference
-        self.references_ = np.linspace(0, 1, self.n_quantiles,
+        self.references_ = np.linspace(0, 1, self.n_quantiles_,
                                        endpoint=True)
         if sparse.issparse(X):
             self._sparse_fit(X, rng)
@@ -2443,9 +2460,13 @@ def quantile_transform(X, axis=0, n_quantiles=1000,
         Axis used to compute the means and standard deviations along. If 0,
         transform each feature, otherwise (if 1) transform each sample.
 
-    n_quantiles : int, optional (default=1000)
+    n_quantiles : int, optional (default=1000 or n_samples)
         Number of quantiles to be computed. It corresponds to the number
         of landmarks used to discretize the cumulative distribution function.
+        If n_quantiles is larger than the number of samples, n_quantiles is set
+        to the number of samples as a larger number of quantiles does not give
+        a better approximation of the cumulative distribution function
+        estimator.
 
     output_distribution : str, optional (default='uniform')
         Marginal distribution for the transformed data. The choices are

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/preprocessing/data.py | 427 | 427 | - | 1 | -
| sklearn/preprocessing/data.py | 595 | 595 | - | 1 | -
| sklearn/preprocessing/data.py | 2044 | 2044 | 1 | 1 | 831
| sklearn/preprocessing/data.py | 2075 | 2075 | 1 | 1 | 831
| sklearn/preprocessing/data.py | 2221 | 2221 | 2 | 1 | 1178
| sklearn/preprocessing/data.py | 2446 | 2446 | 3 | 1 | 2229


## Problem Statement

```
DOC Improve doc of n_quantiles in QuantileTransformer 
#### Description
The `QuantileTransformer` uses numpy.percentile(X_train, .) as the estimator of the quantile function of the training data. To know this function perfectly we just need to take `n_quantiles=n_samples`. Then it is just a linear interpolation (which is done in the code afterwards). Therefore I don't think we should be able to choose `n_quantiles > n_samples` and we should prevent users from thinking that the higher `n_quantiles` the better the transformation. As mentioned by @GaelVaroquaux IRL it is however true that it can be relevant to choose `n_quantiles < n_samples` when `n_samples` is very large.

I suggest to add more information on the impact of `n_quantiles` in the doc which currently reads:
\`\`\`python
Number of quantiles to be computed. It corresponds to the number of
landmarks used to discretize the cumulative distribution function.
\`\`\`

For example using 100 times more landmarks result in the same transformation
\`\`\`python
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.testing import assert_allclose

n_samples = 100
X_train = np.random.randn(n_samples, 2)
X_test = np.random.randn(1000, 2)

qf_1 = QuantileTransformer(n_quantiles=n_samples)
qf_1.fit(X_train)
X_trans_1 = qf_1.transform(X_test)

qf_2 = QuantileTransformer(n_quantiles=10000)
qf_2.fit(X_train)
X_trans_2 = qf_2.transform(X_test)

assert_allclose(X_trans_1, X_trans_2)
\`\`\`

Interestingly if you do not choose `n_quantiles > n_samples` correctly, the linear interpolation done afterwards does not correspond to the numpy.percentile(X_train, .) estimator. This is not "wrong" as these are only estimators of the true quantile function/cdf but I think it is confusing and would be better to stick with the original estimator. For instance, the following raises an AssertionError.
\`\`\`python
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.testing import assert_allclose

n_samples = 100
X_train = np.random.randn(n_samples, 2)
X_test = np.random.randn(1000, 2)

qf_1 = QuantileTransformer(n_quantiles=n_samples)
qf_1.fit(X_train)
X_trans_1 = qf_1.transform(X_test)

qf_2 = QuantileTransformer(n_quantiles=200)
qf_2.fit(X_train)
X_trans_2 = qf_2.transform(X_test)

assert_allclose(X_trans_1, X_trans_2)
\`\`\`

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/preprocessing/data.py** | 2022 | 2109| 831 | 831 | 24716 | 
| **-> 2 <-** | **1 sklearn/preprocessing/data.py** | 2189 | 2231| 347 | 1178 | 24716 | 
| **-> 3 <-** | **1 sklearn/preprocessing/data.py** | 2412 | 2526| 1051 | 2229 | 24716 | 
| 4 | **1 sklearn/preprocessing/data.py** | 2233 | 2300| 677 | 2906 | 24716 | 
| 5 | **1 sklearn/preprocessing/data.py** | 2323 | 2331| 119 | 3025 | 24716 | 
| 6 | **1 sklearn/preprocessing/data.py** | 2302 | 2321| 215 | 3240 | 24716 | 
| 7 | **1 sklearn/preprocessing/data.py** | 2111 | 2144| 293 | 3533 | 24716 | 
| 8 | 2 examples/preprocessing/plot_all_scaling.py | 311 | 356| 353 | 3886 | 27824 | 
| 9 | **2 sklearn/preprocessing/data.py** | 2146 | 2187| 387 | 4273 | 27824 | 
| 10 | **2 sklearn/preprocessing/data.py** | 2366 | 2385| 161 | 4434 | 27824 | 
| 11 | 3 sklearn/utils/estimator_checks.py | 976 | 1046| 628 | 5062 | 49736 | 
| 12 | **3 sklearn/preprocessing/data.py** | 2387 | 2409| 183 | 5245 | 49736 | 
| 13 | **3 sklearn/preprocessing/data.py** | 2333 | 2364| 227 | 5472 | 49736 | 
| 14 | 4 examples/compose/plot_transformed_target.py | 99 | 178| 746 | 6218 | 51550 | 
| 15 | 4 examples/preprocessing/plot_all_scaling.py | 219 | 310| 815 | 7033 | 51550 | 
| 16 | 4 examples/preprocessing/plot_all_scaling.py | 1 | 105| 784 | 7817 | 51550 | 
| 17 | 5 examples/preprocessing/plot_map_data_to_normal.py | 1 | 98| 719 | 8536 | 52785 | 
| 18 | 5 sklearn/utils/estimator_checks.py | 951 | 973| 248 | 8784 | 52785 | 
| 19 | **5 sklearn/preprocessing/data.py** | 1138 | 1194| 492 | 9276 | 52785 | 
| 20 | 5 sklearn/utils/estimator_checks.py | 1152 | 1220| 601 | 9877 | 52785 | 
| 21 | 5 examples/compose/plot_transformed_target.py | 1 | 98| 743 | 10620 | 52785 | 
| 22 | 6 examples/ensemble/plot_gradient_boosting_quantile.py | 1 | 80| 548 | 11168 | 53333 | 
| 23 | **6 sklearn/preprocessing/data.py** | 2832 | 2875| 388 | 11556 | 53333 | 
| 24 | 7 sklearn/datasets/samples_generator.py | 1414 | 1506| 698 | 12254 | 67337 | 
| 25 | **7 sklearn/preprocessing/data.py** | 2638 | 2665| 293 | 12547 | 67337 | 
| 26 | 8 benchmarks/bench_random_projections.py | 44 | 63| 122 | 12669 | 69072 | 
| 27 | 9 sklearn/model_selection/_split.py | 1820 | 1867| 483 | 13152 | 88466 | 
| 28 | 10 examples/preprocessing/plot_function_transformer.py | 1 | 73| 450 | 13602 | 88916 | 
| 29 | 10 sklearn/utils/estimator_checks.py | 2256 | 2291| 401 | 14003 | 88916 | 
| 30 | 10 examples/preprocessing/plot_all_scaling.py | 182 | 216| 381 | 14384 | 88916 | 
| 31 | 11 examples/cluster/plot_color_quantization.py | 1 | 103| 798 | 15182 | 89768 | 
| 32 | **11 sklearn/preprocessing/data.py** | 2878 | 2971| 832 | 16014 | 89768 | 
| 33 | 12 examples/compose/plot_column_transformer_mixed_types.py | 1 | 104| 799 | 16813 | 90589 | 
| 34 | 13 sklearn/ensemble/gradient_boosting.py | 708 | 731| 169 | 16982 | 111830 | 
| 35 | 13 benchmarks/bench_random_projections.py | 85 | 251| 1264 | 18246 | 111830 | 
| 36 | 14 sklearn/model_selection/_validation.py | 1243 | 1299| 590 | 18836 | 124899 | 
| 37 | 14 sklearn/ensemble/gradient_boosting.py | 733 | 759| 243 | 19079 | 124899 | 
| 38 | 15 examples/preprocessing/plot_discretization_classification.py | 1 | 89| 793 | 19872 | 126745 | 
| 39 | **15 sklearn/preprocessing/data.py** | 376 | 411| 235 | 20107 | 126745 | 
| 40 | 15 sklearn/utils/estimator_checks.py | 937 | 948| 143 | 20250 | 126745 | 
| 41 | 16 examples/preprocessing/plot_discretization.py | 1 | 87| 779 | 21029 | 127554 | 
| 42 | 16 sklearn/utils/estimator_checks.py | 2212 | 2253| 399 | 21428 | 127554 | 
| 43 | 17 examples/ensemble/plot_feature_transformation.py | 1 | 83| 740 | 22168 | 128729 | 
| 44 | **17 sklearn/preprocessing/data.py** | 1047 | 1136| 845 | 23013 | 128729 | 
| 45 | 18 sklearn/preprocessing/_function_transformer.py | 105 | 116| 127 | 23140 | 129994 | 
| 46 | 19 examples/cluster/plot_kmeans_assumptions.py | 1 | 65| 511 | 23651 | 130527 | 
| 47 | 19 sklearn/utils/estimator_checks.py | 1424 | 1523| 940 | 24591 | 130527 | 
| 48 | 19 sklearn/preprocessing/_function_transformer.py | 138 | 184| 254 | 24845 | 130527 | 
| 49 | 19 sklearn/utils/estimator_checks.py | 1546 | 1618| 655 | 25500 | 130527 | 
| 50 | 20 examples/model_selection/plot_cv_indices.py | 105 | 150| 381 | 25881 | 131842 | 
| 51 | 20 sklearn/utils/estimator_checks.py | 184 | 199| 174 | 26055 | 131842 | 
| 52 | **20 sklearn/preprocessing/data.py** | 2529 | 2613| 739 | 26794 | 131842 | 
| 53 | 21 sklearn/preprocessing/imputation.py | 249 | 292| 438 | 27232 | 134780 | 
| 54 | 22 sklearn/isotonic.py | 338 | 376| 287 | 27519 | 138092 | 
| 55 | 23 examples/neighbors/plot_nca_classification.py | 1 | 89| 727 | 28246 | 138827 | 
| 56 | 24 examples/text/plot_hashing_vs_dict_vectorizer.py | 1 | 110| 794 | 29040 | 139638 | 
| 57 | 25 sklearn/compose/_column_transformer.py | 283 | 304| 210 | 29250 | 146087 | 
| 58 | **25 sklearn/preprocessing/data.py** | 774 | 815| 292 | 29542 | 146087 | 
| 59 | 26 examples/preprocessing/plot_scaling_importance.py | 1 | 81| 754 | 30296 | 147306 | 
| 60 | 26 sklearn/preprocessing/_function_transformer.py | 118 | 136| 119 | 30415 | 147306 | 
| 61 | 27 sklearn/ensemble/_gb_losses.py | 457 | 484| 234 | 30649 | 154233 | 
| 62 | 27 examples/preprocessing/plot_map_data_to_normal.py | 99 | 138| 477 | 31126 | 154233 | 
| 63 | 28 benchmarks/bench_20newsgroups.py | 1 | 97| 778 | 31904 | 155011 | 
| 64 | 28 sklearn/ensemble/_gb_losses.py | 435 | 455| 135 | 32039 | 155011 | 
| 65 | 28 sklearn/ensemble/gradient_boosting.py | 64 | 122| 448 | 32487 | 155011 | 
| 66 | 29 benchmarks/bench_sample_without_replacement.py | 34 | 202| 1229 | 33716 | 156403 | 
| 67 | 29 sklearn/preprocessing/imputation.py | 169 | 247| 675 | 34391 | 156403 | 
| 68 | 30 sklearn/impute.py | 1 | 49| 298 | 34689 | 167081 | 
| 69 | 31 examples/model_selection/plot_train_error_vs_test_error.py | 1 | 76| 612 | 35301 | 167718 | 
| 70 | 32 sklearn/kernel_approximation.py | 410 | 439| 284 | 35585 | 172962 | 
| 71 | 33 examples/calibration/plot_calibration.py | 1 | 82| 764 | 36349 | 174203 | 
| 72 | 34 benchmarks/bench_plot_nmf.py | 370 | 422| 514 | 36863 | 178099 | 
| 73 | 35 examples/ensemble/plot_gradient_boosting_early_stopping.py | 1 | 103| 772 | 37635 | 179263 | 
| 74 | 35 sklearn/kernel_approximation.py | 384 | 408| 231 | 37866 | 179263 | 
| 75 | 36 examples/model_selection/plot_precision_recall.py | 101 | 205| 770 | 38636 | 181605 | 
| 76 | 36 sklearn/model_selection/_split.py | 1609 | 1670| 758 | 39394 | 181605 | 
| 77 | 36 examples/preprocessing/plot_discretization_classification.py | 109 | 193| 874 | 40268 | 181605 | 
| 78 | 37 examples/plot_johnson_lindenstrauss_bound.py | 94 | 168| 709 | 40977 | 183448 | 
| 79 | 37 sklearn/isotonic.py | 6 | 74| 524 | 41501 | 183448 | 
| 80 | 37 examples/compose/plot_transformed_target.py | 179 | 205| 297 | 41798 | 183448 | 
| 81 | 38 examples/ensemble/plot_gradient_boosting_oob.py | 1 | 95| 793 | 42591 | 184693 | 
| 82 | 38 benchmarks/bench_plot_nmf.py | 230 | 277| 530 | 43121 | 184693 | 
| 83 | 39 examples/plot_anomaly_comparison.py | 81 | 153| 763 | 43884 | 186217 | 
| 84 | 39 sklearn/ensemble/_gb_losses.py | 503 | 511| 121 | 44005 | 186217 | 
| 85 | **39 sklearn/preprocessing/data.py** | 1219 | 1243| 178 | 44183 | 186217 | 
| 86 | **39 sklearn/preprocessing/data.py** | 965 | 985| 152 | 44335 | 186217 | 
| 87 | 39 examples/preprocessing/plot_scaling_importance.py | 82 | 134| 457 | 44792 | 186217 | 
| 88 | 40 examples/neighbors/plot_lof_outlier_detection.py | 1 | 69| 667 | 45459 | 186884 | 
| 89 | 40 sklearn/utils/estimator_checks.py | 625 | 658| 312 | 45771 | 186884 | 
| 90 | 41 examples/text/plot_document_classification_20newsgroups.py | 121 | 200| 689 | 46460 | 189423 | 
| 91 | 41 sklearn/datasets/samples_generator.py | 222 | 254| 368 | 46828 | 189423 | 
| 92 | 42 examples/calibration/plot_compare_calibration.py | 1 | 77| 755 | 47583 | 190612 | 
| 93 | 43 benchmarks/bench_tree.py | 64 | 125| 523 | 48106 | 191482 | 
| 94 | 44 sklearn/compose/_target.py | 107 | 149| 442 | 48548 | 193444 | 
| 95 | 45 examples/preprocessing/plot_discretization_strategies.py | 1 | 96| 787 | 49335 | 194248 | 
| 96 | 46 sklearn/cluster/mean_shift_.py | 30 | 84| 474 | 49809 | 198024 | 
| 97 | 47 examples/datasets/plot_random_dataset.py | 1 | 68| 645 | 50454 | 198669 | 
| 98 | **47 sklearn/preprocessing/data.py** | 677 | 741| 568 | 51022 | 198669 | 


### Hint

```
When you say prevent, do you mean that we should raise an error if
n_quantiles > n_samples, or that we should adjust n_quantiles to
min(n_quantiles, n_samples)? I'd be in favour of the latter, perhaps with a
warning. And yes, improved documentation is always good (albeit often
ignored).

I was only talking about the documentation but yes we should also change the behavior when n_quantiles > n_samples, which will require a deprecation cycle... Ideally the default of n_quantiles should be n_samples. And if too slow users can choose a n_quantiles value smaller than n_samples.
I don't think the second behavior (when `n_quantiles=200`, which leads to a linear interpolation that does not correspond to the numpy.percentile(X_train, .) estimator) is the intended behavior. Unless someone tells me there is a valid reason behind it.
> Therefore I don't think we should be able to choose n_quantiles > n_samples and we should prevent users from thinking that the higher n_quantiles the better the transformation.

+1 for dynamically downgrading n_quantiles to "self.n_quantiles_ = min(n_quantiles, n_samples)" maybe with a warning.

However, -1 for raising an error: people might not know in advance what the sample is.


Sounds good! I will open a PR.
```

## Patch

```diff
diff --git a/sklearn/preprocessing/data.py b/sklearn/preprocessing/data.py
--- a/sklearn/preprocessing/data.py
+++ b/sklearn/preprocessing/data.py
@@ -424,7 +424,7 @@ def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
         X_scaled = X_std * (max - min) + min
 
     where min, max = feature_range.
- 
+
     The transformation is calculated as (when ``axis=0``)::
 
        X_scaled = scale * X + min - X.min(axis=0) * scale
@@ -592,7 +592,7 @@ class StandardScaler(BaseEstimator, TransformerMixin):
     -----
     NaNs are treated as missing values: disregarded in fit, and maintained in
     transform.
-    
+
     We use a biased estimator for the standard deviation, equivalent to
     `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
     affect model performance.
@@ -2041,9 +2041,13 @@ class QuantileTransformer(BaseEstimator, TransformerMixin):
 
     Parameters
     ----------
-    n_quantiles : int, optional (default=1000)
+    n_quantiles : int, optional (default=1000 or n_samples)
         Number of quantiles to be computed. It corresponds to the number
         of landmarks used to discretize the cumulative distribution function.
+        If n_quantiles is larger than the number of samples, n_quantiles is set
+        to the number of samples as a larger number of quantiles does not give
+        a better approximation of the cumulative distribution function
+        estimator.
 
     output_distribution : str, optional (default='uniform')
         Marginal distribution for the transformed data. The choices are
@@ -2072,6 +2076,10 @@ class QuantileTransformer(BaseEstimator, TransformerMixin):
 
     Attributes
     ----------
+    n_quantiles_ : integer
+        The actual number of quantiles used to discretize the cumulative
+        distribution function.
+
     quantiles_ : ndarray, shape (n_quantiles, n_features)
         The values corresponding the quantiles of reference.
 
@@ -2218,10 +2226,19 @@ def fit(self, X, y=None):
                                                        self.subsample))
 
         X = self._check_inputs(X)
+        n_samples = X.shape[0]
+
+        if self.n_quantiles > n_samples:
+            warnings.warn("n_quantiles (%s) is greater than the total number "
+                          "of samples (%s). n_quantiles is set to "
+                          "n_samples."
+                          % (self.n_quantiles, n_samples))
+        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))
+
         rng = check_random_state(self.random_state)
 
         # Create the quantiles of reference
-        self.references_ = np.linspace(0, 1, self.n_quantiles,
+        self.references_ = np.linspace(0, 1, self.n_quantiles_,
                                        endpoint=True)
         if sparse.issparse(X):
             self._sparse_fit(X, rng)
@@ -2443,9 +2460,13 @@ def quantile_transform(X, axis=0, n_quantiles=1000,
         Axis used to compute the means and standard deviations along. If 0,
         transform each feature, otherwise (if 1) transform each sample.
 
-    n_quantiles : int, optional (default=1000)
+    n_quantiles : int, optional (default=1000 or n_samples)
         Number of quantiles to be computed. It corresponds to the number
         of landmarks used to discretize the cumulative distribution function.
+        If n_quantiles is larger than the number of samples, n_quantiles is set
+        to the number of samples as a larger number of quantiles does not give
+        a better approximation of the cumulative distribution function
+        estimator.
 
     output_distribution : str, optional (default='uniform')
         Marginal distribution for the transformed data. The choices are

```

## Test Patch

```diff
diff --git a/sklearn/preprocessing/tests/test_data.py b/sklearn/preprocessing/tests/test_data.py
--- a/sklearn/preprocessing/tests/test_data.py
+++ b/sklearn/preprocessing/tests/test_data.py
@@ -1260,6 +1260,13 @@ def test_quantile_transform_check_error():
     assert_raise_message(ValueError,
                          'Expected 2D array, got scalar array instead',
                          transformer.transform, 10)
+    # check that a warning is raised is n_quantiles > n_samples
+    transformer = QuantileTransformer(n_quantiles=100)
+    warn_msg = "n_quantiles is set to n_samples"
+    with pytest.warns(UserWarning, match=warn_msg) as record:
+        transformer.fit(X)
+    assert len(record) == 1
+    assert transformer.n_quantiles_ == X.shape[0]
 
 
 def test_quantile_transform_sparse_ignore_zeros():

```


## Code snippets

### 1 - sklearn/preprocessing/data.py:

Start line: 2022, End line: 2109

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    n_quantiles : int, optional (default=1000)
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.

    output_distribution : str, optional (default='uniform')
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, optional (default=False)
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, optional (default=1e5)
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Note that this is used by subsampling and smoothing
        noise.

    copy : boolean, optional, (default=True)
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    quantiles_ : ndarray, shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray, shape(n_quantiles, )
        Quantiles of references.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> qt = QuantileTransformer(n_quantiles=10, random_state=0)
    >>> qt.fit_transform(X) # doctest: +ELLIPSIS
    array([...])

    See also
    --------
    quantile_transform : Equivalent function without the estimator API.
    PowerTransformer : Perform mapping to a normal distribution using a power
        transform.
    StandardScaler : Perform standardization that is faster, but less robust
        to outliers.
    RobustScaler : Perform robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
```
### 2 - sklearn/preprocessing/data.py:

Start line: 2189, End line: 2231

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        self : object
        """
        if self.n_quantiles <= 0:
            raise ValueError("Invalid value for 'n_quantiles': %d. "
                             "The number of quantiles must be at least one."
                             % self.n_quantiles)

        if self.subsample <= 0:
            raise ValueError("Invalid value for 'subsample': %d. "
                             "The number of subsamples must be at least one."
                             % self.subsample)

        if self.n_quantiles > self.subsample:
            raise ValueError("The number of quantiles cannot be greater than"
                             " the number of samples used. Got {} quantiles"
                             " and {} samples.".format(self.n_quantiles,
                                                       self.subsample))

        X = self._check_inputs(X)
        rng = check_random_state(self.random_state)

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles,
                                       endpoint=True)
        if sparse.issparse(X):
            self._sparse_fit(X, rng)
        else:
            self._dense_fit(X, rng)

        return self
```
### 3 - sklearn/preprocessing/data.py:

Start line: 2412, End line: 2526

```python
def quantile_transform(X, axis=0, n_quantiles=1000,
                       output_distribution='uniform',
                       ignore_implicit_zeros=False,
                       subsample=int(1e5),
                       random_state=None,
                       copy=False):
    """Transform features using quantiles information.

    This method transforms the features to follow a uniform or a normal
    distribution. Therefore, for a given feature, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.

    The transformation is applied on each feature independently. First an
    estimate of the cumulative distribution function of a feature is
    used to map the original values to a uniform distribution. The obtained
    values are then mapped to the desired output distribution using the
    associated quantile function. Features values of new/unseen data that fall
    below or above the fitted range will be mapped to the bounds of the output
    distribution. Note that this transform is non-linear. It may distort linear
    correlations between variables measured at the same scale but renders
    variables measured at different scales more directly comparable.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like, sparse matrix
        The data to transform.

    axis : int, (default=0)
        Axis used to compute the means and standard deviations along. If 0,
        transform each feature, otherwise (if 1) transform each sample.

    n_quantiles : int, optional (default=1000)
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.

    output_distribution : str, optional (default='uniform')
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    ignore_implicit_zeros : bool, optional (default=False)
        Only applies to sparse matrices. If True, the sparse entries of the
        matrix are discarded to compute the quantile statistics. If False,
        these entries are treated as zeros.

    subsample : int, optional (default=1e5)
        Maximum number of samples used to estimate the quantiles for
        computational efficiency. Note that the subsampling procedure may
        differ for value-identical sparse and dense matrices.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Note that this is used by subsampling and smoothing
        noise.

    copy : boolean, optional, (default=True)
        Set to False to perform inplace transformation and avoid a copy (if the
        input is already a numpy array).

    Attributes
    ----------
    quantiles_ : ndarray, shape (n_quantiles, n_features)
        The values corresponding the quantiles of reference.

    references_ : ndarray, shape(n_quantiles, )
        Quantiles of references.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import quantile_transform
    >>> rng = np.random.RandomState(0)
    >>> X = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    >>> quantile_transform(X, n_quantiles=10, random_state=0)
    ... # doctest: +ELLIPSIS
    array([...])

    See also
    --------
    QuantileTransformer : Performs quantile-based scaling using the
        ``Transformer`` API (e.g. as part of a preprocessing
        :class:`sklearn.pipeline.Pipeline`).
    power_transform : Maps data to a normal distribution using a
        power transformation.
    scale : Performs standardization that is faster, but less robust
        to outliers.
    robust_scale : Performs robust standardization that removes the influence
        of outliers but does not put outliers and inliers on the same scale.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """
    n = QuantileTransformer(n_quantiles=n_quantiles,
                            output_distribution=output_distribution,
                            subsample=subsample,
                            ignore_implicit_zeros=ignore_implicit_zeros,
                            random_state=random_state,
                            copy=copy)
    if axis == 0:
        return n.fit_transform(X)
    elif axis == 1:
        return n.fit_transform(X.T).T
    else:
        raise ValueError("axis should be either equal to 0 or 1. Got"
                         " axis={}".format(axis))
```
### 4 - sklearn/preprocessing/data.py:

Start line: 2233, End line: 2300

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def _transform_col(self, X_col, quantiles, inverse):
        """Private function to transform a single feature"""

        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            #  for inverse transform, match a uniform distribution
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if output_distribution == 'normal':
                lower_bounds_idx = (X_col - BOUNDS_THRESHOLD <
                                    lower_bound_x)
                upper_bounds_idx = (X_col + BOUNDS_THRESHOLD >
                                    upper_bound_x)
            if output_distribution == 'uniform':
                lower_bounds_idx = (X_col == lower_bound_x)
                upper_bounds_idx = (X_col == upper_bound_x)

        isfinite_mask = ~np.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col[isfinite_mask] = .5 * (
                np.interp(X_col_finite, quantiles, self.references_)
                - np.interp(-X_col_finite, -quantiles[::-1],
                            -self.references_[::-1]))
        else:
            X_col[isfinite_mask] = np.interp(X_col_finite,
                                             self.references_, quantiles)

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    X_col = stats.norm.ppf(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = stats.norm.ppf(1 - (BOUNDS_THRESHOLD -
                                                   np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col
```
### 5 - sklearn/preprocessing/data.py:

Start line: 2323, End line: 2331

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def _check_is_fitted(self, X):
        """Check the inputs before transforming"""
        check_is_fitted(self, 'quantiles_')
        # check that the dimension of X are adequate with the fitted data
        if X.shape[1] != self.quantiles_.shape[1]:
            raise ValueError('X does not have the same number of features as'
                             ' the previously fitted data. Got {} instead of'
                             ' {}.'.format(X.shape[1],
                                           self.quantiles_.shape[1]))
```
### 6 - sklearn/preprocessing/data.py:

Start line: 2302, End line: 2321

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def _check_inputs(self, X, accept_sparse_negative=False):
        """Check inputs before fit and transform"""
        X = check_array(X, accept_sparse='csc', copy=self.copy,
                        dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')
        # we only accept positive sparse matrix when ignore_implicit_zeros is
        # false and that we call fit or transform.
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if (not accept_sparse_negative and not self.ignore_implicit_zeros
                    and (sparse.issparse(X) and np.any(X.data < 0))):
                raise ValueError('QuantileTransformer only accepts'
                                 ' non-negative sparse matrices.')

        # check the output distribution
        if self.output_distribution not in ('normal', 'uniform'):
            raise ValueError("'output_distribution' has to be either 'normal'"
                             " or 'uniform'. Got '{}' instead.".format(
                                 self.output_distribution))

        return X
```
### 7 - sklearn/preprocessing/data.py:

Start line: 2111, End line: 2144

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, n_quantiles=1000, output_distribution='uniform',
                 ignore_implicit_zeros=False, subsample=int(1e5),
                 random_state=None, copy=True):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy

    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        if self.ignore_implicit_zeros:
            warnings.warn("'ignore_implicit_zeros' takes effect only with"
                          " sparse matrix. This parameter has no effect.")

        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for col in X.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(n_samples,
                                                    size=self.subsample,
                                                    replace=False)
                col = col.take(subsample_idx, mode='clip')
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
```
### 8 - examples/preprocessing/plot_all_scaling.py:

Start line: 311, End line: 356

```python
make_plot(6)

##############################################################################
# QuantileTransformer (Gaussian output)
# -------------------------------------
#
# ``QuantileTransformer`` has an additional ``output_distribution`` parameter
# allowing to match a Gaussian distribution instead of a uniform distribution.
# Note that this non-parametetric transformer introduces saturation artifacts
# for extreme values.

make_plot(7)

###################################################################
# QuantileTransformer (uniform output)
# ------------------------------------
#
# ``QuantileTransformer`` applies a non-linear transformation such that the
# probability density function of each feature will be mapped to a uniform
# distribution. In this case, all the data will be mapped in the range [0, 1],
# even the outliers which cannot be distinguished anymore from the inliers.
#
# As ``RobustScaler``, ``QuantileTransformer`` is robust to outliers in the
# sense that adding or removing outliers in the training set will yield
# approximately the same transformation on held out data. But contrary to
# ``RobustScaler``, ``QuantileTransformer`` will also automatically collapse
# any outlier by setting them to the a priori defined range boundaries (0 and
# 1).

make_plot(8)

##############################################################################
# Normalizer
# ----------
#
# The ``Normalizer`` rescales the vector for each sample to have unit norm,
# independently of the distribution of the samples. It can be seen on both
# figures below where all samples are mapped onto the unit circle. In our
# example the two selected features have only positive values; therefore the
# transformed data only lie in the positive quadrant. This would not be the
# case if some original features had a mix of positive and negative values.

make_plot(9)

plt.show()
```
### 9 - sklearn/preprocessing/data.py:

Start line: 2146, End line: 2187

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def _sparse_fit(self, X, random_state):
        """Compute percentiles for sparse matrices.

        Parameters
        ----------
        X : sparse matrix CSC, shape (n_samples, n_features)
            The data used to scale along the features axis. The sparse matrix
            needs to be nonnegative.
        """
        n_samples, n_features = X.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for feature_idx in range(n_features):
            column_nnz_data = X.data[X.indptr[feature_idx]:
                                     X.indptr[feature_idx + 1]]
            if len(column_nnz_data) > self.subsample:
                column_subsample = (self.subsample * len(column_nnz_data) //
                                    n_samples)
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=column_subsample,
                                           dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=self.subsample, dtype=X.dtype)
                column_data[:column_subsample] = random_state.choice(
                    column_nnz_data, size=column_subsample, replace=False)
            else:
                if self.ignore_implicit_zeros:
                    column_data = np.zeros(shape=len(column_nnz_data),
                                           dtype=X.dtype)
                else:
                    column_data = np.zeros(shape=n_samples, dtype=X.dtype)
                column_data[:len(column_nnz_data)] = column_nnz_data

            if not column_data.size:
                # if no nnz, an error will be raised for computing the
                # quantiles. Force the quantiles to be zeros.
                self.quantiles_.append([0] * len(references))
            else:
                self.quantiles_.append(
                        np.nanpercentile(column_data, references))
        self.quantiles_ = np.transpose(self.quantiles_)
```
### 10 - sklearn/preprocessing/data.py:

Start line: 2366, End line: 2385

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Feature-wise transformation of the data.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X)
        self._check_is_fitted(X)

        return self._transform(X, inverse=False)
```
### 12 - sklearn/preprocessing/data.py:

Start line: 2387, End line: 2409

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def inverse_transform(self, X):
        """Back-projection to the original space.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns
        -------
        Xt : ndarray or sparse matrix, shape (n_samples, n_features)
            The projected data.
        """
        X = self._check_inputs(X, accept_sparse_negative=True)
        self._check_is_fitted(X)

        return self._transform(X, inverse=True)

    def _more_tags(self):
        return {'allow_nan': True}
```
### 13 - sklearn/preprocessing/data.py:

Start line: 2333, End line: 2364

```python
class QuantileTransformer(BaseEstimator, TransformerMixin):

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, optional (default=False)
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            Projected data
        """

        if sparse.issparse(X):
            for feature_idx in range(X.shape[1]):
                column_slice = slice(X.indptr[feature_idx],
                                     X.indptr[feature_idx + 1])
                X.data[column_slice] = self._transform_col(
                    X.data[column_slice], self.quantiles_[:, feature_idx],
                    inverse)
        else:
            for feature_idx in range(X.shape[1]):
                X[:, feature_idx] = self._transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx],
                    inverse)

        return X
```
### 19 - sklearn/preprocessing/data.py:

Start line: 1138, End line: 1194

```python
class RobustScaler(BaseEstimator, TransformerMixin):

    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        X = check_array(X, accept_sparse='csc', copy=self.copy, estimator=self,
                        dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if self.with_centering:
            if sparse.issparse(X):
                raise ValueError(
                    "Cannot center sparse matrices: use `with_centering=False`"
                    " instead. See docstring for motivation and alternatives.")
            self.center_ = np.nanmedian(X, axis=0)
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = []
            for feature_idx in range(X.shape[1]):
                if sparse.issparse(X):
                    column_nnz_data = X.data[X.indptr[feature_idx]:
                                             X.indptr[feature_idx + 1]]
                    column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
                    column_data[:len(column_nnz_data)] = column_nnz_data
                else:
                    column_data = X[:, feature_idx]

                quantiles.append(np.nanpercentile(column_data,
                                                  self.quantile_range))

            quantiles = np.transpose(quantiles)

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        else:
            self.scale_ = None

        return self
```
### 23 - sklearn/preprocessing/data.py:

Start line: 2832, End line: 2875

```python
class PowerTransformer(BaseEstimator, TransformerMixin):

    def _check_input(self, X, check_positive=False, check_shape=False,
                     check_method=False):
        """Validate the input before fit and transform.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        check_positive : bool
            If True, check that all data is positive and non-zero (only if
            ``self.method=='box-cox'``).

        check_shape : bool
            If True, check that n_features matches the length of self.lambdas_

        check_method : bool
            If True, check that the transformation method is valid.
        """
        X = check_array(X, ensure_2d=True, dtype=FLOAT_DTYPES, copy=self.copy,
                        force_all_finite='allow-nan')

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings(
                'ignore', r'All-NaN (slice|axis) encountered')
            if (check_positive and self.method == 'box-cox' and
                    np.nanmin(X) <= 0):
                raise ValueError("The Box-Cox transformation can only be "
                                 "applied to strictly positive data")

        if check_shape and not X.shape[1] == len(self.lambdas_):
            raise ValueError("Input data has a different number of features "
                             "than fitting data. Should have {n}, data has {m}"
                             .format(n=len(self.lambdas_), m=X.shape[1]))

        valid_methods = ('box-cox', 'yeo-johnson')
        if check_method and self.method not in valid_methods:
            raise ValueError("'method' must be one of {}, "
                             "got {} instead."
                             .format(valid_methods, self.method))

        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 25 - sklearn/preprocessing/data.py:

Start line: 2638, End line: 2665

```python
class PowerTransformer(BaseEstimator, TransformerMixin):

    def _fit(self, X, y=None, force_transform=False):
        X = self._check_input(X, check_positive=True, check_method=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        optim_function = {'box-cox': self._box_cox_optimize,
                          'yeo-johnson': self._yeo_johnson_optimize
                          }[self.method]
        with np.errstate(invalid='ignore'):  # hide NaN warnings
            self.lambdas_ = np.array([optim_function(col) for col in X.T])

        if self.standardize or force_transform:
            transform_function = {'box-cox': boxcox,
                                  'yeo-johnson': self._yeo_johnson_transform
                                  }[self.method]
            for i, lmbda in enumerate(self.lambdas_):
                with np.errstate(invalid='ignore'):  # hide NaN warnings
                    X[:, i] = transform_function(X[:, i], lmbda)

        if self.standardize:
            self._scaler = StandardScaler(copy=False)
            if force_transform:
                X = self._scaler.fit_transform(X)
            else:
                self._scaler.fit(X)

        return X
```
### 32 - sklearn/preprocessing/data.py:

Start line: 2878, End line: 2971

```python
def power_transform(X, method='warn', standardize=True, copy=True):
    """
    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, power_transform supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to be transformed using a power transformation.

    method : str
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

        The default method will be changed from 'box-cox' to 'yeo-johnson'
        in version 0.23. To suppress the FutureWarning, explicitly set the
        parameter.

    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.

    Returns
    -------
    X_trans : array-like, shape (n_samples, n_features)
        The transformed data.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import power_transform
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(power_transform(data, method='box-cox'))  # doctest: +ELLIPSIS
    [[-1.332... -0.707...]
     [ 0.256... -0.707...]
     [ 1.076...  1.414...]]

    See also
    --------
    PowerTransformer : Equivalent transformation with the
        ``Transformer`` API (e.g. as part of a preprocessing
        :class:`sklearn.pipeline.Pipeline`).

    quantile_transform : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """
    if method == 'warn':
        warnings.warn("The default value of 'method' will change from "
                      "'box-cox' to 'yeo-johnson' in version 0.23. Set "
                      "the 'method' argument explicitly to silence this "
                      "warning in the meantime.",
                      FutureWarning)
        method = 'box-cox'
    pt = PowerTransformer(method=method, standardize=standardize, copy=copy)
    return pt.fit_transform(X)
```
### 39 - sklearn/preprocessing/data.py:

Start line: 376, End line: 411

```python
class MinMaxScaler(BaseEstimator, TransformerMixin):

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 44 - sklearn/preprocessing/data.py:

Start line: 1047, End line: 1136

```python
class RobustScaler(BaseEstimator, TransformerMixin):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).

    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the ``transform`` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators. Typically this is done by removing the mean
    and scaling to unit variance. However, outliers can often influence the
    sample mean / variance in a negative way. In such cases, the median and
    the interquartile range often give better results.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    with_centering : boolean, True by default
        If True, center the data before scaling.
        This will cause ``transform`` to raise an exception when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_scaling : boolean, True by default
        If True, scale the data to interquartile range.

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.

        .. versionadded:: 0.18

    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.

    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.

        .. versionadded:: 0.17
           *scale_* attribute.

    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> transformer = RobustScaler().fit(X)
    >>> transformer  # doctest: +NORMALIZE_WHITESPACE
    RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True,
           with_scaling=True)
    >>> transformer.transform(X)
    array([[ 0. , -2. ,  0. ],
           [-1. ,  0. ,  0.4],
           [ 1. ,  0. , -1.6]])

    See also
    --------
    robust_scale: Equivalent function without the estimator API.

    :class:`sklearn.decomposition.PCA`
        Further removes the linear correlation across features with
        'whiten=True'.

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    https://en.wikipedia.org/wiki/Median
    https://en.wikipedia.org/wiki/Interquartile_range
    """
```
### 52 - sklearn/preprocessing/data.py:

Start line: 2529, End line: 2613

```python
class PowerTransformer(BaseEstimator, TransformerMixin):
    """Apply a power transform featurewise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Read more in the :ref:`User Guide <preprocessing_transformer>`.

    Parameters
    ----------
    method : str, (default='yeo-johnson')
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : boolean, default=True
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    copy : boolean, optional, default=True
        Set to False to perform inplace computation during transformation.

    Attributes
    ----------
    lambdas_ : array of float, shape (n_features,)
        The parameters of the power transformation for the selected features.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import PowerTransformer
    >>> pt = PowerTransformer()
    >>> data = [[1, 2], [3, 2], [4, 5]]
    >>> print(pt.fit(data))
    PowerTransformer(copy=True, method='yeo-johnson', standardize=True)
    >>> print(pt.lambdas_)
    [ 1.386... -3.100...]
    >>> print(pt.transform(data))
    [[-1.316... -0.707...]
     [ 0.209... -0.707...]
     [ 1.106...  1.414...]]

    See also
    --------
    power_transform : Equivalent function without the estimator API.

    QuantileTransformer : Maps data to a standard normal distribution with
        the parameter `output_distribution='normal'`.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

    References
    ----------

    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).

    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).
    """
    def __init__(self, method='yeo-johnson', standardize=True, copy=True):
        self.method = method
        self.standardize = standardize
        self.copy = copy
```
### 58 - sklearn/preprocessing/data.py:

Start line: 774, End line: 815

```python
class StandardScaler(BaseEstimator, TransformerMixin):

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 85 - sklearn/preprocessing/data.py:

Start line: 1219, End line: 1243

```python
class RobustScaler(BaseEstimator, TransformerMixin):

    def inverse_transform(self, X):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        check_is_fitted(self, 'center_', 'scale_')
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_scaling:
                X *= self.scale_
            if self.with_centering:
                X += self.center_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 86 - sklearn/preprocessing/data.py:

Start line: 965, End line: 985

```python
class MaxAbsScaler(BaseEstimator, TransformerMixin):

    def inverse_transform(self, X):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be transformed back.
        """
        check_is_fitted(self, 'scale_')
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            inplace_column_scale(X, self.scale_)
        else:
            X *= self.scale_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
```
### 98 - sklearn/preprocessing/data.py:

Start line: 677, End line: 741

```python
class StandardScaler(BaseEstimator, TransformerMixin):

    def partial_fit(self, X, y=None):
        # ... other code

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")

            sparse_constructor = (sparse.csr_matrix
                                  if X.format == 'csr' else sparse.csc_matrix)
            counts_nan = sparse_constructor(
                        (np.isnan(X.data), X.indices, X.indptr),
                        shape=X.shape).sum(axis=0).A.ravel()

            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = (X.shape[0] -
                                        counts_nan).astype(np.int64)

            if self.with_std:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
                if hasattr(self, 'scale_'):
                    self.n_samples_seen_ += X.shape[0] - counts_nan
        else:
            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = np.zeros(X.shape[1], dtype=np.int64)

            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - np.isnan(X).sum(axis=0)
            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if np.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self
```

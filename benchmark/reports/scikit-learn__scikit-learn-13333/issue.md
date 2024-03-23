# scikit-learn__scikit-learn-13333

* repo: scikit-learn/scikit-learn
* base_commit: `04a5733b86bba57a48520b97b9c0a5cd325a1b9a`

## Problem statement

DOC Improve doc of n_quantiles in QuantileTransformer 
#### Description
The `QuantileTransformer` uses numpy.percentile(X_train, .) as the estimator of the quantile function of the training data. To know this function perfectly we just need to take `n_quantiles=n_samples`. Then it is just a linear interpolation (which is done in the code afterwards). Therefore I don't think we should be able to choose `n_quantiles > n_samples` and we should prevent users from thinking that the higher `n_quantiles` the better the transformation. As mentioned by @GaelVaroquaux IRL it is however true that it can be relevant to choose `n_quantiles < n_samples` when `n_samples` is very large.

I suggest to add more information on the impact of `n_quantiles` in the doc which currently reads:
```python
Number of quantiles to be computed. It corresponds to the number of
landmarks used to discretize the cumulative distribution function.
```

For example using 100 times more landmarks result in the same transformation
```python
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
```

Interestingly if you do not choose `n_quantiles > n_samples` correctly, the linear interpolation done afterwards does not correspond to the numpy.percentile(X_train, .) estimator. This is not "wrong" as these are only estimators of the true quantile function/cdf but I think it is confusing and would be better to stick with the original estimator. For instance, the following raises an AssertionError.
```python
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

# scikit-learn__scikit-learn-23099

| **scikit-learn/scikit-learn** | `42d235924efa64987a19e945035c85414c53d4f0` |
| ---- | ---- |
| **No of patches** | 1 |
| **All found context length** | 15694 |
| **Any found context length** | 536 |
| **Avg pos** | 121.0 |
| **Min pos** | 1 |
| **Max pos** | 31 |
| **Top file pos** | 1 |
| **Missing snippets** | 6 |
| **Missing patch files** | 0 |


## Expected patch

```diff
diff --git a/sklearn/gaussian_process/_gpr.py b/sklearn/gaussian_process/_gpr.py
--- a/sklearn/gaussian_process/_gpr.py
+++ b/sklearn/gaussian_process/_gpr.py
@@ -110,6 +110,14 @@ def optimizer(obj_func, initial_theta, bounds):
         which might cause predictions to change if the data is modified
         externally.
 
+    n_targets : int, default=None
+        The number of dimensions of the target values. Used to decide the number
+        of outputs when sampling from the prior distributions (i.e. calling
+        :meth:`sample_y` before :meth:`fit`). This parameter is ignored once
+        :meth:`fit` has been called.
+
+        .. versionadded:: 1.3
+
     random_state : int, RandomState instance or None, default=None
         Determines random number generation used to initialize the centers.
         Pass an int for reproducible results across multiple function calls.
@@ -181,6 +189,7 @@ def optimizer(obj_func, initial_theta, bounds):
         "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
         "normalize_y": ["boolean"],
         "copy_X_train": ["boolean"],
+        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
         "random_state": ["random_state"],
     }
 
@@ -193,6 +202,7 @@ def __init__(
         n_restarts_optimizer=0,
         normalize_y=False,
         copy_X_train=True,
+        n_targets=None,
         random_state=None,
     ):
         self.kernel = kernel
@@ -201,6 +211,7 @@ def __init__(
         self.n_restarts_optimizer = n_restarts_optimizer
         self.normalize_y = normalize_y
         self.copy_X_train = copy_X_train
+        self.n_targets = n_targets
         self.random_state = random_state
 
     def fit(self, X, y):
@@ -243,6 +254,13 @@ def fit(self, X, y):
             dtype=dtype,
         )
 
+        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
+        if self.n_targets is not None and n_targets_seen != self.n_targets:
+            raise ValueError(
+                "The number of targets seen in `y` is different from the parameter "
+                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
+            )
+
         # Normalize target value
         if self.normalize_y:
             self._y_train_mean = np.mean(y, axis=0)
@@ -393,12 +411,23 @@ def predict(self, X, return_std=False, return_cov=False):
                 )
             else:
                 kernel = self.kernel
-            y_mean = np.zeros(X.shape[0])
+
+            n_targets = self.n_targets if self.n_targets is not None else 1
+            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()
+
             if return_cov:
                 y_cov = kernel(X)
+                if n_targets > 1:
+                    y_cov = np.repeat(
+                        np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
+                    )
                 return y_mean, y_cov
             elif return_std:
                 y_var = kernel.diag(X)
+                if n_targets > 1:
+                    y_var = np.repeat(
+                        np.expand_dims(y_var, -1), repeats=n_targets, axis=-1
+                    )
                 return y_mean, np.sqrt(y_var)
             else:
                 return y_mean

```

## Expected file changes

| File | Start line | End line | Found on position | Found file position | Context length |
| ---- | ---------- | -------- | ----------------- | ------------------- | -------------- |
| sklearn/gaussian_process/_gpr.py | 113 | 113 | 5 | 1 | 3461
| sklearn/gaussian_process/_gpr.py | 184 | 184 | 31 | 1 | 15694
| sklearn/gaussian_process/_gpr.py | 196 | 196 | 31 | 1 | 15694
| sklearn/gaussian_process/_gpr.py | 204 | 204 | 31 | 1 | 15694
| sklearn/gaussian_process/_gpr.py | 246 | 246 | 1 | 1 | 536
| sklearn/gaussian_process/_gpr.py | 396 | 396 | 22 | 1 | 10955


## Problem Statement

```
GPR `sample_y` enforce `n_targets=1` before calling `fit`
In `GaussianProcessRegressor`, sampling in the prior before calling `fit` via `sample_y` will assume that `y` is made of a single target. However, this is not necessarily the case. Therefore, the shape of the output of `sample_y` before and after `fit` is different.

In order to solve this inconsistency, we need to introduce a new parameter `n_targets=None`. Before calling `fit` this parameter should be explicitly set by the user. After `fit`, we can use the information of the target seen during `fit` without explicitly setting the parameter.

```

## Retrieved code snippets

| Position | File | Start line | End line | Tokens | Sum tokens | File tokens |
| -------- | ---- | ---------- | -------- | ------ | ---------- | ----------- |
| **-> 1 <-** | **1 sklearn/gaussian_process/_gpr.py** | 206 | 269| 536 | 536 | 5927 | 
| 2 | 2 sklearn/naive_bayes.py | 242 | 273| 243 | 779 | 18989 | 
| 3 | 3 examples/gaussian_process/plot_gpr_noisy_targets.py | 110 | 154| 353 | 1132 | 20254 | 
| 4 | 3 examples/gaussian_process/plot_gpr_noisy_targets.py | 1 | 109| 792 | 1924 | 20254 | 
| **-> 5 <-** | **3 sklearn/gaussian_process/_gpr.py** | 27 | 175| 1537 | 3461 | 20254 | 
| 6 | **3 sklearn/gaussian_process/_gpr.py** | 271 | 340| 658 | 4119 | 20254 | 
| 7 | 4 sklearn/multioutput.py | 985 | 1014| 191 | 4310 | 28090 | 
| 8 | **4 sklearn/gaussian_process/_gpr.py** | 466 | 503| 326 | 4636 | 28090 | 
| 9 | 5 sklearn/utils/multiclass.py | 198 | 222| 184 | 4820 | 32809 | 
| 10 | 6 sklearn/gaussian_process/_gpc.py | 682 | 756| 542 | 5362 | 40833 | 
| 11 | 6 sklearn/multioutput.py | 427 | 466| 319 | 5681 | 40833 | 
| 12 | 6 sklearn/multioutput.py | 569 | 645| 662 | 6343 | 40833 | 
| 13 | 7 sklearn/kernel_approximation.py | 341 | 386| 428 | 6771 | 50201 | 
| 14 | 7 sklearn/gaussian_process/_gpc.py | 172 | 268| 788 | 7559 | 50201 | 
| 15 | 8 sklearn/compose/_target.py | 200 | 269| 523 | 8082 | 52713 | 
| 16 | 8 sklearn/naive_bayes.py | 395 | 481| 798 | 8880 | 52713 | 
| 17 | 9 sklearn/cross_decomposition/_pls.py | 443 | 471| 245 | 9125 | 61796 | 
| 18 | 10 sklearn/preprocessing/_target_encoder.py | 179 | 197| 125 | 9250 | 64846 | 
| 19 | 10 sklearn/naive_bayes.py | 731 | 790| 522 | 9772 | 64846 | 
| 20 | 10 sklearn/compose/_target.py | 306 | 333| 195 | 9967 | 64846 | 
| 21 | 11 examples/gaussian_process/plot_gpr_prior_posterior.py | 240 | 259| 208 | 10175 | 67174 | 
| **-> 22 <-** | **11 sklearn/gaussian_process/_gpr.py** | 389 | 464| 780 | 10955 | 67174 | 
| 23 | 11 sklearn/multioutput.py | 261 | 328| 558 | 11513 | 67174 | 
| 24 | 11 examples/gaussian_process/plot_gpr_prior_posterior.py | 82 | 162| 767 | 12280 | 67174 | 
| 25 | 11 examples/gaussian_process/plot_gpr_prior_posterior.py | 163 | 239| 755 | 13035 | 67174 | 
| 26 | 12 sklearn/linear_model/_glm/glm.py | 171 | 262| 781 | 13816 | 74709 | 
| 27 | 13 sklearn/neighbors/_regression.py | 219 | 260| 350 | 14166 | 78698 | 
| 28 | 13 examples/gaussian_process/plot_gpr_prior_posterior.py | 37 | 79| 337 | 14503 | 78698 | 
| 29 | 14 examples/gaussian_process/plot_gpr_noisy.py | 1 | 102| 768 | 15271 | 80404 | 
| 30 | 14 sklearn/multioutput.py | 330 | 352| 182 | 15453 | 80404 | 
| **-> 31 <-** | **14 sklearn/gaussian_process/_gpr.py** | 177 | 204| 241 | 15694 | 80404 | 
| 32 | 15 sklearn/neural_network/_multilayer_perceptron.py | 730 | 1253| 216 | 15910 | 93909 | 
| 33 | 15 sklearn/naive_bayes.py | 1373 | 1398| 244 | 16154 | 93909 | 
| 34 | 15 sklearn/neighbors/_regression.py | 446 | 502| 363 | 16517 | 93909 | 
| 35 | 16 sklearn/ensemble/_bagging.py | 1198 | 1240| 285 | 16802 | 103385 | 
| 36 | 17 sklearn/ensemble/_gb.py | 1772 | 1791| 168 | 16970 | 119165 | 
| 37 | 17 sklearn/multioutput.py | 797 | 819| 142 | 17112 | 119165 | 
| 38 | 17 sklearn/compose/_target.py | 5 | 129| 1016 | 18128 | 119165 | 
| 39 | 17 sklearn/multioutput.py | 169 | 231| 484 | 18612 | 119165 | 
| 40 | 18 sklearn/datasets/_samples_generator.py | 647 | 693| 608 | 19220 | 136506 | 
| 41 | 19 sklearn/linear_model/_ransac.py | 300 | 398| 797 | 20017 | 141414 | 
| 42 | 19 sklearn/naive_bayes.py | 483 | 509| 246 | 20263 | 141414 | 
| 43 | 19 sklearn/ensemble/_gb.py | 1213 | 1226| 147 | 20410 | 141414 | 
| 44 | **19 sklearn/gaussian_process/_gpr.py** | 622 | 642| 178 | 20588 | 141414 | 
| 45 | 20 examples/gaussian_process/plot_compare_gpr_krr.py | 344 | 396| 373 | 20961 | 144495 | 
| 46 | 21 sklearn/linear_model/_quantile.py | 144 | 235| 814 | 21775 | 147178 | 
| 47 | 22 sklearn/linear_model/_omp.py | 728 | 792| 507 | 22282 | 156542 | 
| 48 | 22 sklearn/kernel_approximation.py | 496 | 537| 390 | 22672 | 156542 | 
| 49 | 22 sklearn/linear_model/_ransac.py | 432 | 516| 683 | 23355 | 156542 | 
| 50 | 23 sklearn/linear_model/_least_angle.py | 1100 | 1347| 375 | 23730 | 176413 | 
| 51 | 24 examples/compose/plot_transformed_target.py | 1 | 95| 740 | 24470 | 178356 | 
| 52 | **24 sklearn/gaussian_process/_gpr.py** | 342 | 387| 434 | 24904 | 178356 | 
| 53 | 24 examples/gaussian_process/plot_gpr_noisy.py | 103 | 193| 833 | 25737 | 178356 | 
| 54 | 24 sklearn/linear_model/_glm/glm.py | 263 | 319| 499 | 26236 | 178356 | 
| 55 | 24 sklearn/ensemble/_gb.py | 134 | 201| 554 | 26790 | 178356 | 
| 56 | 24 sklearn/multioutput.py | 355 | 425| 591 | 27381 | 178356 | 


### Hint

```
I see that we have the same issue with `predict` indeed.
```

## Patch

```diff
diff --git a/sklearn/gaussian_process/_gpr.py b/sklearn/gaussian_process/_gpr.py
--- a/sklearn/gaussian_process/_gpr.py
+++ b/sklearn/gaussian_process/_gpr.py
@@ -110,6 +110,14 @@ def optimizer(obj_func, initial_theta, bounds):
         which might cause predictions to change if the data is modified
         externally.
 
+    n_targets : int, default=None
+        The number of dimensions of the target values. Used to decide the number
+        of outputs when sampling from the prior distributions (i.e. calling
+        :meth:`sample_y` before :meth:`fit`). This parameter is ignored once
+        :meth:`fit` has been called.
+
+        .. versionadded:: 1.3
+
     random_state : int, RandomState instance or None, default=None
         Determines random number generation used to initialize the centers.
         Pass an int for reproducible results across multiple function calls.
@@ -181,6 +189,7 @@ def optimizer(obj_func, initial_theta, bounds):
         "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
         "normalize_y": ["boolean"],
         "copy_X_train": ["boolean"],
+        "n_targets": [Interval(Integral, 1, None, closed="left"), None],
         "random_state": ["random_state"],
     }
 
@@ -193,6 +202,7 @@ def __init__(
         n_restarts_optimizer=0,
         normalize_y=False,
         copy_X_train=True,
+        n_targets=None,
         random_state=None,
     ):
         self.kernel = kernel
@@ -201,6 +211,7 @@ def __init__(
         self.n_restarts_optimizer = n_restarts_optimizer
         self.normalize_y = normalize_y
         self.copy_X_train = copy_X_train
+        self.n_targets = n_targets
         self.random_state = random_state
 
     def fit(self, X, y):
@@ -243,6 +254,13 @@ def fit(self, X, y):
             dtype=dtype,
         )
 
+        n_targets_seen = y.shape[1] if y.ndim > 1 else 1
+        if self.n_targets is not None and n_targets_seen != self.n_targets:
+            raise ValueError(
+                "The number of targets seen in `y` is different from the parameter "
+                f"`n_targets`. Got {n_targets_seen} != {self.n_targets}."
+            )
+
         # Normalize target value
         if self.normalize_y:
             self._y_train_mean = np.mean(y, axis=0)
@@ -393,12 +411,23 @@ def predict(self, X, return_std=False, return_cov=False):
                 )
             else:
                 kernel = self.kernel
-            y_mean = np.zeros(X.shape[0])
+
+            n_targets = self.n_targets if self.n_targets is not None else 1
+            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()
+
             if return_cov:
                 y_cov = kernel(X)
+                if n_targets > 1:
+                    y_cov = np.repeat(
+                        np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
+                    )
                 return y_mean, y_cov
             elif return_std:
                 y_var = kernel.diag(X)
+                if n_targets > 1:
+                    y_var = np.repeat(
+                        np.expand_dims(y_var, -1), repeats=n_targets, axis=-1
+                    )
                 return y_mean, np.sqrt(y_var)
             else:
                 return y_mean

```

## Test Patch

```diff
diff --git a/sklearn/gaussian_process/tests/test_gpr.py b/sklearn/gaussian_process/tests/test_gpr.py
--- a/sklearn/gaussian_process/tests/test_gpr.py
+++ b/sklearn/gaussian_process/tests/test_gpr.py
@@ -773,6 +773,57 @@ def test_sample_y_shapes(normalize_y, n_targets):
     assert y_samples.shape == y_test_shape
 
 
+@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
+@pytest.mark.parametrize("n_samples", [1, 5])
+def test_sample_y_shape_with_prior(n_targets, n_samples):
+    """Check the output shape of `sample_y` is consistent before and after `fit`."""
+    rng = np.random.RandomState(1024)
+
+    X = rng.randn(10, 3)
+    y = rng.randn(10, n_targets if n_targets is not None else 1)
+
+    model = GaussianProcessRegressor(n_targets=n_targets)
+    shape_before_fit = model.sample_y(X, n_samples=n_samples).shape
+    model.fit(X, y)
+    shape_after_fit = model.sample_y(X, n_samples=n_samples).shape
+    assert shape_before_fit == shape_after_fit
+
+
+@pytest.mark.parametrize("n_targets", [None, 1, 2, 3])
+def test_predict_shape_with_prior(n_targets):
+    """Check the output shape of `predict` with prior distribution."""
+    rng = np.random.RandomState(1024)
+
+    n_sample = 10
+    X = rng.randn(n_sample, 3)
+    y = rng.randn(n_sample, n_targets if n_targets is not None else 1)
+
+    model = GaussianProcessRegressor(n_targets=n_targets)
+    mean_prior, cov_prior = model.predict(X, return_cov=True)
+    _, std_prior = model.predict(X, return_std=True)
+
+    model.fit(X, y)
+    mean_post, cov_post = model.predict(X, return_cov=True)
+    _, std_post = model.predict(X, return_std=True)
+
+    assert mean_prior.shape == mean_post.shape
+    assert cov_prior.shape == cov_post.shape
+    assert std_prior.shape == std_post.shape
+
+
+def test_n_targets_error():
+    """Check that an error is raised when the number of targets seen at fit is
+    inconsistent with n_targets.
+    """
+    rng = np.random.RandomState(0)
+    X = rng.randn(10, 3)
+    y = rng.randn(10, 2)
+
+    model = GaussianProcessRegressor(n_targets=1)
+    with pytest.raises(ValueError, match="The number of targets seen in `y`"):
+        model.fit(X, y)
+
+
 class CustomKernel(C):
     """
     A custom kernel that has a diag method that returns the first column of the

```


## Code snippets

### 1 - sklearn/gaussian_process/_gpr.py:

Start line: 206, End line: 269

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def fit(self, X, y):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance.
        """
        self._validate_params()

        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") * RBF(
                1.0, length_scale_bounds="fixed"
            )
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False
        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            y_numeric=True,
            ensure_2d=ensure_2d,
            dtype=dtype,
        )

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), copy=False)

            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std

        else:
            shape_y_stats = (y.shape[1],) if y.ndim == 2 else 1
            self._y_train_mean = np.zeros(shape=shape_y_stats)
            self._y_train_std = np.ones(shape=shape_y_stats)

        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError(
                    "alpha must be a scalar or an array with same number of "
                    f"entries as y. ({self.alpha.shape[0]} != {y.shape[0]})"
                )

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        # ... other code
```
### 2 - sklearn/naive_bayes.py:

Start line: 242, End line: 273

```python
class GaussianNB(_BaseNB):

    def fit(self, X, y, sample_weight=None):
        """Fit Gaussian Naive Bayes according to X, y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()
        y = self._validate_data(y=y)
        return self._partial_fit(
            X, y, np.unique(y), _refit=True, sample_weight=sample_weight
        )

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, reset=False)
```
### 3 - examples/gaussian_process/plot_gpr_noisy_targets.py:

Start line: 110, End line: 154

```python
y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)

# %%
# We create a similar Gaussian process model. In addition to the kernel, this
# time, we specify the parameter `alpha` which can be interpreted as the
# variance of a Gaussian noise.
gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9
)
gaussian_process.fit(X_train, y_train_noisy)
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

# %%
# Let's plot the mean prediction and the uncertainty region as before.
plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.errorbar(
    X_train,
    y_train_noisy,
    noise_std,
    linestyle="None",
    color="tab:blue",
    marker=".",
    markersize=10,
    label="Observations",
)
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    color="tab:orange",
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on a noisy dataset")

# %%
# The noise affects the predictions close to the training samples: the
# predictive uncertainty near to the training samples is larger because we
# explicitly model a given level target noise independent of the input
# variable.
```
### 4 - examples/gaussian_process/plot_gpr_noisy_targets.py:

Start line: 1, End line: 109

```python
"""
=========================================================
Gaussian Processes regression: basic introductory example
=========================================================

A simple one-dimensional regression example computed in two different ways:

1. A noise-free case
2. A noisy case with known noise-level per datapoint

In both cases, the kernel's parameters are estimated using the maximum
likelihood principle.

The figures illustrate the interpolating property of the Gaussian Process model
as well as its probabilistic nature in the form of a pointwise 95% confidence
interval.

Note that `alpha` is a parameter to control the strength of the Tikhonov
regularization on the assumed training points' covariance matrix.
"""
import numpy as np

X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

# %%
import matplotlib.pyplot as plt

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")

# %%
# We will use this dataset in the next experiment to illustrate how Gaussian
# Process regression is working.
#
# Example with noise-free target
# ------------------------------
#
# In this first example, we will use the true generative process without
# adding any noise. For training the Gaussian Process regression, we will only
# select few samples.
rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

# %%
# Now, we fit a Gaussian process on these few training data samples. We will
# use a radial basis function (RBF) kernel and a constant parameter to fit the
# amplitude.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train, y_train)
gaussian_process.kernel_

# %%
# After fitting our model, we see that the hyperparameters of the kernel have
# been optimized. Now, we will use our kernel to compute the mean prediction
# of the full dataset and plot the 95% confidence interval.
mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
plt.scatter(X_train, y_train, label="Observations")
plt.plot(X, mean_prediction, label="Mean prediction")
plt.fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"95% confidence interval",
)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")

# %%
# We see that for a prediction made on a data point close to the one from the
# training set, the 95% confidence has a small amplitude. Whenever a sample
# falls far from training data, our model's prediction is less accurate and the
# model prediction is less precise (higher uncertainty).
#
# Example with noisy targets
# --------------------------
#
# We can repeat a similar experiment adding an additional noise to the target
# this time. It will allow seeing the effect of the noise on the fitted model.
#
# We add some random Gaussian noise to the target with an arbitrary
# standard deviation.
noise_std = 0.75
```
### 5 - sklearn/gaussian_process/_gpr.py:

Start line: 27, End line: 175

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of [RW2006]_.

    In addition to standard scikit-learn estimator API,
    :class:`GaussianProcessRegressor`:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method `sample_y(X)`, which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method `log_marginal_likelihood(theta)`, which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".

    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with :class:`~sklearn.linear_model.Ridge`.

    optimizer : "fmin_l_bfgs_b", callable or None, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func': the objective function to be minimized, which
                #   takes the hyperparameters theta as a parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are: `{'fmin_l_bfgs_b'}`.

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts_optimizer == 0` implies that one
        run is performed.

    normalize_y : bool, default=False
        Whether or not to normalize the target values `y` by removing the mean
        and scaling to unit-variance. This is recommended for cases where
        zero-mean, unit-variance priors are used. Note that, in this
        implementation, the normalisation is reversed before the GP predictions
        are reported.

        .. versionchanged:: 0.23

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values in training data (also required for prediction).

    kernel_ : kernel instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters.

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``.

    alpha_ : array-like of shape (n_samples,)
        Dual coefficients of training data points in kernel space.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianProcessClassifier : Gaussian process classification (GPC)
        based on Laplace approximation.

    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))
    """
```
### 6 - sklearn/gaussian_process/_gpr.py:

Start line: 271, End line: 340

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def fit(self, X, y):
        # ... other code

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                (
                    self._constrained_optimization(
                        obj_func, self.kernel_.theta, self.kernel_.bounds
                    )
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self
```
### 7 - sklearn/multioutput.py:

Start line: 985, End line: 1014

```python
class RegressorChain(MetaEstimatorMixin, RegressorMixin, _BaseChain):

    def fit(self, X, Y, **fit_params):
        """Fit the model to data matrix X and targets Y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Y : array-like of shape (n_samples, n_classes)
            The target values.

        **fit_params : dict of string -> object
            Parameters passed to the `fit` method at each step
            of the regressor chain.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        self._validate_params()

        super().fit(X, Y, **fit_params)
        return self

    def _more_tags(self):
        return {"multioutput_only": True}
```
### 8 - sklearn/gaussian_process/_gpr.py:

Start line: 466, End line: 503

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = [
                rng.multivariate_normal(
                    y_mean[:, target], y_cov[..., target], n_samples
                ).T[:, np.newaxis]
                for target in range(y_mean.shape[1])
            ]
            y_samples = np.hstack(y_samples)
        return y_samples
```
### 9 - sklearn/utils/multiclass.py:

Start line: 198, End line: 222

```python
def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
        Target values.
    """
    y_type = type_of_target(y, input_name="y")
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError(
            f"Unknown label type: {y_type}. Maybe you are trying to fit a "
            "classifier, which expects discrete classes on a "
            "regression target with continuous values."
        )
```
### 10 - sklearn/gaussian_process/_gpc.py:

Start line: 682, End line: 756

```python
class GaussianProcessClassifier(ClassifierMixin, BaseEstimator):

    def fit(self, X, y):
        """Fit Gaussian process classification model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,)
            Target values, must be binary.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        self._validate_params()

        if isinstance(self.kernel, CompoundKernel):
            raise ValueError("kernel cannot be a CompoundKernel")

        if self.kernel is None or self.kernel.requires_vector_input:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=True, dtype="numeric"
            )
        else:
            X, y = self._validate_data(
                X, y, multi_output=False, ensure_2d=False, dtype=None
            )

        self.base_estimator_ = _BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state,
        )

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError(
                "GaussianProcessClassifier requires 2 or more "
                "distinct classes; got %d class (only class %s "
                "is present)" % (self.n_classes_, self.classes_[0])
            )
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = OneVsRestClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = OneVsOneClassifier(
                    self.base_estimator_, n_jobs=self.n_jobs
                )
            else:
                raise ValueError("Unknown multi-class mode %s" % self.multi_class)

        self.base_estimator_.fit(X, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [
                    estimator.log_marginal_likelihood()
                    for estimator in self.base_estimator_.estimators_
                ]
            )
        else:
            self.log_marginal_likelihood_value_ = (
                self.base_estimator_.log_marginal_likelihood()
            )

        return self
```
### 22 - sklearn/gaussian_process/_gpr.py:

Start line: 389, End line: 464

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def predict(self, X, return_std=False, return_cov=False):
        # ... other code

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                    1.0, length_scale_bounds="fixed"
                )
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
                y_cov = self.kernel_(X) - V.T @ V

                # undo normalisation
                y_cov = np.outer(y_cov, self._y_train_std**2).reshape(
                    *y_cov.shape, -1
                )
                # if y_cov has shape (n_samples, n_samples, 1), reshape to
                # (n_samples, n_samples)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            elif return_std:
                # Compute variance of predictive distribution
                # Use einsum to avoid explicitly forming the large matrix
                # V^T @ V just to extract its diagonal afterward.
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum("ij,ji->i", V.T, V)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = np.outer(y_var, self._y_train_std**2).reshape(
                    *y_var.shape, -1
                )

                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
```
### 31 - sklearn/gaussian_process/_gpr.py:

Start line: 177, End line: 204

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    _parameter_constraints: dict = {
        "kernel": [None, Kernel],
        "alpha": [Interval(Real, 0, None, closed="left"), np.ndarray],
        "optimizer": [StrOptions({"fmin_l_bfgs_b"}), callable, None],
        "n_restarts_optimizer": [Interval(Integral, 0, None, closed="left")],
        "normalize_y": ["boolean"],
        "copy_X_train": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        random_state=None,
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
```
### 44 - sklearn/gaussian_process/_gpr.py:

Start line: 622, End line: 642

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min

    def _more_tags(self):
        return {"requires_fit": False}
```
### 52 - sklearn/gaussian_process/_gpr.py:

Start line: 342, End line: 387

```python
class GaussianProcessRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)
        # ... other code
```
